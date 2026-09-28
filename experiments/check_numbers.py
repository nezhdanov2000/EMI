"""
Every number the paper quotes from the comparison, recomputed from the csv
files and looked up in paper/main.tex.

Standard library only, so it runs wherever the paper is compiled. Each
check recomputes one quantity from `experiments/results/pairwise_partial.csv`
(written by pairwise_partial.py), `comparison_summary.csv`
(summarize_comparison.py) or the generated tables under `paper/tables/`,
renders it the way the text states it, and searches main.tex for that
statement (a regular expression with the number substituted). A check
fails when the statement is absent: either the text is stale or the
statement was reworded - in both cases the pattern below is the place to
look. Exit status 1 on any failure, so `run_all.sh` stops on a stale paper.

Add a check whenever a new number enters the text; remove it when the
sentence goes. Keep patterns short: the number and a few surrounding words.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PAPER = ROOT / "paper"

Row = Dict[str, str]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def read_csv(path: Path) -> List[Row]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def counts(rows: Iterable[Row], column: str) -> Counter:
    return Counter(r[column] for r in rows if r.get(column, "") != "")


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else float("nan")


def floats(rows: Iterable[Row], column: str) -> List[float]:
    return [float(r[column]) for r in rows if r.get(column, "") not in ("", "nan")]


def n_configs(rows: Iterable[Row]) -> int:
    return len({(r["dataset"], r["positive"], r["tau"], r["m"]) for r in rows})


def parse_compactness(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    """{(selection, method label): {coverage, conditions, groups, columns}} from the generated table."""
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in line or line.startswith("\\") or line.startswith("selection"):
            continue
        cells = [c.strip() for c in line.rstrip("\\").split("&")]
        if len(cells) != 7:
            continue
        sel, method, _, cov, cond, groups, cols = cells
        out[(sel, method)] = {
            "coverage": float(cov), "conditions": float(cond), "groups": float(groups),
            "columns": float("nan") if cols == "--" else float(cols),
        }
    return out


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self, tex: str) -> None:
        self.tex = tex
        self.failures: List[str] = []
        self.passed = 0

    def expect(self, label: str, pattern: str) -> None:
        """`pattern` is a regular expression; whitespace in it matches any run of whitespace."""
        rx = re.compile(r"\s+".join(pattern.split()), re.S)
        if rx.search(self.tex):
            self.passed += 1
        else:
            self.failures.append(f"{label}: expected the text to contain /{pattern}/")

    def report(self) -> int:
        for f in self.failures:
            print("FAIL", f)
        print(f"{self.passed} statements found, {len(self.failures)} missing")
        return 1 if self.failures else 0


def pct(x: float) -> str:
    return f"{round(100 * x):d}"


def run(tex_path: Path) -> int:
    tex = tex_path.read_text(encoding="utf-8")
    ck = Checker(tex)
    pw = read_csv(RESULTS / "pairwise_partial.csv")
    obs = [r for r in pw if r["selection"] == "purity"]
    cert = [r for r in pw if r["selection"] == "certified"]
    obs32 = [r for r in obs if r["budget"] == "32"]
    cert32 = [r for r in cert if r["budget"] == "32"]
    n_obs, n_cert = len(obs), len(cert)
    c_obs, c_cert = n_configs(obs), n_configs(cert)

    # --- cells and configurations
    ck.expect("informative configs", rf"{c_obs} observed-purity and {c_cert} certified configurations")
    ck.expect("certified cells in 7.3", rf"\({n_cert} cells\)")

    # --- counts table statements (Section 7.1, abstract, conclusion)
    g_rd = counts(obs, "vsf_vs_rules_disjoint")
    p_rd = counts(obs, "partial_vs_rules_disjoint")
    p_g = counts(obs, "partial_vs_vsf")
    ck.expect("grid vs disjoint, observed", rf"loses to free disjoint rules in {g_rd['-']} of {n_obs} cells and wins {g_rd['+']}")
    ck.expect("partial vs disjoint, observed", rf"loses {p_rd['-']} and wins {p_rd['+']}")
    ck.expect("conclusion: grid losses", rf"loses {g_rd['-']} of {n_obs} observed-purity cells and {counts(cert, 'vsf_vs_rules_disjoint')['-']} of {n_cert} certified ones")
    cg_rd = counts(cert, "vsf_vs_rules_disjoint")
    cp_rd = counts(cert, "partial_vs_rules_disjoint")
    cp_g = counts(cert, "partial_vs_vsf")
    ck.expect("grid vs disjoint, certified", rf"the full grid loses {cg_rd['-']} of {n_cert} cells; partial centres lose {cp_rd['-']}")
    ck.expect("abstract: certified counts", rf"{cp_rd['+']} wins, {cp_rd['-']} losses, {cp_rd['=']} ties over {n_cert} configuration--budget cells")
    ck.expect("7.3: ties/wins/losses", rf"ties free disjoint rules in {cp_rd['=']} of {n_cert} cells, wins {cp_rd['+']}")
    ck.expect("7.3: partial vs grid", rf"better in {cp_g['+']} of {n_cert} cells")

    # --- Holm columns
    hp_rd_o = counts(obs, "partial_vs_rules_disjoint_holm")
    hp_rd_c = counts(cert, "partial_vs_rules_disjoint_holm")
    hp_g_o = counts(obs, "partial_vs_vsf_holm")
    hp_g_c = counts(cert, "partial_vs_vsf_holm")
    ck.expect("Holm sentence", rf"still lose {hp_rd_o['-']} observed-purity cells and {hp_rd_c['-']} certified ones to disjoint rules.*\({hp_g_o['+']} and {hp_g_c['+']} wins survive\)")

    # --- gap closed
    gap_o = statistics.median(floats(obs, "gap_closed"))
    ck.expect("gap closed, observed (7.1)", rf"closes a median {pct(gap_o)}\\,\\% of the gap")
    ck.expect("gap closed, observed (abstract)", rf"closes a median {pct(gap_o)}\\,\\% of the gap under the observed-purity rule")

    # --- union purity reaches tau
    ok_rules_o = mean(floats(obs, "purity_ok_rules"))
    ok_any_c = mean(floats(cert, "purity_ok_partial"))
    ck.expect("union purity, certified vs overlapping observed",
              rf"reaches \$\\tau\$ in {pct(ok_any_c)}\\,\\% of splits for every method under the certificate, against {pct(ok_rules_o)}\\,\\% for overlapping rules")

    # --- net coverage means over informative cells
    def net_means(rows: Sequence[Row]) -> Dict[str, float]:
        return {k: mean(floats(rows, k)) for k in ("net_partial", "net_vsf", "net_rules_disjoint", "net_rules", "net_tree")}
    nc = net_means(cert)
    no = net_means(obs)
    ck.expect("net means, certified",
              rf"\${nc['net_partial']:.2f}\$ for partial centres, \${nc['net_vsf']:.2f}\$ for the full grid, \${nc['net_rules_disjoint']:.2f}\$ for disjoint and for overlapping rules and \${nc['net_tree']:.2f}\$ for the tree")
    ck.expect("net means, observed",
              rf"\${no['net_vsf']:.2f}\$ for the full grid and for partial centres, \${no['net_rules_disjoint']:.2f}\$ for disjoint rules and the tree and \${no['net_rules']:.2f}\$ for overlapping rules")
    p_net = counts(obs, "partial_vs_rules_net")
    ck.expect("net: partial vs overlapping", rf"beat overlapping rules in {p_net['+']} of {n_obs} cells and lose {p_net['-']}")

    # --- tree
    t_o, t_c = counts(obs, "partial_vs_tree"), counts(cert, "partial_vs_tree")
    t_o32, t_c32 = counts(obs32, "partial_vs_tree"), counts(cert32, "partial_vs_tree")
    ck.expect("tree, observed", rf"beat it \(paired \$\|t\|\\ge2\$\) in {t_o['+']} of {n_obs} cells and lose {t_o['-']} \(at \$B=32\$: {t_o32['+']} wins, {t_o32['=']} ties, no loss\)")
    ck.expect("tree, certified", rf"under the certificate {t_c['+']} / {t_c['=']} / {t_c['-']} \(at \$B=32\$: {t_c32['+']} / {t_c32['=']} / {t_c32['-']}\)")

    # --- greedy chain
    gr_o, gr_c = counts(obs, "vsf_vs_greedy"), counts(cert, "vsf_vs_greedy")
    gap_go = 100 * mean(float(r["cov_vsf"]) - float(r["cov_greedy"]) for r in obs32)
    gap_gc = 100 * mean(float(r["cov_vsf"]) - float(r["cov_greedy"]) for r in cert32)
    ck.expect("greedy chain", rf"worse than the exhaustive winner in {gr_o['+']} of {n_obs} cells under the observed rule and {gr_c['+']} of {n_cert} under the certificate, never better \(mean gap at \$B=32\$: {gap_go:.1f} and {gap_gc:.1f} points\)")

    # --- SSD++
    s_o, s_c = counts(obs, "partial_vs_ssdpp"), counts(cert, "partial_vs_ssdpp")
    cov_p_o, cov_s_o = mean(floats(obs32, "cov_partial")), mean(floats(obs32, "cov_ssdpp"))
    cov_p_c, cov_s_c = mean(floats(cert32, "cov_partial")), mean(floats(cert32, "cov_ssdpp"))
    ck.expect("SSD++ counts and coverage",
              rf"beat it in {s_o['+']} of {n_obs} observed-purity cells and lose {s_o['-']} \(certified: {s_c['+']} / {s_c['=']} / {s_c['-']}\); at \$B=32\$ its mean coverage is \${cov_s_o:.2f}\$ against \${cov_p_o:.2f}\$ \(certified \${cov_s_c:.2f}\$ against \${cov_p_c:.2f}\$\)")
    net_s_c, net_s_o = mean(floats(cert32, "net_ssdpp")), mean(floats(obs32, "net_ssdpp"))
    net_p_c, net_p_o = mean(floats(cert32, "net_partial")), mean(floats(obs32, "net_partial"))
    ck.expect("SSD++ net coverage", rf"\(\${net_s_c:.2f}\$ and \${net_s_o:.2f}\$ against \${net_p_c:.2f}\$ and \${net_p_o:.2f}\$ for partial centres\)")

    # --- compactness (abstract and 7.2)
    comp = parse_compactness(PAPER / "tables" / "compactness_B32.tex")
    po, ro = comp[("purity", "partial centres")], comp[("purity", r"rules$_{\mathrm{disj}}$")]
    pc, rc = comp[("certified", "partial centres")], comp[("certified", r"rules$_{\mathrm{disj}}$")]
    ov = comp[("purity", "rules")]
    ck.expect("abstract: compactness", rf"fewer conditions \({po['conditions']:.1f} against {ro['conditions']:.1f} at a budget of 32\) and fewer than half the columns \({po['columns']:.1f} against {ro['columns']:.1f}\)")
    share_cov = pct(po["coverage"] / ro["coverage"])
    share_cond = pct(po["conditions"] / ro["conditions"])
    ck.expect("7.2 observed", rf"reach {share_cov}\\,\\% of the coverage of free disjoint rules \(\${po['coverage']:.2f}\$ against \${ro['coverage']:.2f}\$\) with {share_cond}\\,\\% of the conditions \(\${po['conditions']:.1f}\$ against \${ro['conditions']:.1f}\$\) and fewer than half the columns \(\${po['columns']:.1f}\$ against \${ro['columns']:.1f}\$\)")
    words = {8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
    n_cols_ov = int(round(ov["columns"]))
    ck.expect("7.2 overlapping rules", rf"overlapping rules reach \${ov['coverage']:.2f}\$ by naming {words.get(n_cols_ov, str(n_cols_ov))} columns and spending \${int(round(ov['conditions']))}\$ conditions")
    ck.expect("7.2 certified", rf"\(\${pc['coverage']:.2f}\$ against \${rc['coverage']:.2f}\$ coverage, \${pc['conditions']:.1f}\$ against \${rc['conditions']:.1f}\$ conditions, \${pc['columns']:.1f}\$ against \${rc['columns']:.1f}\$ columns\)")

    # --- certified configurations that certify anything (comparison_summary, real tables)
    cs = read_csv(RESULTS / "comparison_summary.csv")
    best: Dict[Tuple[str, str, str, str], float] = defaultdict(float)
    for r in cs:
        if r["selection"] == "certified" and not r["dataset"].startswith("synthetic"):
            key = (r["dataset"], r["positive"], r["tau"], r["min_samples"])
            best[key] = max(best[key], float(r["coverage"]))
    n_real = len(best)
    n_nonempty = sum(1 for v in best.values() if v > 0.01)
    ck.expect("7.3: certified configurations on real tables", rf"Of the {n_real} certified configurations on real tables, {n_nonempty} certify anything at all")
    ck.expect("Section 4: non-empty certificate", rf"non-empty on {n_nonempty} of the {n_real} real configurations")

    return ck.report()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", type=Path, default=PAPER / "main.tex")
    args = ap.parse_args()
    return run(args.tex)


if __name__ == "__main__":
    raise SystemExit(main())
