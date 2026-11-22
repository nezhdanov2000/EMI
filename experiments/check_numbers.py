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
    cg_rd = counts(cert, "vsf_vs_rules_disjoint")
    cp_rd = counts(cert, "partial_vs_rules_disjoint")
    cp_g = counts(cert, "partial_vs_vsf")
    ck.expect("grid vs disjoint, certified", rf"the full grid loses {cg_rd['-']} of {n_cert} cells; partial centres lose {cp_rd['-']}")
    ck.expect("7.3: ties/wins/losses", rf"ties free disjoint rules in {cp_rd['=']} of {n_cert} cells, wins {cp_rd['+']}")
    ck.expect("7.3: partial vs grid", rf"better in {cp_g['+']} of {n_cert} cells")

    # --- Holm columns
    hp_rd_o = counts(obs, "partial_vs_rules_disjoint_holm")
    hp_rd_c = counts(cert, "partial_vs_rules_disjoint_holm")
    hp_g_o = counts(obs, "partial_vs_vsf_holm")
    hp_g_c = counts(cert, "partial_vs_vsf_holm")
    ck.expect("Holm sentence", rf"lose {hp_rd_o['-']} of {n_obs} observed-purity cells and {hp_rd_c['-']} of {n_cert} certified ones to disjoint rules.*\({hp_g_o['+']} and {hp_g_c['+']} wins survive\)")

    # --- where the certified losses to disjoint rules are (7.1, contributions)
    lost = [r for r in cert if r["partial_vs_rules_disjoint"] == "-"]
    on_c4 = sum(1 for r in lost if r["dataset"] == "connect_4")
    others = sorted({(r["dataset"], r["tau"], r["budget"]) for r in lost if r["dataset"] != "connect_4"})
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    ck.expect("certified losses on connect_4", rf"lose {len(lost)} \({words[on_c4]} on \\texttt\{{connect\\_4\}}")
    for ds, tau, b in others:
        ck.expect(f"certified loss on {ds}", rf"\\texttt\{{{ds}\}} at \$\\tau={float(tau):.1f}\$, \$B={b}\$")

    # --- one cell per configuration at B = 32 (7.1)
    c32_rd, c32_g = counts(cert32, "partial_vs_rules_disjoint"), counts(cert32, "partial_vs_vsf")
    o32_rd, o32_g = counts(obs32, "partial_vs_rules_disjoint"), counts(obs32, "partial_vs_vsf")
    ck.expect("B=32 certified summary", rf"tie disjoint rules on {c32_rd['=']} of the {len(cert32)} configurations and lose {c32_rd['-']}, and beat the full grid on {c32_g['+']}")
    ck.expect("B=32 observed summary", rf"win {o32_rd['+']} of {len(obs32)}, tie {o32_rd['=']} and lose {o32_rd['-']}, and beat the full grid on {o32_g['+']}")
    ck.expect("contributions: observed-rule losses", rf"lose {p_rd['-']} of {n_obs} cells \({o32_rd['-']} of {len(obs32)} configurations at \$B=32\$\)")
    ck.expect("abstract: certified ties", rf"tie free disjoint rules in {cp_rd['=']} of {n_cert} \(configuration, budget\) cells")

    # --- gap closed
    gap_o = statistics.median(floats(obs, "gap_closed"))
    ck.expect("gap closed, observed (7.1)", rf"closes a median {pct(gap_o)}\\,\\% of the gap")

    # --- union purity reaches tau
    ok_rules_o = mean(floats(obs, "purity_ok_rules"))
    ok_any_c = mean(floats(cert, "purity_ok_partial"))
    ck.expect("union purity, certified vs overlapping observed",
              rf"reaches \$\\tau\$ in {pct(ok_any_c)}\\,\\% of splits for every method under the certificate, against {pct(ok_rules_o)}\\,\\% for overlapping rules")

    # --- why the disjoint baseline (Section 5) and the abstract's share
    ok_g, ok_p, ok_rd = (mean(floats(obs, k)) for k in ("purity_ok_vsf", "purity_ok_partial", "purity_ok_rules_disjoint"))
    lo, hi = min(pct(ok_g), pct(ok_p), pct(ok_rd)), max(pct(ok_g), pct(ok_p), pct(ok_rd))
    ck.expect("disjoint baseline shares", rf"reaches \$\\tau\$ in {pct(ok_rules_o)}\\,\\% of the splits, against {lo}--{hi}\\,\\% for")
    ck.expect("abstract: overlapping unions below tau", rf"its unions fall below \$\\tau\$ on held-out rows in {100 - int(pct(ok_rules_o))}\\,\\% of the splits")

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
    ck.expect("abstract: compactness", rf"fewer than half the columns of the free rules \({po['columns']:.1f} against {ro['columns']:.1f} at budget 32\)")
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
    main_methods = {"vsf", "vsf_partial", "vsf_greedy", "rules", "rules_pure", "rules_disjoint", "tree", "ssdpp"}
    for r in cs:
        if r["selection"] == "certified" and not r["dataset"].startswith("synthetic"):
            key = (r["dataset"], r["positive"], r["tau"], r["min_samples"])
            # the extra methods do not count: wracc_raw ignores the purity floor
            if r["method"] in main_methods:
                best[key] = max(best[key], float(r["coverage"]))
            else:
                best.setdefault(key, 0.0)
    n_real = len(best)
    n_nonempty = sum(1 for v in best.values() if v > 0.01)
    ck.expect("7.3: certified configurations on real tables", rf"Of the {n_real} certified configurations on real tables, {n_nonempty} certify anything at all")
    ck.expect("Section 4: non-empty certificate", rf"non-empty on {n_nonempty} of the {n_real} real configurations")

    # --- optimal tree (Section 7.5), on the cells where tree_opt was run
    to_o = [r for r in obs if r.get("cov_tree_opt", "") not in ("", "nan")]
    to_c = [r for r in cert if r.get("cov_tree_opt", "") not in ("", "nan")]
    if to_o and to_c:
        o32 = [r for r in to_o if r["budget"] == "32"]
        c32 = [r for r in to_c if r["budget"] == "32"]
        m_o, m_c = counts(to_o, "partial_vs_tree_opt"), counts(to_c, "partial_vs_tree_opt")
        ck.expect("tree_opt counts and coverage",
                  rf"partial centres beat it in {m_o['+']} of {len(to_o)} cells and lose {m_o['-']} \(at \$B=32\$: \${mean(floats(o32, 'cov_partial')):.2f}\$ against \${mean(floats(o32, 'cov_tree_opt')):.2f}\$ mean coverage\); under the certificate they win {m_c['+']} and lose {m_c['-']} \(\${mean(floats(c32, 'cov_partial')):.2f}\$ against \${mean(floats(c32, 'cov_tree_opt')):.2f}\$\)")
        ck.expect("tree_opt net coverage",
                  rf"\(\${mean(floats(o32, 'net_tree_opt')):.2f}\$ against \${mean(floats(o32, 'net_partial')):.2f}\$ and \${mean(floats(c32, 'net_tree_opt')):.2f}\$ against \${mean(floats(c32, 'net_partial')):.2f}\$ at \$B=32\$\)")
        more = sum(1 for r in o32 if float(r["cov_tree_opt"]) - float(r["cov_tree"]) >= 0.01)
        less = sum(1 for r in o32 if float(r["cov_tree"]) - float(r["cov_tree_opt"]) >= 0.01)
        words = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
        ck.expect("tree_opt vs CART", rf"covers more on {words.get(more, str(more))} tables at \$B=32\$ and less on {words.get(less, str(less))}")

    # --- WRAcc lists (Section 7.6), on the cells where wracc was run
    w_o = [r for r in obs if r.get("cov_wracc", "") not in ("", "nan")]
    w_c = [r for r in cert if r.get("cov_wracc", "") not in ("", "nan")]
    if w_o and w_c:
        wo32 = [r for r in w_o if r["budget"] == "32"]
        mw_o, mw_c = counts(w_o, "partial_vs_wracc"), counts(w_c, "partial_vs_wracc")
        mr_o = counts(w_o, "partial_vs_wracc_raw")
        ck.expect("wracc vs max coverage", rf"\(\${mean(floats(wo32, 'cov_wracc')):.2f}\$ against \${mean(floats(wo32, 'cov_rules')):.2f}\$ at \$B=32\$ under the observed rule\)")
        ck.expect("wracc purity vs rules", rf"reaches \$\\tau\$ in \${pct(mean(floats(w_o, 'purity_ok_wracc')))}\\,\\%\$ of the splits against \${pct(mean(floats(w_o, 'purity_ok_rules')))}\\,\\%\$ for maximum coverage")
        ck.expect("partial vs wracc counts", rf"\(\${mean(floats(wo32, 'cov_wracc')):.2f}\$ against \${mean(floats(wo32, 'cov_partial')):.2f}\$; partial centres win {mw_o['+']} of {len(w_o)} cells and lose {mw_o['-']}, under the certificate {mw_c['+']} and {mw_c['-']}\)")
        ck.expect("wracc_raw", rf"\(\${mean(floats(wo32, 'cov_wracc_raw')):.2f}\$ at \$B=32\$, beating partial centres in {mr_o['-']} of {len(w_o)} cells\) with a union that reaches \$\\tau\$ in \${pct(mean(floats(w_o, 'purity_ok_wracc_raw')))}\\,\\%\$ of the splits")
        zero = sum(1 for r in wo32 if float(r["purity_ok_wracc_raw"]) == 0.0)
        ck.expect("wracc_raw zero share", rf"in none of them on {zero} of the {len(wo32)} configurations at \$B=32\$")

    # --- holdout split (Section 4)
    spl = [r for r in pw if r["selection"] == "split"]
    if spl:
        fam32 = {(r["dataset"], r["tau"]): r for r in cert32}
        spl32 = {(r["dataset"], r["tau"]): r for r in spl if r["budget"] == "32"}
        common = sorted(set(fam32) & set(spl32))
        lower = sum(1 for k in common if float(spl32[k]["cov_partial"]) < float(fam32[k]["cov_partial"]))
        words = {8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
        ck.expect("split: lower on n of the common configurations",
                  rf"on the {words.get(len(common), str(len(common)))} configurations that certify anything under the family bound, partial centres certified by the split cover less on {words.get(lower, str(lower))} \(mean \${round(100 * mean(float(spl32[k]['cov_partial']) for k in common))}\$ against \${round(100 * mean(float(fam32[k]['cov_partial']) for k in common))}\$ points at \$B=32\$")
        ok_split = mean(floats(spl, "purity_ok_partial"))
        ck.expect("conclusion: split purity shares", rf"\({pct(ok_any_c)} against {pct(ok_split)}\\,\\%\)")
        ck.expect("split: held-out purity share", rf"reaches \$\\tau\$ in \${pct(ok_split)}\\,\\%\$ of the splits for partial centres, against \${pct(ok_any_c)}\\,\\%\$ under the family bound")

    # --- exact packing (Section 7.10, 3.3)
    pk_path = RESULTS / "packing_exact.csv"
    if pk_path.exists():
        pk = read_csv(pk_path)
        pko = [r for r in pk if r["selection"] == "purity"]
        pkc = [r for r in pk if r["selection"] == "certified"]
        gp = lambda rows: 100 * mean(float(r["partial_exact"]) - float(r["partial_greedy"]) for r in rows)
        gr = lambda rows: 100 * mean(float(r["rules_disj_exact"]) - float(r["rules_disj_greedy"]) for r in rows)
        gap = lambda rows, which: 100 * mean(float(r[f"rules_disj_{which}"]) - float(r[f"partial_{which}"]) for r in rows)
        pk32 = [r for r in pko if r["budget"] == "32"]
        unsolved = sum(1 for r in pk if r["partial_exact_status"] != "optimal" or r["rules_disj_exact_status"] != "optimal")
        ck.expect("packing: cells", rf"{len(pk)} cells; every program reached proven optimality" if unsolved == 0 else rf"{len(pk)} cells")
        ck.expect("packing: certified gaps", rf"within \${gp(pkc):.2f}\$ points of the exact optimum on average for partial centres \(largest gap \${max(100 * (float(r['partial_exact']) - float(r['partial_greedy'])) for r in pkc):.1f}\$")
        ck.expect("packing: observed gaps", rf"the mean gap is \${gp(pko):.1f}\$ points for partial centres and \${gr(pko):.1f}\$ for disjoint rules")
        ck.expect("packing: exact vs greedy gap", rf"\(\${gap(pko, 'exact'):.1f}\$ against \${gap(pko, 'greedy'):.1f}\$ points on average over these cells, \${gap(pk32, 'exact'):.1f}\$ against \${gap(pk32, 'greedy'):.1f}\$ at \$B=32\$\)")
        ck.expect("3.3: packing sentence", rf"within \${gp(pkc):.2f}\$ coverage points of the optimum on average under the certificate and \${gp(pko):.1f}\$ points under the observed-purity rule, less than the \${gr(pko):.1f}\$ points")

    # --- pure-noise false certificates (Section 4)
    def thin(n: int) -> str:
        """The number as the text typesets it: a thin space every three digits (regex-escaped)."""
        return f"{n:,}".replace(",", r"\\,")

    nc = read_csv(RESULTS / "null_certificate.csv")
    any_rows = {r["method"]: r for r in nc if r["d"] == "any"}
    if "per_schema" in any_rows and "family_bonferroni" in any_rows:
        ps, fb = any_rows["per_schema"], any_rows["family_bonferroni"]
        ck.expect("noise: per-schema rate",
                  rf"false-certificate rate of {round(100 * float(ps['rate']))}\\,\\% on pure noise at \$d\\le4\$ \({thin(int(ps['runs']))} runs; 95\\,\\% interval \${round(100 * float(ps['ci95_low']))}\$--\${round(100 * float(ps['ci95_high']))}\\,\\%\$\)")
        if int(fb["false_certificates"]) == 0:
            ck.expect("noise: family rate", rf"makes none in \${thin(int(fb['runs']))}\$ runs")

    # --- power curve over N (Section 4)
    pc_path = RESULTS / "certificate_power_curve.csv"
    if pc_path.exists():
        pc = {int(r["n"]): r for r in read_csv(pc_path) if r["method"] == "family_bonferroni"}
        if {500, 1000, 2000, 5000, 20000} <= set(pc):
            def fmt(x: float) -> str:
                v = 100 * x
                return f"{v:.1f}" if 99 <= v < 100 else f"{int(v + 0.5 + 1e-9)}"  # 99.5 stays, 56.5 rounds up
            ck.expect("power curve", rf"{fmt(float(pc[500]['power']))}\\,\\% at \$N=500\$, {fmt(float(pc[1000]['power']))}\\,\\% at \$N=1\\,000\$, {fmt(float(pc[2000]['power']))}\\,\\% at \$N=2\\,000\$")
            false_total = sum(round(float(r["fwer"]) * int(r["runs"])) for r in pc.values())
            total = sum(int(r["runs"]) for r in pc.values())
            words = {0: "no", 1: "one", 2: "two", 3: "three"}
            ck.expect("power curve: false certificates", rf"with {words.get(false_total, str(false_total))} false certificates? in the \${thin(total)}\$ runs")

    # --- permutation null (Related work, Section 4)
    pn_path = RESULTS / "permutation_null.csv"
    if pn_path.exists():
        pn = read_csv(pn_path)
        if all(float(r["wy_flagged_share"]) == 1.0 and float(r["certified_share"]) == 0.0 and float(r["wy_adjusted_p_median"]) < 0.01 for r in pn):
            ck.expect("permutation null", r"adjusted \$p<0.01\$ in every run, certificate empty in every run")

    return ck.report()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", type=Path, default=PAPER / "main.tex")
    args = ap.parse_args()
    return run(args.tex)


if __name__ == "__main__":
    raise SystemExit(main())
