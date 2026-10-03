"""LaTeX tables for paper/ from results/pairwise_partial.csv and benchmark_data/MANIFEST.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from compare_baselines import CONFIGS  # noqa: E402

OUT = ROOT / "paper" / "tables"


def esc(s: str) -> str:
    return s.replace("_", r"\_").replace(">", r"\textgreater{}").replace("<", r"\textless{}")


def coverage_table(d: pd.DataFrame, selection: str, budget: int) -> str:
    s = d[(d.selection == selection) & (d.budget == budget)].copy()
    lines = [r"\begin{tabular}{llrrrrrcc}", r"\toprule",
             r"dataset & $\tau$ & grid & partial & rules$_{\mathrm{disj}}$ & rules & tree & p$\,$vs$\,$g & p$\,$vs$\,$r\\",
             r"\midrule"]
    for _, r in s.iterrows():
        lines.append(f"{esc(r.dataset)} & {r.tau:.1f} & {100*r.cov_vsf:.0f} & {100*r.cov_partial:.0f} & "
                     f"{100*r.cov_rules_disjoint:.0f} & {100*r.cov_rules:.0f} & {100*r.cov_tree:.0f} & "
                     f"{r.partial_vs_vsf} & {r.partial_vs_rules_disjoint}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def counts_table(d: pd.DataFrame) -> str:
    """
    Wins / ties / losses per comparison. Two columns of counts: the
    descriptive mark (|t| >= 2) and the mark after a Holm correction over
    the budgets of each configuration (`pairwise_partial.py`); the Holm
    columns are empty when the csv predates them.
    """
    has_holm = "partial_vs_vsf_holm" in d.columns
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"& & \multicolumn{3}{c}{$|t|\ge2$} & \multicolumn{3}{c}{Holm within configuration}\\",
             r"selection & comparison & better & tie & worse & better & tie & worse\\", r"\midrule"]
    for sel, s in d.groupby("selection"):
        for col, name in (("vsf_vs_rules_disjoint", "full grid vs disjoint rules"),
                          ("partial_vs_rules_disjoint", "partial centres vs disjoint rules"),
                          ("partial_vs_vsf", "partial centres vs full grid")):
            c = s[col].value_counts().reindex(["+", "=", "-"], fill_value=0)
            if has_holm:
                h = s[f"{col}_holm"].value_counts().reindex(["+", "=", "-"], fill_value=0)
                holm = f"{h['+']} & {h['=']} & {h['-']}"
            else:
                holm = "-- & -- & --"
            lines.append(f"{sel} & {name} & {c['+']} & {c['=']} & {c['-']} & {holm}\\\\")
        g = s.gap_closed.dropna()
        lines.append(f"{sel} & \\multicolumn{{7}}{{l}}{{gap closed by partial centres: median {g.median():.2f}, mean {g.mean():.2f} ($n={len(g)}$)}}\\\\")
        lines.append(f"{sel} & \\multicolumn{{7}}{{l}}{{held-out union purity $\\ge\\tau$: grid {100*s.purity_ok_vsf.mean():.0f}\\,\\%, partial {100*s.purity_ok_partial.mean():.0f}\\,\\%, rules$_{{\\mathrm{{disj}}}}$ {100*s.purity_ok_rules_disjoint.mean():.0f}\\,\\%, rules {100*s.purity_ok_rules.mean():.0f}\\,\\%}}\\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def net_table(d: pd.DataFrame, budget: int) -> str:
    """
    Held-out coverage and net coverage (k - tau/(1-tau)(n-k))/P of the
    union, both in points, for the three disjoint methods, overlapping
    rules, the tree and SSD++ at one budget, both selection rules. A
    negative net coverage means the union is below tau on held-out rows.
    """
    cols = (("cov_partial", "net_partial", "partial"), ("cov_vsf", "net_vsf", "grid"),
            ("cov_rules_disjoint", "net_rules_disjoint", r"rules$_{\mathrm{disj}}$"),
            ("cov_rules", "net_rules", "rules"), ("cov_tree", "net_tree", "tree"), ("cov_ssdpp", "net_ssdpp", "SSD++"))
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, _, name in cols)
    lines = [r"\begin{tabular}{lll" + "rr" * len(cols) + "}", r"\toprule",
             r"& & & " + head + r"\\",
             r"selection & dataset & $\tau$ " + " & cov & net" * len(cols) + r"\\", r"\midrule"]
    for sel in ("purity", "certified"):
        s = d[(d.selection == sel) & (d.budget == budget)]
        for _, r in s.iterrows():
            cells = []
            for cov, net, _ in cols:
                c = getattr(r, cov, float("nan"))
                n = getattr(r, net, float("nan"))
                cells.append("--" if pd.isna(c) else f"{100*c:.0f}")
                cells.append("--" if pd.isna(n) else f"{100*n:.0f}")
            lines.append(f"{sel} & {esc(r.dataset)} & {r.tau:.1f} & " + " & ".join(cells) + r"\\")
        means = []
        for cov, net, _ in cols:
            means.append(f"{100*s[cov].mean():.0f}" if cov in s else "--")
            means.append(f"{100*s[net].mean():.0f}" if net in s else "--")
        lines.append(rf"{sel} & \emph{{mean}} & & " + " & ".join(means) + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def split_table(d: pd.DataFrame, budget: int) -> str:
    """
    Family certificate against the exploratory/holdout split on the same
    folds: held-out coverage (points) of the full grid, partial centres,
    disjoint rules and the tree at one budget under each certificate, and
    the share of splits whose held-out union purity reaches tau for partial
    centres. "--" where a configuration was not run under that rule.
    """
    cols = (("cov_vsf", "grid"), ("cov_partial", "partial"), ("cov_rules_disjoint", r"rules$_{\mathrm{disj}}$"), ("cov_tree", "tree"))
    fam = d[(d.selection == "certified") & (d.budget == budget)].set_index(["dataset", "tau"])
    spl = d[(d.selection == "split") & (d.budget == budget)].set_index(["dataset", "tau"])
    keys = sorted(set(fam.index) | set(spl.index))
    head = " & ".join(name for _, name in cols)
    lines = [r"\begin{tabular}{ll" + "r" * (2 * len(cols) + 2) + "}", r"\toprule",
             rf"& & \multicolumn{{{len(cols) + 1}}}{{c}}{{family certificate}} & \multicolumn{{{len(cols) + 1}}}{{c}}{{holdout split}}\\",
             rf"dataset & $\tau$ & {head} & $\ge\tau$ & {head} & $\ge\tau$\\", r"\midrule"]

    def cells(frame: pd.DataFrame, key) -> str:
        if key not in frame.index:
            return " & ".join(["--"] * (len(cols) + 1))
        r = frame.loc[key]
        out = [f"{100 * float(r[c]):.0f}" for c, _ in cols]
        ok = r["purity_ok_partial"]
        out.append("--" if pd.isna(ok) else f"{100 * float(ok):.0f}")
        return " & ".join(out)

    for key in keys:
        lines.append(f"{esc(key[0])} & {float(key[1]):.1f} & {cells(fam, key)} & {cells(spl, key)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tree_opt_table(d: pd.DataFrame, budget: int) -> str:
    """
    Optimal tree (DL8.5, net-coverage weights) against partial centres and
    CART at one budget: held-out coverage and net coverage in points, the
    paired mark of partial centres against the optimal tree, and the share
    of splits on which a DL8.5 fit hit its time limit. Rows: configurations
    where tree_opt was run.
    """
    lines = [r"\begin{tabular}{lllrrrrrrcr}", r"\toprule",
             r"& & & \multicolumn{3}{c}{coverage} & \multicolumn{3}{c}{net coverage} & &\\",
             r"selection & dataset & $\tau$ & partial & opt.\ tree & CART & partial & opt.\ tree & CART & p$\,$vs$\,$opt & timeouts\\",
             r"\midrule"]
    for sel in ("purity", "certified"):
        s = d[(d.selection == sel) & (d.budget == budget) & d["cov_tree_opt"].notna()]
        for _, r in s.iterrows():
            lines.append(f"{sel} & {esc(r.dataset)} & {r.tau:.1f} & {100*r.cov_partial:.0f} & {100*r.cov_tree_opt:.0f} & {100*r.cov_tree:.0f} & "
                         f"{100*r.net_partial:.0f} & {100*r.net_tree_opt:.0f} & {100*r.net_tree:.0f} & {r.partial_vs_tree_opt} & "
                         f"{100*float(r.tree_opt_timeouts):.0f}\\%\\\\")
        lines.append(rf"{sel} & \emph{{mean}} & & {100*s.cov_partial.mean():.0f} & {100*s.cov_tree_opt.mean():.0f} & {100*s.cov_tree.mean():.0f} & "
                     f"{100*s.net_partial.mean():.0f} & {100*s.net_tree_opt.mean():.0f} & {100*s.net_tree.mean():.0f} & &\\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def data_table() -> str:
    m = pd.read_csv(ROOT / "benchmark_data" / "MANIFEST.csv")
    cfg = pd.DataFrame([c.__dict__ for c in CONFIGS])
    lines = [r"\begin{tabular}{lrrlrl}", r"\toprule", r"dataset & $N$ & $M$ & positive value & base rate & $\tau$ / $m$ / repeats\\", r"\midrule"]
    for ds, g in cfg.groupby("dataset", sort=False):
        row = m[m.file == f"{ds}.csv"].iloc[0]
        frame = pd.read_csv(row.path if "path" in row else ROOT / "benchmark_data" / row.file, dtype=str)
        pos = g.positive.iloc[0]
        base = (frame[row.target] == pos).mean()
        settings = "; ".join(sorted({f"{r.tau}/{r.min_samples}/{r.n_repeats}" for r in g.itertuples()}))
        lines.append(f"{esc(ds)} & {row.rows} & {row.columns - 1} & {esc(str(pos))} & {100*base:.0f}\\,\\% & {settings}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "experiments" / "results" / "pairwise_partial.csv")
    (OUT / "coverage_certified_B32.tex").write_text(coverage_table(d, "certified", 32) + "\n")
    (OUT / "coverage_purity_B32.tex").write_text(coverage_table(d, "purity", 32) + "\n")
    (OUT / "coverage_purity_B8.tex").write_text(coverage_table(d, "purity", 8) + "\n")
    (OUT / "counts.tex").write_text(counts_table(d) + "\n")
    (OUT / "net_B32.tex").write_text(net_table(d, 32) + "\n")
    (OUT / "split_B32.tex").write_text(split_table(d, 32) + "\n")
    (OUT / "tree_opt_B32.tex").write_text(tree_opt_table(d, 32) + "\n")
    (OUT / "datasets.tex").write_text(data_table() + "\n")
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
