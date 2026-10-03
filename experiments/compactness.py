"""
Compactness of the selected description at one budget.

Reads `experiments/results/comparison_summary.csv` (written by
`summarize_comparison.py`) and reports, per selection rule and method, over
the informative configurations at budget B (some method above 1 % held-out
coverage at that budget) on which EVERY method of METHOD_ORDER ran, so the
rows are paired:

* mean held-out coverage;
* mean conditions actually spent (averaged over splits by the comparison);
* mean number of groups and mean number of DISTINCT COLUMNS named by the
  description on fold 0 (`fold0_groups`; the column set is not recorded
  per split, so this is one fold, stated as such). Columns are parsed from
  the `X_<j>` tokens of the group labels; the tree and SSD++ record no
  column labels and get a blank.

Writes `paper/tables/compactness_B<B>.tex` (booktabs) and prints the same
table. Pure standard library on purpose: it runs where pandas is absent.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

METHOD_ORDER: Tuple[str, ...] = (
    "vsf", "vsf_partial", "vsf_greedy", "rules_disjoint", "rules", "tree", "ssdpp",
)
METHOD_LABEL: Dict[str, str] = {
    "vsf": "full grid",
    "vsf_partial": "partial centres",
    "vsf_greedy": "greedy chain",
    "rules_disjoint": r"rules$_{\mathrm{disj}}$",
    "rules": "rules",
    "tree": "tree",
    "ssdpp": "SSD++",
}
_COL = re.compile(r"X_(\d+)")
_LABELLED = ("vsf", "vsf_partial", "vsf_greedy", "rules", "rules_pure", "rules_disjoint")


def _columns_and_groups(method: str, fold0_groups: str) -> Tuple[Optional[int], Optional[int]]:
    """(#distinct columns, #groups) on fold 0, or (None, None) when the label carries no columns."""
    if not fold0_groups.strip():
        return None, 0
    groups = [g for g in fold0_groups.split(" | ") if g.strip()]
    if method not in _LABELLED:
        return None, len(groups)
    cols = {int(m) for m in _COL.findall(fold0_groups)}
    return len(cols), len(groups)


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _fmt(value: Optional[float], digits: int) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def build(rows: List[Dict[str, str]], budget: int) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    at_budget = [r for r in rows if int(r["budget"]) == budget]
    best: Dict[Tuple[str, str], float] = defaultdict(float)
    for r in at_budget:
        key = (r["selection"], r["config"])
        # informative by the methods of the comparison only: wracc_raw (no
        # purity floor) would make every certified configuration informative
        if r["method"] in METHOD_ORDER:
            best[key] = max(best[key], float(r["coverage"]))
        else:
            best.setdefault(key, 0.0)
    present: Dict[Tuple[str, str], set] = defaultdict(set)
    for r in at_budget:
        present[(r["selection"], r["config"])].add(r["method"])
    # Informative AND paired: every method of METHOD_ORDER ran on the
    # configuration, so all rows of the table average over the same cells.
    informative = {k for k, v in best.items()
                   if v > 0.01 and set(METHOD_ORDER) <= present[k]}
    acc: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in at_budget:
        if (r["selection"], r["config"]) not in informative:
            continue
        m = r["method"]
        if m not in METHOD_ORDER:
            continue
        sel = r["selection"]
        n_cols, n_groups = _columns_and_groups(m, r["fold0_groups"])
        acc[sel][m]["coverage"].append(float(r["coverage"]))
        acc[sel][m]["conditions"].append(float(r["conditions_spent"]))
        acc[sel][m]["groups"].append(float(n_groups))
        if n_cols is not None:
            acc[sel][m]["columns"].append(float(n_cols))
        acc[sel][m]["n"].append(1.0)
    out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for sel, per_method in acc.items():
        out[sel] = {}
        for m, stats in per_method.items():
            out[sel][m] = {
                "coverage": _mean(stats["coverage"]),
                "conditions": _mean(stats["conditions"]),
                "groups": _mean(stats["groups"]),
                "columns": _mean(stats["columns"]),
                "n": float(len(stats["n"])),
            }
    return out


def to_latex(table: Dict[str, Dict[str, Dict[str, Optional[float]]]]) -> str:
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"selection & method & cells & coverage & conditions & groups & columns\\", r"\midrule"]
    for sel in ("purity", "certified"):
        if sel not in table:
            continue
        for m in METHOD_ORDER:
            if m not in table[sel]:
                continue
            s = table[sel][m]
            lines.append(
                f"{sel} & {METHOD_LABEL[m]} & {int(s['n'] or 0)} & {_fmt(s['coverage'], 2)} & "
                f"{_fmt(s['conditions'], 1)} & {_fmt(s['groups'], 1)} & {_fmt(s['columns'], 1)}\\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=Path("experiments/results/comparison_summary.csv"))
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None, help="default paper/tables/compactness_B<budget>.tex")
    args = ap.parse_args()
    with args.summary.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    table = build(rows, args.budget)
    tex = to_latex(table)
    out = args.out or Path("paper/tables") / f"compactness_B{args.budget}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
