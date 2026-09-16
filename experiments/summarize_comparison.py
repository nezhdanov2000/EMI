"""
One table from every results/compare_*.csv: per configuration and budget,
held-out coverage and purity of each method and the paired difference to
`vsf` (Nadeau-Bengio se; |t| >= 2 marked). Writes results/comparison_summary.csv
and prints a compact view at the budgets given.

Usage:
    python experiments/summarize_comparison.py --budgets 4 8 16
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"


def _cell(row: pd.Series) -> str:
    """Format one table cell as 'coverage/purity' plus the significance mark."""
    purity = "  - " if pd.isna(row["purity"]) else f"{100 * row['purity']:3.0f}"
    return f"{100 * row['coverage']:5.1f}/{purity}{row['sig']:1s}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16])
    args = ap.parse_args()
    files = sorted(glob.glob(str(RESULTS / "compare_*.csv")))
    if not files:
        raise SystemExit("no results/compare_*.csv yet")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["config"] = (d["dataset"] + " " + d["positive"].astype(str) + " t=" + d["tau"].astype(str)
                   + " m=" + d["min_samples"].astype(str) + " d<=" + d["max_d"].astype(str))
    t = d["minus_vsf"] / d["minus_vsf_se"].where(d["minus_vsf_se"] > 0)
    d["sig"] = ""
    d.loc[t >= 2, "sig"] = "+"
    d.loc[t <= -2, "sig"] = "-"
    d.loc[(d["minus_vsf_se"] == 0) & (d["minus_vsf"] > 0), "sig"] = "+"
    d.loc[(d["minus_vsf_se"] == 0) & (d["minus_vsf"] < 0), "sig"] = "-"
    d.to_csv(RESULTS / "comparison_summary.csv", index=False)
    view = d[d["budget"].isin(args.budgets)].copy()
    view["cell"] = view.apply(_cell, axis=1)
    table = view.pivot_table(index=["config", "budget"], columns="method", values="cell", aggfunc="first")
    cols = [c for c in ("vsf", "vsf_greedy", "rules_pure", "rules", "tree") if c in table.columns]
    pd.set_option("display.width", 200)
    print("held-out coverage % / purity %;  +/- : paired difference to vsf at |t| >= 2")
    print(table[cols].to_string())
    marks = view["sig"].replace("", "=")
    wins = marks[view["method"] != "vsf"].groupby(view["method"]).value_counts().unstack(fill_value=0)
    print("\ncount of (config, budget) cells, method vs vsf:")
    print(wins.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
