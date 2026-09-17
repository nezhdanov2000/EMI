"""
One table from every results/compare_*.csv: per configuration and budget,
each method's held-out coverage, purity and net coverage, and the paired
differences to `vsf` (Nadeau-Bengio se; |t| >= 2 marked + / -), plus the
description stability. Writes results/comparison_summary.csv and prints a
compact view at the budgets given, and counts of +/=/- over all budgets.

Configurations listed in EXCLUDED, and configurations where no method
covers at least MIN_INFORMATIVE of the held-out positives at any budget
(e.g. the certified selection at N = 2000, which certifies nothing), are
kept in the csv but left out of the counts and medians: they would only add
ties.

Usage:
    python experiments/summarize_comparison.py --budgets 4 8 16
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
METHOD_ORDER = ("vsf", "vsf_greedy", "rules_pure", "rules", "tree")
MIN_INFORMATIVE = 0.01
EXCLUDED: Dict[str, str] = {}


def _mark(diff: pd.Series, se: pd.Series) -> pd.Series:
    """'+' / '-' where |diff / se| >= 2 (se = 0 with a nonzero diff counts as significant), else '='."""
    t = diff / se.where(se > 0)
    out = pd.Series("=", index=diff.index)
    out[(t >= 2) | ((se == 0) & (diff > 0))] = "+"
    out[(t <= -2) | ((se == 0) & (diff < 0))] = "-"
    return out


def _pct(value: float, width: int) -> str:
    return f"{'-':>{width}s}" if pd.isna(value) or value == "" else f"{100 * float(value):{width}.0f}"


def _cell(row: pd.Series) -> str:
    """'coverage/purity net stability' with the two marks (coverage, net) against vsf."""
    marks = "" if row["method"] == "vsf" else f"{row['sig']}{row['net_sig']}"
    return (f"{100 * row['coverage']:4.0f}/{_pct(row['purity'], 3)}"
            f" {_pct(row['net_coverage'], 4)} s{_pct(row['stability'], 3)} {marks:2s}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16])
    args = ap.parse_args()
    files = sorted(glob.glob(str(RESULTS / "compare_*.csv")))
    if not files:
        raise SystemExit("no results/compare_*.csv yet")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["config"] = (d["dataset"] + " " + d["positive"].astype(str) + " t=" + d["tau"].astype(str)
                   + " m=" + d["min_samples"].astype(str) + " d<=" + d["max_d"].astype(str)
                   + " " + d["selection"].str[:4])
    d["sig"] = _mark(d["minus_vsf"], d["minus_vsf_se"])
    d["net_sig"] = _mark(d["net_minus_vsf"], d["net_minus_vsf_se"])
    d.to_csv(RESULTS / "comparison_summary.csv", index=False)

    view = d[d["budget"].isin(args.budgets)].copy()
    view["cell"] = view.apply(_cell, axis=1)
    table = view.pivot_table(index=["config", "budget"], columns="method", values="cell", aggfunc="first")
    cols = [c for c in METHOD_ORDER if c in table.columns]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print("cell: coverage %/purity % | net coverage % | s = stability % | marks vs vsf (coverage, net), |t| >= 2")
    print(table[cols].to_string())

    best = d.groupby("config")["coverage"].transform("max")
    informative = ~d["dataset"].isin(EXCLUDED) & (best >= MIN_INFORMATIVE)
    dropped = sorted(set(d.loc[~informative, "config"]))
    print(f"\nleft out of the counts ({len(dropped)}): " + "; ".join(dropped))
    counted = d[informative & (d["method"] != "vsf")]
    for column, title in (("sig", "coverage"), ("net_sig", "net coverage")):
        wins = counted.groupby(["selection", "method"])[column].value_counts().unstack(fill_value=0)
        wins = wins.reindex(columns=["+", "=", "-"], fill_value=0)
        n_cfg = counted.groupby("selection")["config"].nunique().to_dict()
        print(f"\n{title}: (config, budget) cells, method vs vsf, all budgets"
              f" (informative configs per selection: {n_cfg})")
        print(wins.to_string())
    stab = (d[informative]
            .pivot_table(index="method", columns="selection", values="stability", aggfunc="median")
            .reindex([m for m in METHOD_ORDER if m in set(d["method"])]))
    print("\nmedian description stability over (config, budget) cells:")
    print(stab.round(3).to_string())
    secs = d.pivot_table(index="dataset", columns="method", values="selection_seconds", aggfunc="max")
    print("\nselection seconds per split (max over budgets and configs):")
    print(secs.reindex(columns=[m for m in METHOD_ORDER if m in secs.columns]).round(2).to_string())
    return 0 if np.isfinite(d["coverage"]).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
