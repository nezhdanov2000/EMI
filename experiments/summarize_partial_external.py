"""Summary of partial_centres results.

For every (selection, dataset, tau, budget) configuration and every method:
held-out coverage, held-out purity, share of splits with purity >= tau, and
paired Nadeau-Bengio t versus the full grid and versus free disjoint rules.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

KEYS = ["selection", "dataset", "tau", "budget"]


def nb_t(diff: np.ndarray, rho: float) -> float:
    J = len(diff)
    if J < 2:
        return float("nan")
    var = diff.var(ddof=1) * (1.0 / J + rho)
    return float(diff.mean() / np.sqrt(var)) if var > 0 else 0.0


def mark(t: float) -> str:
    return "+" if t >= 2 else ("-" if t <= -2 else "=")


def paired(d: pd.DataFrame, a: str, b: str, rho: float) -> pd.DataFrame:
    rows = []
    for key, g in d.groupby(KEYS):
        A = g[g.method == a].set_index(["repeat", "fold"]).test_coverage
        B = g[g.method == b].set_index(["repeat", "fold"]).test_coverage
        common = A.index.intersection(B.index)
        diff = (A.loc[common] - B.loc[common]).to_numpy()
        t = nb_t(diff, rho)
        rows.append(dict(zip(KEYS, key), method=a, diff=float(diff.mean()), t=t, mark=mark(t)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="experiments/results/partial_external/part_*_*.csv")
    ap.add_argument("--out", default="experiments/results/partial_external/summary.csv")
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(a.glob))], ignore_index=True)
    d["test_purity"] = d["test_purity"].fillna(0.0)
    rho = 1.0 / (a.splits - 1)
    cov_max = d.groupby(KEYS).test_coverage.transform("max")
    d = d[cov_max >= 0.01].copy()

    agg = d.groupby(KEYS + ["method"]).agg(
        N=("N", "first"), M=("M", "first"), base=("base_rate", "first"), T=("family_T", "first"),
        coverage=("test_coverage", "mean"), purity=("test_purity", "mean"),
        n_centres=("n_centres", "mean"), conditions=("conditions_used", "mean")).reset_index()
    ok = d.assign(ok=(d.test_purity >= d.tau)).groupby(KEYS + ["method"]).ok.mean().rename("purity_ok_share")
    agg = agg.merge(ok.reset_index(), on=KEYS + ["method"])
    parts = []
    for meth in sorted(d.method.unique()):
        pg = paired(d, meth, "grid", rho).rename(columns={"diff": "diff_vs_grid", "t": "t_vs_grid", "mark": "mark_vs_grid"})
        pr = paired(d, meth, "rules_disjoint", rho).rename(columns={"diff": "diff_vs_rules", "t": "t_vs_rules", "mark": "mark_vs_rules"})
        parts.append(pg.merge(pr, on=KEYS + ["method"]))
    agg = agg.merge(pd.concat(parts), on=KEYS + ["method"])
    agg.to_csv(a.out, index=False)

    for sel, s in agg.groupby("selection"):
        print(f"\n================ selection = {sel} ================")
        print("configs:", s.groupby(["dataset", "tau", "budget"]).ngroups)
        print("\nvs full grid  (+ better / = / - worse, |t|>=2):")
        print(s[s.method != "grid"].groupby("method").mark_vs_grid.value_counts().unstack(fill_value=0).to_string())
        print("\nvs rules_disjoint:")
        print(s[s.method != "rules_disjoint"].groupby("method").mark_vs_rules.value_counts().unstack(fill_value=0).to_string())
        print("\nshare of splits with held-out purity >= tau:")
        print(s.groupby("method").purity_ok_share.mean().round(3).to_string())
        print("\nmean held-out coverage by budget:")
        print(s.pivot_table(index="budget", columns="method", values="coverage").round(3).to_string())
        p = s.pivot_table(index=["dataset", "tau", "budget"], columns="method", values="coverage")
        gap = p.rules_disjoint - p.grid
        sel_ = gap >= 0.01
        if sel_.any():
            closed = (p.partial_disjoint - p.grid)[sel_] / gap[sel_]
            closed_u = (p.partial_union - p.grid)[sel_] / gap[sel_]
            print(f"\nconfigs where rules beat grid by >= 1 pp: {int(sel_.sum())}; "
                  f"gap closed by partial_disjoint: median {closed.median():.2f}, mean {closed.mean():.2f}; "
                  f"by partial_union: median {closed_u.median():.2f}, mean {closed_u.mean():.2f}")
        print("\ncoverage by dataset x tau (budget 32):")
        print(s[s.budget == 32].pivot_table(index=["dataset", "tau"], columns="method", values="coverage").round(3).to_string())


if __name__ == "__main__":
    main()
