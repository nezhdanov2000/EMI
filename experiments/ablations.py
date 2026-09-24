"""
Ablations (PLAN.md phase 3.4): the grid-capacity rule and the size floor.

For each (dataset, tau) configuration of compare_baselines.CONFIGS with
selection="purity", every capacity divisor c in {5, 10, 20, none} (a grid may
occupy at most floor(N_train / c) cells; "none" disables merging) and every
size floor m in {1, 5, 20}: held-out coverage and purity at B = 8 and 32 of
`vsf` (full grid), `vsf_partial` and `rules_disjoint`, 5 x R stratified
repeated folds (the same folds as compare_baselines), training-only encoding.

The capacity rule is applied by overriding `_CandidateFactory.capacity` on
the training-fold factory before any partition is built, so every method
that reads a partition through the factory sees the same rule. Free rules do
not use the factory (their cells are the uncoarsened partitions), so their
rows are the reference for what merging costs.

Output: results/ablations.csv, one row per (dataset, tau, divisor, m, budget,
method) with mean held-out coverage, purity, the share of splits with
purity >= tau, and the mean number of merged schemas (schemas whose training
partition was coarsened) among the winning schemas.

    python experiments/ablations.py --only titanic        # one dataset per call
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import (  # noqa: E402
    held_out_counts, select_rules, select_vsf, select_vsf_partial, union_members,
)
from compare_baselines import CONFIGS, N_SPLITS, RESULTS  # noqa: E402
from datasets import load_dataset  # noqa: E402
from vsf.avr import _CandidateFactory, _prepare_search  # noqa: E402
from vsf.centers import CenterSpec, stratified_repeated_kfold  # noqa: E402

DIVISORS: Sequence[Optional[int]] = (5, 10, 20, None)
FLOORS: Sequence[int] = (1, 5, 20)
BUDGETS_ABL: Sequence[int] = (8, 32)
METHODS_ABL: Sequence[str] = ("vsf", "vsf_partial", "rules_disjoint")


def run(dataset: str, positive: str, tau: float, n_repeats: int, seed: int) -> List[Dict[str, object]]:
    frame, entry = load_dataset(dataset)
    feats = [c for c in frame.columns if c != entry.target]
    X, Z = frame[feats].values, frame[entry.target].values
    prepared = _prepare_search(X, Z, None, positive, CenterSpec(tau=tau, min_samples=1), "presence")
    if prepared is None:
        raise ValueError("no feature columns")
    factory, z, names = prepared
    X_all = factory._raw.astype(np.int64, copy=False)
    splits = stratified_repeated_kfold(z, N_SPLITS, n_repeats, seed)
    rows: List[Dict[str, object]] = []
    for divisor in DIVISORS:
        for m in FLOORS:
            acc: Dict[tuple, List[tuple]] = {}
            merged_count: List[int] = []
            t0 = time.perf_counter()
            for train, test in splits:
                z_train = z[train]
                fit = _CandidateFactory(X_all[train], factory.bin_counts, int(train.size), ordered=factory.ordered)
                fit.capacity = (max(1, int(train.size) // divisor) if divisor is not None
                                else int(np.iinfo(np.int64).max))
                sel = {
                    "vsf": select_vsf(fit, X_all, train, z_train, tau, m, BUDGETS_ABL, names),
                    "vsf_partial": select_vsf_partial(fit, X_all, train, z_train, tau, m, BUDGETS_ABL, names),
                    "rules_disjoint": select_rules(X_all, train, z_train, tau, m, BUDGETS_ABL, names, disjoint=True),
                }
                for mth, res in sel.items():
                    for b in BUDGETS_ABL:
                        groups = res.groups[b]
                        inside = union_members(groups, z.shape[0])
                        k_in, n_in, n_pos = held_out_counts(inside, z, test)
                        acc.setdefault((mth, b), []).append((k_in, n_in, n_pos, sum(g.cost for g in groups)))
            seconds = time.perf_counter() - t0
            for (mth, b), vals in acc.items():
                cov = np.array([k / max(1, p) for k, _, p, _ in vals])
                pur = np.array([k / n if n else np.nan for k, n, _, _ in vals])
                rows.append(dict(
                    dataset=dataset, positive=positive, tau=tau, divisor=divisor if divisor is not None else "none",
                    m=m, budget=b, method=mth, coverage=round(float(cov.mean()), 4),
                    purity=round(float(np.nanmean(pur)), 4) if np.isfinite(pur).any() else "",
                    purity_reaches_tau=round(float(np.nanmean(pur >= tau)), 3) if np.isfinite(pur).any() else "",
                    conditions=round(float(np.mean([c for _, _, _, c in vals])), 2),
                    splits=len(vals), seconds=round(seconds, 1),
                ))
            print(f"{dataset} tau={tau} c={divisor} m={m}: {seconds:.0f}s", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    configs = [c for c in CONFIGS if c.dataset == args.only and c.selection == "purity"]
    if not configs:
        raise SystemExit(f"no purity configuration for {args.only!r}")
    seen = set()
    rows: List[Dict[str, object]] = []
    for cfg in configs:
        if (cfg.positive, cfg.tau) in seen:
            continue
        seen.add((cfg.positive, cfg.tau))
        rows += run(cfg.dataset, cfg.positive, cfg.tau, min(cfg.n_repeats, 2), args.seed)
    out = RESULTS / "ablations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, object]] = []
    if out.exists():
        with out.open(newline="", encoding="utf-8") as fh:
            existing = [r for r in csv.DictReader(fh) if r["dataset"] != args.only]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(existing)
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} new rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
