"""
The price of the family certificate: how large a cell must be before it can
be certified at all, as a function of the dataset.

For a family of T hypotheses (`vsf.avr.family_cell_count`: distinct row sets
of every partition a search over d <= 4 can show) the per-cell level is
alpha / T, and a cell of n rows with k of them positive is certified iff
k >= `min_successes_to_certify_heterogeneous`(n, tau, alpha / T).
Reported per floor tau:

  n_min_pure   smallest n at which a cell whose rows are ALL positive is
               certified (a lower bound on any certifiable cell);
  n_min_mid    smallest n from which on (up to N_MAX = 20 000 rows) every
               cell with observed purity at least (1 + tau) / 2 is certified;
               -1 when that does not happen within N_MAX rows.

T depends only on the feature columns, never on the target, tau or
min_samples. Two sources:

  --dataset NAME        a bundled dataset (every column but its target);
  --synthetic M N       M independent columns, 4 equiprobable levels each,
                        N rows, seed 0.

Each call appends one row per tau to results/n_min_table.csv.

Usage:
    python experiments/n_min_table.py --dataset titanic
    python experiments/n_min_table.py --synthetic 20 5000
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset  # noqa: E402
from vsf.avr import MAX_BRANCH_D, _CandidateFactory  # noqa: E402
from vsf.centers import min_successes_to_certify_heterogeneous  # noqa: E402
from vsf.pmd import discretize_dataset, ordered_columns  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "n_min_table.csv"
TAUS: Tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 0.95)
ALPHA = 0.05
N_MAX = 20_000


def family_counts(X: np.ndarray, max_d: int) -> Tuple[int, int]:
    """(cells listed with repetition, distinct row sets) of the family."""
    X_discrete, bin_counts = discretize_dataset(X)
    ordered = ordered_columns(X)
    factory = _CandidateFactory(X_discrete, bin_counts, X.shape[0], ordered=ordered)
    listed = 0
    for _, _, n_cells in factory.iter_candidates(max_d):
        listed += int(n_cells)
        if factory.last_coarsened:
            listed += int(factory.last_raw_cells)
    fresh = _CandidateFactory(X_discrete, bin_counts, X.shape[0], ordered=ordered)
    return listed, fresh.family_cell_count(max_d)


def n_min(tau: float, rho: float, level: float) -> Tuple[int, int]:
    """
    (n_min_pure, n_min_mid) for one floor, from one table of thresholds for
    n = 1 .. N_MAX. -1 means "not reached within N_MAX rows".
    """
    n = np.arange(1, N_MAX + 1)
    k_min = min_successes_to_certify_heterogeneous(n, tau, level)
    pure = np.flatnonzero(k_min <= n)
    # k = ceil(rho * n) in Python integers (num * n overflows int64)
    num, den = float(rho).as_integer_ratio()
    k_mid = np.fromiter((-((-int(v) * num) // den) for v in n.tolist()), dtype=np.int64, count=n.size)
    ok = k_min <= k_mid
    if not ok[-1]:
        mid = -1
    else:
        bad = np.flatnonzero(~ok)
        mid = int(bad[-1] + 2) if bad.size else 1
    return (int(n[pure[0]]) if pure.size else -1), mid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset")
    src.add_argument("--synthetic", nargs=2, type=int, metavar=("M", "N"))
    ap.add_argument("--max-d", type=int, default=MAX_BRANCH_D)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    if args.dataset:
        frame, entry = load_dataset(args.dataset)
        X = frame[[c for c in frame.columns if c != entry.target]].values
        source = args.dataset
    else:
        m, n = args.synthetic
        X = np.random.default_rng(0).integers(0, 4, size=(n, m)).astype(str)
        source = f"synthetic L=4 M={m} N={n}"
    t0 = time.perf_counter()
    listed, distinct = family_counts(X, args.max_d)
    seconds = time.perf_counter() - t0
    level = ALPHA / distinct

    rows: List[Dict[str, object]] = []
    for tau in TAUS:
        rho = min(1.0, (1.0 + tau) / 2.0)
        pure, mid = n_min(tau, rho, level)
        pure_listed, _ = n_min(tau, 1.0, ALPHA / listed)
        rows.append({
            "source": source, "N": int(X.shape[0]), "M": int(X.shape[1]), "max_d": args.max_d,
            "T_listed": listed, "T_distinct": distinct, "alpha": ALPHA,
            "per_cell_level": level, "tau": tau,
            "n_min_pure": pure,
            "purity_mid": rho,
            "n_min_mid": mid,
            "n_min_pure_listed_T": pure_listed,
            "seconds_family": round(seconds, 1),
        })
    print(f"{source}: N={X.shape[0]} M={X.shape[1]} T listed {listed:,} distinct {distinct:,} ({seconds:.1f} s)")
    for r in rows:
        print(f"  tau={r['tau']:.2f}: pure cell >= {r['n_min_pure']} rows "
              f"(without dedup {r['n_min_pure_listed_T']}); purity {r['purity_mid']:.3f} from {r['n_min_mid']} rows")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    with args.out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        if new_file:
            w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
