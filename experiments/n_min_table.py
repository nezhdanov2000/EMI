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


def family_counts(X: np.ndarray, max_d: int) -> Tuple[int, int, np.ndarray]:
    """(cells listed with repetition, distinct row sets, sorted sizes of the distinct row sets)."""
    X_discrete, bin_counts = discretize_dataset(X)
    ordered = ordered_columns(X)
    factory = _CandidateFactory(X_discrete, bin_counts, X.shape[0], ordered=ordered)
    listed = 0
    for _, _, n_cells in factory.iter_candidates(max_d):
        listed += int(n_cells)
        if factory.last_coarsened:
            listed += int(factory.last_raw_cells)
    fresh = _CandidateFactory(X_discrete, bin_counts, X.shape[0], ordered=ordered)
    sizes = fresh.family_cell_sizes(max_d)
    return listed, int(sizes.shape[0]), sizes


def _pure_n_min(tau: float, level: float) -> int:
    """Exact smallest n for a pure cell (k = n) under heterogeneous certificate. O(1)."""
    import math
    target = math.log(level * (1.0 - 1e-9))
    n = max(1, int(math.ceil(target / math.log(tau))))
    num, den = float(tau).as_integer_ratio()
    while True:
        k_guard = -((-n * num) // den) + 1
        if k_guard <= n and (n * math.log(tau)) <= target:
            return n
        n += 1


def tarone_family(sizes: np.ndarray, tau: float, alpha: float = ALPHA) -> Tuple[int, int]:
    """
    Tarone (1990) for the family certificate at a FIXED tau: the number K of
    hypotheses that can reach the threshold at level alpha / K, and the
    resulting per-cell level's n_min for a pure cell. A row set of n rows
    can be certified at level L iff n >= n_min_pure(L), so m(K) = #{sizes >=
    n_min_pure(alpha / K)} is the number of testable hypotheses at alpha / K;
    m is non-increasing in K, and K* is the smallest K with m(K) <= K (found
    by bisection on [1, T]). Valid only when tau is fixed before the data
    are read (PMD 4.14): the family then depends on tau, which the
    interactive product does not allow.
    """
    if sizes.size == 0:
        return 0, -1
    T = int(sizes.shape[0])

    def testable(K: int) -> int:
        pure = _pure_n_min(tau, alpha / K)
        return int(np.count_nonzero(sizes >= pure)) if pure > 0 else 0

    lo, hi = 1, T                      # m(T) <= T always
    while lo < hi:
        mid = (lo + hi) // 2
        if testable(mid) <= mid:
            hi = mid
        else:
            lo = mid + 1
    pure = _pure_n_min(tau, alpha / lo)
    return lo, pure


def n_min(tau: float, rho: float, level: float) -> Tuple[int, int]:
    """
    (n_min_pure, n_min_mid) for one floor, from one table of thresholds for
    n = 1 .. N_MAX. -1 means "not reached within N_MAX rows".
    """
    from certificate_price import n_min as cp_n_min
    pure = cp_n_min(tau, level, 1.0)
    mid = cp_n_min(tau, level, rho)
    return (pure if pure is not None else -1), (mid if mid is not None else -1)


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
    listed, distinct, sizes = family_counts(X, args.max_d)
    seconds = time.perf_counter() - t0
    level = ALPHA / distinct

    rows: List[Dict[str, object]] = []
    for tau in TAUS:
        rho = min(1.0, (1.0 + tau) / 2.0)
        pure, mid = n_min(tau, rho, level)
        pure_listed, _ = n_min(tau, 1.0, ALPHA / listed)
        k_tarone, pure_tarone = tarone_family(sizes, tau)
        _, mid_tarone = n_min(tau, rho, ALPHA / max(1, k_tarone))
        rows.append({
            "source": source, "N": int(X.shape[0]), "M": int(X.shape[1]), "max_d": args.max_d,
            "T_listed": listed, "T_distinct": distinct, "alpha": ALPHA,
            "per_cell_level": level, "tau": tau,
            "n_min_pure": pure,
            "purity_mid": rho,
            "n_min_mid": mid,
            "n_min_pure_listed_T": pure_listed,
            "K_tarone": k_tarone,
            "n_min_pure_tarone": pure_tarone,
            "n_min_mid_tarone": mid_tarone,
            "seconds_family": round(seconds, 1),
        })
    print(f"{source}: N={X.shape[0]} M={X.shape[1]} T listed {listed:,} distinct {distinct:,} ({seconds:.1f} s)")
    for r in rows:
        print(f"  tau={r['tau']:.2f}: pure cell >= {r['n_min_pure']} rows "
              f"(without dedup {r['n_min_pure_listed_T']}, Tarone K={r['K_tarone']}: {r['n_min_pure_tarone']}); "
              f"purity {r['purity_mid']:.3f} from {r['n_min_mid']} rows (Tarone {r['n_min_mid_tarone']})")
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
