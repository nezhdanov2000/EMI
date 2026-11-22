"""
A permutation (association) test answers a different question than the
certificate: the paper's example, made numerical.

Resampling-based significant-pattern mining (Westfall-Young permutation,
Llinares-Lopez et al. 2015; SPuManTE; FSR) controls the family-wise error
of the hypothesis "the pattern's target rate equals the base rate", i.e. of
independence. The certificate of Section 4 bounds the one-sided LEVEL
"the cell's mean target probability is at most tau". A cell whose purity
is far above the base rate but below tau is significant under the first
and must not be certified under the second.

Planted design (as experiments/certificate_power.py): `--features` columns
with 3 uniform levels, P(Z = 1 | x) = `--purity` (default 0.85) in the cell
x0 = 0, x1 = 0, the base rate `--background` (0.10) elsewhere; tau = 0.90.
Family: every occupied cell of every schema of d <= `--max-d` columns, the
same family the certificate counts (distinct row sets). For the planted
cell this reports

  * the family certificate at tau (vsf.centers threshold at alpha / T):
    certified or not;
  * the one-sided hypergeometric p-value of the cell's count against the
    base rate (the per-pattern test the permutation methods adjust);
  * its Westfall-Young adjusted p-value: the share of `--permutations`
    target permutations in which the minimum p-value over the whole family
    is at most the observed one (min-p adjustment, single-step).

Usage:
    python experiments/permutation_null.py --sizes 1000 3000 --runs 20
Output:
    experiments/results/permutation_null.csv  (one row per N, averaged over runs)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vsf.centers import min_successes_to_certify_heterogeneous  # noqa: E402

TAU = 0.90


def generate(seed: int, n: int, m: int, purity: float, background: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, m))
    p = np.full(n, background)
    p[(X[:, 0] == 0) & (X[:, 1] == 0)] = purity
    Z = (rng.random(n) < p).astype(np.int64)
    return X, Z


def family_cells(X: np.ndarray, max_d: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    (codes, sizes, T): `codes[j]` is the cell index of every row under
    family cell j's partition (one column per distinct row set), `sizes`
    the cell sizes. Distinct row sets are kept once.
    """
    n, m = X.shape
    seen: Dict[bytes, int] = {}
    cols: List[np.ndarray] = []
    sizes: List[int] = []
    for d in range(1, max_d + 1):
        for combo in itertools.combinations(range(m), d):
            _, inv = np.unique(X[:, combo], axis=0, return_inverse=True)
            inv = inv.ravel()
            counts = np.bincount(inv)
            for c in range(counts.size):
                member = inv == c
                key = np.packbits(member).tobytes()
                if key in seen:
                    continue
                seen[key] = len(cols)
                cols.append(member)
                sizes.append(int(counts[c]))
    M = np.stack(cols, axis=1)  # rows x T, bool
    return M, np.array(sizes, dtype=np.int64), M.shape[1]


def min_p_over_family(M: np.ndarray, sizes: np.ndarray, Z: np.ndarray, n: int, P: int) -> np.ndarray:
    """One-sided hypergeometric p-value of every family cell, P(K >= k)."""
    k = Z @ M.astype(np.int64)
    return hypergeom.sf(k - 1, n, P, sizes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 3000])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--features", type=int, default=6)
    ap.add_argument("--max-d", type=int, default=3)
    ap.add_argument("--purity", type=float, default=0.85)
    ap.add_argument("--background", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "permutation_null.csv")
    args = ap.parse_args()

    rows: List[Dict[str, object]] = []
    for n in args.sizes:
        t0 = time.perf_counter()
        certified = flagged = 0
        pvals: List[float] = []
        adj: List[float] = []
        purities: List[float] = []
        cell_n: List[int] = []
        T_seen: List[int] = []
        for seed in range(args.runs):
            X, Z = generate(seed, n, args.features, args.purity, args.background)
            M, sizes, T = family_cells(X, args.max_d)
            T_seen.append(T)
            P = int(Z.sum())
            # the planted cell is a member of the family: find its column
            member = (X[:, 0] == 0) & (X[:, 1] == 0)
            j = int(np.flatnonzero((M == member[:, None]).all(axis=0))[0])
            nc, kc = int(sizes[j]), int(Z[member].sum())
            purities.append(kc / nc)
            cell_n.append(nc)
            # certificate at tau, alpha / T, heterogeneous threshold
            k_star = int(min_successes_to_certify_heterogeneous(np.array([nc]), TAU, args.alpha / T)[0])
            certified += int(kc >= k_star)
            # per-pattern association p-value and its Westfall-Young min-p adjustment
            p_obs = float(hypergeom.sf(kc - 1, n, P, nc))
            pvals.append(p_obs)
            rng = np.random.default_rng(10_000 + seed)
            hits = 0
            for _ in range(args.permutations):
                Zp = rng.permutation(Z)
                if min_p_over_family(M, sizes, Zp, n, P).min() <= p_obs:
                    hits += 1
            p_adj = (hits + 1) / (args.permutations + 1)
            adj.append(p_adj)
            flagged += int(p_adj <= args.alpha)
        row = {
            "n": n, "runs": args.runs, "permutations": args.permutations, "features": args.features,
            "max_d": args.max_d, "family_T_mean": round(float(np.mean(T_seen)), 1),
            "planted_purity": args.purity, "background": args.background, "tau": TAU, "alpha": args.alpha,
            "cell_rows_mean": round(float(np.mean(cell_n)), 1),
            "cell_purity_mean": round(float(np.mean(purities)), 4),
            "certified_share": certified / args.runs,
            "hypergeom_p_median": float(np.median(pvals)),
            "wy_adjusted_p_median": float(np.median(adj)),
            "wy_flagged_share": flagged / args.runs,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        rows.append(row)
        print(f"N={n:5d} T~{row['family_T_mean']:.0f} planted cell n~{row['cell_rows_mean']:.0f} "
              f"purity {row['cell_purity_mean']:.3f} < tau={TAU}: certified in {row['certified_share']:.2f} "
              f"of runs; Westfall-Young adjusted p median {row['wy_adjusted_p_median']:.4f}, "
              f"flagged significant in {row['wy_flagged_share']:.2f} of runs ({row['seconds']} s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
