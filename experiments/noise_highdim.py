"""
What the product reports on PURE NOISE shaped like a small high-dimensional
dataset (colon: 62 rows, 40 positives, ternary features).

The features are independent of the target by construction, so every centre
the search reports is false. For each number of features M the script runs
`vsf.discover_branches` (d <= 2) twice on the same data:

  observed   CenterSpec(rule="purity", tau, min_samples)          - share >= tau
  certified  CenterSpec(rule="certified", multiplicity="family")  - the default colouring

and prints, per branch, the reported coverage, its cross-validated value,
the number of centres and the largest centre. It also counts, over ALL pairs, the cells that pass the
observed rule (what a user scanning the grids could find).

Usage:
    python experiments/noise_highdim.py                  # M = 50, 200, 500
    python experiments/noise_highdim.py --features 1000 --seed 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsf.avr import BranchResult, discover_branches  # noqa: E402
from vsf.centers import CenterSpec  # noqa: E402


def noise(n: int, n_pos: int, n_features: int, levels: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, levels, size=(n, n_features)).astype(str)
    z = np.zeros(n, dtype=np.int64)
    z[rng.choice(n, n_pos, replace=False)] = 1
    return X, np.where(z == 1, "yes", "no")


def passing_pairs(X: np.ndarray, Z: np.ndarray, tau: float, min_samples: int) -> int:
    """Occupied cells of all 2-D raw grids with n >= min_samples and k / n >= tau."""
    z = (Z == "yes").astype(np.int64)
    levels = sorted(set(X.ravel().tolist()))
    onehot = np.stack([(X == v) for v in levels]).astype(np.int64)  # L x n x M
    iu = np.triu_indices(X.shape[1], 1)
    total = 0
    for a in range(len(levels)):
        for b in range(len(levels)):
            n_cell = (onehot[a].T @ onehot[b])[iu]
            k_cell = ((onehot[a] * z[:, None]).T @ onehot[b])[iu]
            total += int(np.count_nonzero((n_cell >= min_samples) & (k_cell >= tau * n_cell)))
    return total


def describe(branches: Dict[int, BranchResult]) -> str:
    parts = []
    for d, br in sorted(branches.items()):
        rep = br.centers
        largest = max((c.n for c in rep.centers), default=0)
        cv = "   -" if rep.coverage_cv is None else f"{100 * rep.coverage_cv.mean:4.1f}%"
        parts.append(f"{d}D coverage {100 * rep.coverage:5.1f}% (CV {cv}) centres {rep.n_centers:2d}"
                     f" largest {largest:2d} [{' + '.join(br.selected_feature_names)}]")
    return " | ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=int, nargs="+", default=[50, 200, 500])
    ap.add_argument("--rows", type=int, default=62)
    ap.add_argument("--positives", type=int, default=40)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    print(f"pure noise: {args.rows} rows, {args.positives} positives (base rate "
          f"{args.positives / args.rows:.1%}), {args.levels}-level features; tau = {args.tau}, "
          f"min_samples = {args.min_samples}")
    for m in args.features:
        X, Z = noise(args.rows, args.positives, m, args.levels, args.seed)
        names = [f"f{j}" for j in range(m)]
        print(f"\nM = {m}: {m * (m - 1) // 2} pairs; cells passing the observed rule over all pairs:"
              f" {passing_pairs(X, Z, args.tau, args.min_samples)}")
        specs = {
            "observed ": CenterSpec(tau=args.tau, min_samples=args.min_samples),
            "certified": CenterSpec(tau=args.tau, min_samples=args.min_samples, rule="certified",
                                    multiplicity="family"),
        }
        for label, spec in specs.items():
            t0 = time.perf_counter()
            branches = discover_branches(X, Z, names, max_d=2, positive_class="yes", center_spec=spec,
                                         n_permutations_centers=0, cv_repeats=1)
            print(f"  {label}: {describe(branches)}  ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
