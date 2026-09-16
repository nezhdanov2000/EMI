"""
How strict is the default lattice colouring for one cell, and what would
other error-rate policies decide?

For a cell with k positives out of n at boundary tau, over the family of
distinct row sets the search can show (T, `family_cell_count`), prints:
- the one-sided p-value P(Bin(n, tau) >= k) (1 when condition (i) of
  Proposition 3, k >= ceil(n tau) + 1, fails);
- the largest family for which the cell would be certified at 5 %
  family-wise;
- Clopper-Pearson lower bounds at the single-test, per-schema, and family
  levels;
- Tarone's reduced family (cells too small to ever be certified dropped;
  valid for a tau fixed in advance only);
- how many family cells Bonferroni, Benjamini-Hochberg (FDR, PRDS) and
  Benjamini-Yekutieli (FDR, any dependence) certify, and whether the cell
  is among them.

Default: titanic without sex and deck, cell 2nd_Class x Mrs (37 / 41),
tau = 0.7 - the case in the screenshot of 2026-09-16.

Usage:
    python experiments/colouring_strictness.py
    python experiments/colouring_strictness.py --drop sex deck --k 37 --n 41 --tau 0.7 --schema-cells 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import beta, binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset  # noqa: E402
from vsf.avr import _FINGERPRINT_SEED, _cell_fingerprints, _prepare_search  # noqa: E402
from vsf.centers import CenterSpec  # noqa: E402

ALPHA = 0.05


def family_cells(X: np.ndarray, Z: np.ndarray, names: List[str], positive: str,
                 max_d: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """(n, k) of every distinct row set in the family (same definition as `family_cell_count`)."""
    prepared = _prepare_search(X, Z, names, positive, CenterSpec(tau=0.5), "presence")
    if prepared is None:
        raise SystemExit("no feature columns")
    factory, z, _ = prepared
    z64 = np.asarray(z, dtype=np.int64)
    rng = np.random.default_rng(_FINGERPRINT_SEED)
    hi = np.iinfo(np.uint64).max
    w1 = rng.integers(0, hi, size=factory.n_samples, dtype=np.uint64, endpoint=True)
    w2 = rng.integers(0, hi, size=factory.n_samples, dtype=np.uint64, endpoint=True)
    cells: Dict[Tuple[int, int, int], int] = {}

    def add(codes: np.ndarray, n_cells: int) -> None:
        codes = np.asarray(codes, dtype=np.int64)
        fps = _cell_fingerprints(codes, n_cells, w1, w2)
        k = np.bincount(codes, weights=z64, minlength=n_cells).astype(np.int64)
        occupied = np.flatnonzero(np.bincount(codes, minlength=n_cells))
        for fp, c in zip(fps, occupied):
            cells[(int(fp["size"]), int(fp["h1"]), int(fp["h2"]))] = int(k[c])

    for _, codes, n_cells in factory.iter_candidates(max_d):
        if n_cells:
            add(codes, n_cells)
        if factory.last_coarsened and factory.last_raw_codes is not None:
            add(factory.last_raw_codes, factory.last_raw_cells)
    keys = list(cells)
    return (np.array([key[0] for key in keys], dtype=np.int64),
            np.array([cells[key] for key in keys], dtype=np.int64))


def p_values(k: np.ndarray, n: np.ndarray, tau: float) -> np.ndarray:
    """Valid for heterogeneous rows: the binomial tail where condition (i) holds, 1 elsewhere."""
    ok = k >= np.floor(n * tau).astype(np.int64) + 1
    return np.where(ok, binom.sf(k - 1, n, tau), 1.0)


def tarone_size(n: np.ndarray, tau: float, alpha: float) -> int:
    """Tarone (1990): smallest T' with #{cells whose best p-value <= alpha / T'} <= T'."""
    best = p_values(n, n, tau)
    for t in range(1, n.size + 1):
        if int(np.count_nonzero(best <= alpha / t)) <= t:
            return t
    return int(n.size)


def step_up(p: np.ndarray, q: float) -> float:
    """Benjamini-Hochberg cut-off at level q (0 when nothing is rejected)."""
    ps = np.sort(p)
    hits = np.flatnonzero(ps <= q * np.arange(1, ps.size + 1) / ps.size)
    return float(ps[hits.max()]) if hits.size else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="titanic")
    ap.add_argument("--positive", default="survived")
    ap.add_argument("--drop", nargs="*", default=["sex", "deck"])
    ap.add_argument("--k", type=int, default=37)
    ap.add_argument("--n", type=int, default=41)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--schema-cells", type=int, default=16,
                    help="occupied cells of the displayed schema (legacy per-schema Bonferroni)")
    args = ap.parse_args(argv)

    frame, entry = load_dataset(args.dataset)
    names = [c for c in frame.columns if c != entry.target and c not in set(args.drop)]
    n, k = family_cells(frame[names].values, frame[entry.target].values, names, args.positive)
    T = int(n.size)
    tau = args.tau
    p_cell = float(p_values(np.array([args.k]), np.array([args.n]), tau)[0])
    print(f"{args.dataset} features {names}; family T = {T} distinct row sets")
    print(f"cell {args.k}/{args.n} = {args.k / args.n:.1%} at tau = {tau}: p = {p_cell:.3g}"
          f" (certified family-wise only if T <= {ALPHA / p_cell:.1f})")
    for label, level in (("single pre-chosen test", ALPHA),
                         (f"per schema, C = {args.schema_cells} (invalid after search)", ALPHA / args.schema_cells),
                         (f"family, T = {T}", ALPHA / T)):
        lower = beta.ppf(level, args.k, args.n - args.k + 1)
        print(f"  lower bound, {label}: {lower:.1%}")
    t_tar = tarone_size(n, tau, ALPHA)
    print(f"  Tarone family at this tau: {t_tar} (lower bound {beta.ppf(ALPHA / t_tar, args.k, args.n - args.k + 1):.1%})")
    p = p_values(k, n, tau)
    harmonic = float(np.sum(1.0 / np.arange(1, T + 1)))
    for label, cut in (("Bonferroni (FWER)", ALPHA / T),
                       ("Tarone (FWER, fixed tau)", ALPHA / t_tar),
                       ("Benjamini-Hochberg (FDR, PRDS)", step_up(p, ALPHA)),
                       ("Benjamini-Yekutieli (FDR, any dependence)", step_up(p, ALPHA / harmonic))):
        print(f"  {label:42s} certifies {int(np.count_nonzero(p <= cut)):4d} family cells;"
              f" this cell: {'yes' if p_cell <= cut else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
