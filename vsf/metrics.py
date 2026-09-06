"""
VSF Metrics: shared cell-code bookkeeping and multiplicity control.

This module holds exactly two things, both of which the coverage search
genuinely depends on:

*   `cell_codes` / `dense_codes_from_flat` - dense joint-cell bookkeeping.
    Given the category codes of a candidate feature subset, they number the
    OCCUPIED cells 0..C-1, which is the partition every quantity in
    `vsf.centers` is computed over. `dense_codes_from_flat`'s sort-free path
    is the single largest saving in the exhaustive search.
*   `benjamini_hochberg` - Benjamini-Hochberg FDR control, available to the
    Global Pattern Scan across the (column, value) family it sweeps. It
    operates on whatever p-values it is given, which are coverage p-values.

No information-theoretic quantity is computed anywhere in this package. The
plug-in mutual information and Shannon entropy that used to live here
(`mutual_information_bits`, `entropy_bits_from_counts`) existed to serve
`vsf.pmd`'s rate-distortion binning of CONTINUOUS features; VSF encodes
every column as categories (see `vsf.pmd`'s module docstring), so both, and
the `contingency_table` / `contingency_from_codes` helpers that only ever
fed them, were deleted along with their last caller. The earlier
bias-corrected association layer that once ranked branches by adjusted
mutual information was removed before them. See git history if an
information-theoretic quantity is ever wanted again -- it would be a new
feature, with its own estimator-bias argument to make, not a restoration.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np

__all__ = [
    "benjamini_hochberg",
    "cell_codes",
    "dense_codes_from_flat",
]


# --------------------------------------------------------------------------
# Label handling
# --------------------------------------------------------------------------
def _as_cell_codes(A: np.ndarray | Sequence[object]) -> np.ndarray:
    """
    Maps an (N,) or (N, d) array of discrete labels onto dense integer codes
    0..C-1 identifying the distinct joint cells, preserving row identity.

    Every caller requires a 1-D code vector, so the mixed-radix path is
    used when it cannot overflow and an exact `np.unique(..., axis=0)`
    otherwise.
    """
    arr = np.asarray(A)
    if arr.ndim == 1:
        _, codes = np.unique(arr, return_inverse=True)
        return codes.astype(np.int64, copy=False).ravel()
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got ndim={arr.ndim}")
    if arr.shape[1] == 1:
        _, codes = np.unique(arr[:, 0], return_inverse=True)
        return codes.astype(np.int64, copy=False).ravel()

    if arr.dtype.kind in ("U", "S", "O", "b"):
        cols = [np.unique(arr[:, j], return_inverse=True)[1] for j in range(arr.shape[1])]
        arr = np.column_stack(cols)

    arr = arr.astype(np.int64, copy=False)
    radices = (arr.max(axis=0) - arr.min(axis=0) + 1).astype(object)
    total = 1
    for r in radices:
        total *= int(r)
    if total < (1 << 62):
        shifted = arr - arr.min(axis=0)
        flat = shifted[:, 0].copy()
        for j in range(1, arr.shape[1]):
            flat = flat * int(radices[j]) + shifted[:, j]
        return dense_codes_from_flat(flat, total)
    _, codes = np.unique(arr, axis=0, return_inverse=True)  # pragma: no cover
    return codes.astype(np.int64, copy=False).ravel()


#: Largest nominal cell count for which `dense_codes_from_flat` relabels
#: through an occupancy `bincount` (O(N + K) time, K int64 of scratch) rather
#: than a sort. Above it a sort is both faster and bounded in memory.
DENSE_CODES_BINCOUNT_LIMIT: Final[int] = 1 << 24


def dense_codes_from_flat(flat: np.ndarray, n_nominal: int) -> np.ndarray:
    """
    Dense codes 0..C-1 for a vector of non-negative integer "nominal" codes
    in [0, n_nominal), numbering the occupied nominal values in increasing
    order - i.e. exactly `np.unique(flat, return_inverse=True)[1]`, produced
    without a sort whenever `n_nominal` is small enough for an occupancy
    table: `occ = bincount(flat) > 0; remap = cumsum(occ) - 1; remap[flat]`.

    The sort-free path is O(N + n_nominal) instead of O(N log N) and is the
    single largest saving in the exhaustive branch search, which calls this
    once per candidate subset. Both paths return identical arrays: `unique`
    numbers the distinct values in ascending order, which is what the
    cumulative occupancy count does as well.
    """
    flat = np.asarray(flat, dtype=np.int64).ravel()
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64)
    if 0 < n_nominal <= DENSE_CODES_BINCOUNT_LIMIT:
        occupied = np.bincount(flat, minlength=n_nominal) > 0
        remap = np.cumsum(occupied, dtype=np.int64) - 1
        return remap[flat]
    _, codes = np.unique(flat, return_inverse=True)
    return codes.astype(np.int64, copy=False).ravel()


def cell_codes(X: np.ndarray | Sequence[object]) -> tuple[np.ndarray, int]:
    """
    Dense integer cell codes for `X` plus the number of occupied cells.
    """
    codes = _as_cell_codes(X)
    n_cells = int(codes.max()) + 1 if codes.size > 0 else 0
    return codes, n_cells


# --------------------------------------------------------------------------
# Multiplicity control across a family of independent tests
# --------------------------------------------------------------------------
def benjamini_hochberg(p_values: Sequence[float] | np.ndarray, q: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg step-up procedure at false-discovery rate `q`.

    Returns a boolean mask of the hypotheses to reject. Used by the Global
    Pattern Scan, where the family is the set of (column, value) targets
    swept in one run: without it, sweeping k targets at a nominal 0.05 yields
    ~0.05k spurious "patterns" by construction.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = p.shape[0]
    reject = np.zeros(m, dtype=bool)
    if m == 0:
        return reject
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")
    order = np.argsort(p, kind="stable")
    thresholds = q * np.arange(1, m + 1, dtype=np.float64) / m
    passing = np.nonzero(p[order] <= thresholds)[0]
    if passing.size > 0:
        reject[order[: int(passing[-1]) + 1]] = True
    return reject
