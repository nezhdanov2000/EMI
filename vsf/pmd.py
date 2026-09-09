"""
VSF PMD: categorical encoding and grid-capacity control.

VSF is a CATEGORICAL framework. Every feature column is a set of discrete
categories, and every value in it -- string, boolean, integer or float -- is
one category, encoded by its position in the column's sorted distinct
values. There is no binning and no continuous-feature model: a numeric
column is not "quantized into k intervals", it is read as the finite set of
values it actually contains (Project_Master_Document.md Section 2).

A genuinely continuous column (thousands of distinct floats) therefore
produces thousands of categories. It is not rejected and not silently
truncated -- it is encoded exactly as it stands, and the grid-capacity
control below is what keeps a candidate combination containing such a
column scorable. Feeding continuous measurements to a categorical
framework yields an axis with thousands of ticks; that is a
data-preparation decision belonging to the caller.

Grid capacity (Section 2.3)
---------------------------
A candidate combination of columns partitions the rows into cells. The
search ranks combinations on per-cell counts, and the certificate bounds
per-cell proportions; both need cells that hold more than a handful of
rows. The capacity rule is therefore stated on the cells that EXIST, not on
the ones that could:

    C_occupied(S) <= max(1, floor(N / 10)),

i.e. at least ten rows per occupied cell on average. The nominal product
prod_j k_j is not the quantity of interest: two seven-level columns can
occupy 12 of their 49 nominal cells, and merging their levels because 49
exceeds a limit that 12 respects destroys information for no statistical
reason -- which is exactly what the previous product-based rule did to
`title` on the Titanic data (7 x 3 x 4 x 3 = 252 nominal cells, 71
occupied, limit 89: coarsened, and the child-identifying `Master` level
merged away by alphabetical adjacency).

When a combination does exceed the limit, levels are merged by
`adaptively_coarsen_bins`: the column with the most levels loses one level
at a time, the occupancy is recomputed, and the process stops the moment
the limit holds. HOW a column loses a level depends on the one thing the
encoding knows about it - whether its values are ordered:

* a NOMINAL column (strings, booleans, objects) has no neighbourhood
  between its levels, so its rarest kept level joins an "other" group:
  every dominant level survives intact, and what is merged is the part of
  the column with the fewest rows;
* an ORDERED column (any numeric dtype) is merged between ADJACENT values
  into groups of near-equal row count, so that a measurement with
  thousands of distinct values coarsens into contiguous ranges rather than
  into "the 299 smallest values and everything else".

Both rules are functions of the column alone - never of the target - so
the partition inherits no selection effect from them (Section 2.3).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "adaptively_coarsen_bins",
    "check_grid_capacity",
    "coarsen_column",
    "column_is_ordered",
    "discretize_dataset",
    "discretize_feature",
    "grid_capacity",
    "merge_state",
    "occupied_cells",
    "ordered_columns",
]


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
def discretize_feature(X: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Encodes one column as integer category codes.

    Every distinct value becomes one category. Codes are assigned in the
    column's sorted distinct-value order (`np.unique`), which makes the
    encoding a deterministic function of the column's contents alone --
    independent of row order, of the target, and of every other column.

    Works for any dtype. An object column holding values that cannot be
    ordered against each other (mixed strings and numbers, or NaN beside
    strings) falls back to comparing their string forms, so that such a
    column encodes rather than raising; note that `1` and `"1"` then become
    the same category, which is the only sane reading of a column that
    contains both.

    Returns:
        (codes, k) -- integer codes in [0, k) and the number of categories.
    """
    arr = np.asarray(X)
    if arr.ndim > 1:
        arr = arr.ravel()
    if arr.size == 0:
        return np.zeros(0, dtype=int), 0
    try:
        _, codes = np.unique(arr, return_inverse=True)
    except TypeError:
        # Unorderable mix inside an object column (e.g. str beside float).
        _, codes = np.unique(arr.astype(str), return_inverse=True)
    codes = np.asarray(codes).ravel().astype(int)
    return codes, int(codes.max()) + 1


def discretize_dataset(X_matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """
    Encodes a full (N, M) feature matrix as integer category codes,
    column by column (`discretize_feature`).

    Returns:
        (X_discrete, category_counts)
    """
    X_arr = np.asarray(X_matrix, dtype=object)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n_rows, n_cols = X_arr.shape

    discrete_cols = []
    category_counts = []

    for j in range(n_cols):
        codes, k = discretize_feature(X_arr[:, j])
        discrete_cols.append(codes)
        category_counts.append(k)

    X_discrete = np.column_stack(discrete_cols).astype(int)
    return X_discrete, category_counts


def column_is_ordered(X: np.ndarray) -> bool:
    """
    True when a raw column's values carry an order the coarsening may use:
    a numeric dtype, or an object column whose every non-missing value is a
    number (not bool). Strings, booleans and mixed columns are nominal.
    """
    arr = np.asarray(X)
    if arr.ndim > 1:
        arr = arr.ravel()
    if arr.dtype.kind in ("i", "u", "f"):
        return True
    if arr.dtype.kind != "O":
        return False
    for v in arr.tolist():
        if v is None or isinstance(v, bool):
            return False
        if isinstance(v, float) and v != v:  # NaN is allowed
            continue
        if not isinstance(v, (int, float, np.integer, np.floating)):
            return False
    return arr.size > 0


def ordered_columns(X_matrix: np.ndarray) -> list[bool]:
    """`column_is_ordered` for every column of a raw (N, M) matrix."""
    X_arr = np.asarray(X_matrix, dtype=object)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    return [column_is_ordered(X_arr[:, j]) for j in range(X_arr.shape[1])]


# --------------------------------------------------------------------------
# Grid capacity
# --------------------------------------------------------------------------
def grid_capacity(n_samples: int) -> int:
    """The largest number of occupied cells a candidate may have: max(1, N // 10)."""
    return max(1, int(n_samples) // 10)


def occupied_cells(discrete_features: np.ndarray) -> int:
    """
    Number of distinct joint category tuples (occupied cells) in an (N, d)
    integer code matrix. Uses a mixed-radix code and an occupancy count;
    exact for any d whose nominal cell count fits in 62 bits, and the
    row-wise `np.unique` otherwise.
    """
    arr = np.asarray(discrete_features)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == 0:
        return 0
    arr = arr.astype(np.int64, copy=False)
    lo = arr.min(axis=0)
    radices = (arr.max(axis=0) - lo + 1).tolist()
    total = 1
    for r in radices:
        total *= int(r)
    if total >= (1 << 62):  # pragma: no cover - >2^62 nominal cells
        return int(np.unique(arr, axis=0).shape[0])
    flat = arr[:, 0] - lo[0]
    for j in range(1, arr.shape[1]):
        flat = flat * int(radices[j]) + (arr[:, j] - lo[j])
    return occupied_from_flat(flat, total)


def occupied_from_flat(flat: np.ndarray, n_nominal: int) -> int:
    """
    Distinct values in a non-negative integer code vector with values below
    `n_nominal`: an occupancy count when the nominal range is small relative
    to N (O(N + n_nominal)), a sort otherwise (O(N log N)) - a column with
    thousands of levels must not cost a bincount over billions of cells.
    """
    n = int(flat.shape[0])
    if n == 0:
        return 0
    if 0 < n_nominal <= max(1 << 16, 16 * n):
        return int(np.count_nonzero(np.bincount(flat, minlength=n_nominal)))
    return int(np.unique(flat).shape[0])


def check_grid_capacity(discrete_features: np.ndarray, n_samples: int) -> bool:
    """
    True when the combination's OCCUPIED cell count respects the capacity
    `grid_capacity(n_samples)`; False when it exceeds it and the levels must
    be merged before the combination is scored.
    """
    return occupied_cells(discrete_features) <= grid_capacity(n_samples)


def level_frequency_order(col: np.ndarray) -> np.ndarray:
    """
    The column's levels ordered from most to least frequent (ties broken by
    ascending code), as an array of codes. This is the order in which
    `coarsen_column` keeps levels; it depends on the column alone.
    """
    col = np.asarray(col, dtype=np.int64)
    if col.size == 0:
        return np.zeros(0, dtype=np.int64)
    counts = np.bincount(col - col.min())
    codes = np.arange(counts.shape[0], dtype=np.int64) + col.min()
    present = counts > 0
    order = np.lexsort((codes[present], -counts[present]))
    return codes[present][order]


def coarsen_column(
    col: np.ndarray, k_max: int, order: np.ndarray | None = None, ordered: bool = False
) -> np.ndarray:
    """
    Reduces a column to at most `k_max` levels. A column with at most
    `k_max` levels is returned unchanged; `k_max == 1` collapses it to a
    constant 0.

    Nominal (`ordered=False`, the default): keep the `k_max - 1` most
    frequent levels, re-coded 0 .. k_max - 2 in frequency order, and merge
    every other level into one "other" group with code `k_max - 1`.
    Adjacent codes are adjacent entries of the sorted distinct values -
    alphabetical for strings - which is no neighbourhood at all for a
    nominal column and, merged, destroyed the levels that carried the
    signal. Keeping the dominant levels preserves every cell that holds a
    substantial share of the rows.

    Ordered (`ordered=True`, numeric columns): merge ADJACENT levels - in
    value order, which is code order - into at most `k_max` contiguous
    groups of near-equal row count (group of a level = floor(rows before it
    * k_max / N)), re-coded 0 .. g in value order. A measurement with
    thousands of distinct values thus coarsens into ranges, as an analyst
    would read it, not into "the smallest 299 values and the rest".

    `order` may pass a precomputed `level_frequency_order(col)` (nominal
    mode only).
    """
    col = np.asarray(col, dtype=np.int64)
    if col.size == 0:
        return col
    lo = int(col.min())
    hi = int(col.max())
    if ordered:
        counts = np.bincount(col - lo)
        present = np.nonzero(counts)[0]
        k = int(present.shape[0])
        if k <= k_max:
            return col
        if k_max <= 1:
            return np.zeros_like(col)
        n = int(col.shape[0])
        before = np.cumsum(counts[present]) - counts[present]
        group = (before * int(k_max)) // n  # monotone in value order, < k_max
        # dense re-code of the groups actually formed
        _, group = np.unique(group, return_inverse=True)
        lut = np.zeros(hi - lo + 1, dtype=np.int64)
        lut[present] = group
        return lut[col - lo]
    if order is None:
        order = level_frequency_order(col)
    k = int(order.shape[0])
    if k <= k_max:
        return col
    if k_max <= 1:
        return np.zeros_like(col)
    lut = np.full(hi - lo + 1, k_max - 1, dtype=np.int64)  # default: "other"
    for new_code, old_code in enumerate(order[: k_max - 1].tolist()):
        lut[old_code - lo] = new_code
    return lut[col - lo]


def adaptively_coarsen_bins(
    discrete_features: np.ndarray,
    n_samples: int,
    target_max_cells: int | None = None,
    ordered: Sequence[bool] | None = None,
) -> np.ndarray:
    """
    Merges category levels until the joint partition occupies at most
    `target_max_cells` cells (default `grid_capacity(n_samples)`). Returns
    the input unchanged when the limit already holds.

    Greedy, deterministic and target-blind: at each step the column with
    the most remaining levels (ties: the lowest column index) loses one
    level via `coarsen_column` - by frequency for a nominal column, by
    adjacency for an ordered one (`ordered[j]`, default all nominal) - and
    occupancy is recomputed; the process stops as soon as the limit holds,
    or when every column is constant. Reducing the widest column first
    removes the most cells per merged level and leaves narrow, usually more
    meaningful columns untouched for longest.
    """
    arr = np.asarray(discrete_features)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = arr.astype(np.int64, copy=False)
    if target_max_cells is None:
        target_max_cells = grid_capacity(n_samples)
    coarsened, _ = coarsen_to_capacity(arr, int(target_max_cells), ordered=ordered)
    return coarsened


def merge_state(levels: Sequence[int], n_steps: int) -> List[int]:
    """
    The per-column level counts after `n_steps` merge steps of the greedy
    rule: each step takes one level from the column with the most remaining
    levels (ties: the lowest column index), never below one level. A pure
    function of the starting counts, so the merge sequence can be searched
    without touching the data.
    """
    k = [int(v) for v in levels]
    for _ in range(int(n_steps)):
        widest = max(range(len(k)), key=lambda j: (k[j], -j))
        if k[widest] <= 1:
            break
        k[widest] -= 1
    return k


def coarsen_to_capacity(
    arr: np.ndarray,
    target_max_cells: int,
    orders: Sequence[np.ndarray] | None = None,
    ordered: Sequence[bool] | None = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    `adaptively_coarsen_bins`'s engine: returns the coarsened matrix and the
    per-column level counts it ended with. `orders` may pass the columns'
    `level_frequency_order`s (callers scoring many combinations cache them).

    The result is the FIRST state of the greedy merge sequence
    (`merge_state`) whose occupancy respects the limit. Occupancy is
    non-increasing along the sequence (merging levels can only merge
    cells), so that state is found by bisection over the step count with
    O(log steps) occupancy evaluations rather than one per step - a
    column with thousands of levels would otherwise cost thousands of
    passes over the rows.
    """
    n_rows, n_cols = arr.shape
    if n_rows == 0:
        return arr, [0] * n_cols
    if orders is None:
        orders = [level_frequency_order(arr[:, j]) for j in range(n_cols)]
    if ordered is None:
        ordered = [False] * n_cols
    k0 = [int(o.shape[0]) for o in orders]
    if occupied_cells(arr) <= target_max_cells:
        return arr, k0

    def build(k: List[int]) -> np.ndarray:
        return np.column_stack([
            coarsen_column(arr[:, j], k[j], orders[j], ordered=bool(ordered[j])) if k[j] < k0[j] else arr[:, j]
            for j in range(n_cols)
        ])

    total_steps = sum(max(0, v - 1) for v in k0)
    k_last = merge_state(k0, total_steps)
    joint_last = build(k_last)
    if occupied_cells(joint_last) > target_max_cells:
        return joint_last, k_last  # every column constant: nothing more to merge
    lo, hi = 0, total_steps  # occ(lo) > cap, occ(hi) <= cap
    best = (joint_last, k_last)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        k_mid = merge_state(k0, mid)
        joint_mid = build(k_mid)
        if occupied_cells(joint_mid) <= target_max_cells:
            hi, best = mid, (joint_mid, k_mid)
        else:
            lo = mid
    return best
