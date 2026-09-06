"""
VSF PMD: categorical encoding and grid-capacity control.

VSF is a CATEGORICAL framework. Every feature column is a set of discrete
categories, and every value in it -- string, boolean, integer or float -- is
one category, encoded by its position in the column's sorted distinct
values. There is no binning, no continuous-feature model and no
rate-distortion machinery here: a numeric column is not "quantized into k
intervals", it is read as the finite set of values it actually contains.

What this module used to do, and why it is gone
-----------------------------------------------
`discretize_feature` used to branch on dtype: strings and booleans were
label-encoded (as above), while numeric columns went through a
Freedman-Diaconis bin count, quantile cut points, per-channel level caps
(`CHANNEL_LIMITS`), and a rate-distortion "distortion" figure
`D_j = 1 - I(X_discrete; X_fine) / H(X_fine)` computed with the plug-in
mutual information of `vsf.metrics`. That was the last consumer of mutual
information anywhere in this package.

It was removed rather than kept behind a flag, for three reasons, in the
order they matter:

1.  It contradicted the product. A discrete centre is a statement about a
    combination of category values; an interval boundary chosen from the
    data by a density heuristic is not a category, and an axis tick reading
    "(3.7, 4.1]" is not a value the analyst can act on.
2.  Nothing consumed its output. `D_j` was computed for every column on
    every analysis and discarded at every call site
    (`X_discrete, bin_counts, _ = discretize_dataset(...)` in `vsf.avr`).
    It was pure cost.
3.  It was an uncorrected statistical exposure. Bin edges chosen from the
    same data that the search then ranks on are a selection effect nobody
    was accounting for. Encoding distinct values removes it by
    construction, since nothing about the encoding depends on the target.

Consequence, stated plainly: a genuinely continuous column (thousands of
distinct floats) now produces thousands of categories rather than <=200
quantile bins. It is not silently rejected and not silently truncated -- it
is encoded exactly as it stands, and `adaptively_coarsen_bins` below is what
keeps the joint grid estimable when such a column enters a candidate
combination. Feeding continuous measurements to a categorical framework
gives an axis with thousands of ticks; that is a data-preparation decision
for the caller, made visible rather than papered over.

What remains here is grid-capacity control: `prod(k_j) <= N / 10`, enforced
per candidate combination by merging levels, so that cell counts stay
estimable in the exhaustive search (Project_Master_Document.md Section 2.3).
"""

import numpy as np

__all__ = [
    "adaptively_coarsen_bins",
    "check_grid_capacity",
    "coarsen_column",
    "discretize_dataset",
    "discretize_feature",
    "max_bins_per_dimension",
]


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


def check_grid_capacity(bin_counts: list[int], n_samples: int) -> bool:
    """
    Checks the hypervolume grid capacity prod(k_j) <= N / 10
    (Project_Master_Document.md Section 2.3). Returns True when the limit is
    satisfied, False when it is exceeded.
    """
    total_cells = int(np.prod(bin_counts))
    max_allowed = max(1, n_samples // 10)
    return total_cells <= max_allowed


def adaptively_coarsen_bins(
    discrete_features: np.ndarray, n_samples: int, target_max_cells: int | None = None
) -> np.ndarray:
    """
    Merges category levels until the joint grid satisfies
    prod(k_j) <= N / 10, so that the cell counts the search is ranked on stay
    estimable. Returns the input unchanged when the limit already holds.
    """
    arr = np.asarray(discrete_features)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    n_rows, n_cols = arr.shape
    if target_max_cells is None:
        target_max_cells = max(1, n_samples // 10)

    k_counts = [len(np.unique(arr[:, j])) for j in range(n_cols)]
    prod_k = np.prod(k_counts)

    if prod_k <= target_max_cells:
        return arr

    k_max_per_dim = max_bins_per_dimension(n_cols, n_samples, target_max_cells)

    coarsened = np.zeros_like(arr)
    for j in range(n_cols):
        coarsened[:, j] = coarsen_column(arr[:, j], k_max_per_dim)

    return coarsened


def max_bins_per_dimension(n_cols: int, n_samples: int, target_max_cells: int | None = None) -> int:
    """
    The per-axis level cap `adaptively_coarsen_bins` applies to a `n_cols`-
    dimensional grid: the largest k with k ** n_cols <= max(1, N // 10)
    (or `target_max_cells`). Exposed so callers that score many subsets of
    the same dimensionality can coarsen each column once instead of once per
    subset - the cap depends only on (n_cols, N), never on which columns are
    combined.
    """
    if target_max_cells is None:
        target_max_cells = max(1, n_samples // 10)
    k_max = max(1, int(np.floor(target_max_cells ** (1.0 / n_cols))))
    if (k_max ** n_cols) > target_max_cells and k_max > 1:
        k_max = max(1, k_max - 1)
    return k_max


def coarsen_column(col: np.ndarray, k_max: int) -> np.ndarray:
    """
    One column of `adaptively_coarsen_bins`: if the column has more than
    `k_max` distinct levels, its sorted distinct values are split into
    `k_max` near-equal consecutive groups (`np.array_split`) and each value
    is replaced by its group index; otherwise the column is returned
    unchanged. `k_max == 1` collapses the column to a constant 0.

    Merging ADJACENT codes means merging adjacent entries of the column's
    sorted distinct values, which is meaningful for an ordered category set
    and arbitrary (though deterministic) for an unordered one. It is applied
    only when a candidate combination would otherwise exceed the capacity
    limit, i.e. only where the alternative is scoring cells that hold a
    handful of rows each.

    Implemented as a lookup table indexed by the (non-negative integer)
    level, which is what `discretize_dataset` produces; the previous
    `np.vectorize(dict.get)` did the same mapping one Python call per row.
    """
    col = np.asarray(col)
    unique_vals = np.unique(col)
    if len(unique_vals) <= k_max:
        return col
    if k_max == 1:
        return np.zeros_like(col)
    groups = np.array_split(unique_vals, k_max)
    if col.dtype.kind in ("i", "u") and unique_vals.min() >= 0:
        lut = np.empty(int(unique_vals.max()) + 1, dtype=col.dtype)
        for group_idx, grp in enumerate(groups):
            lut[grp] = group_idx
        return lut[col]
    val_map = {}
    for group_idx, grp in enumerate(groups):
        for val in grp.tolist():
            val_map[val] = group_idx
    return np.vectorize(val_map.get, otypes=[col.dtype])(col)


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
