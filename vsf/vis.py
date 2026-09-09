"""
VSF Visualization Module: Payload Generator & Standalone HTML Exporter
Prepares 3D visual coordinates, HUD metrics, and generates interactive WebGL scatter plots.

v2.0 note: `prepare_visualization_payload` now takes a single `BranchResult`
(Project_Master_Document.md Section 4.2) rather than a v1.0 `AVRResult` —
called once per independently-discovered branch (d in {1,2,3,4}), never once
per dataset. The grid-building machinery below (`build_grid`, the 1D/2D/3D
marginals, the 4D film-strip slices) is unchanged: it already operates on
whatever feature subset `branch.selected_features` names, regardless of
whether that subset is nested inside another branch's subset — see
Project_Master_Document.md Section 5.6 for why that distinction matters for
the collapse/split animation built on top of this payload.
"""

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple
from .avr import BranchResult
from .centers import CenterSpec, purity_bounds, select_centers

# Shape of an optional dataset translation table accepted throughout this
# module: {"columns": {raw_col_name: display_name},
#          "values": {raw_col_name: {raw_value_str: display_value}}}.
# This module carries NO built-in dataset vocabulary of its own — it is
# generic over whatever dataset it is pointed at. For a worked example (the
# UCI Mushroom dataset translations previously hardcoded here), see
# `examples/mushroom_demo.py`; pass that dict as `translations` to get the
# old behavior back. With no `translations` supplied, every function in this
# module falls back to the raw column/value strings unchanged.
Translations = Dict[str, Dict]


def humanize_val(col_name: str, val, translations: Optional[Translations] = None) -> str:
    """
    Translates a raw category code to a human-readable display value, if a
    `translations` table is supplied (see `Translations` above). With no
    `translations` (the default), returns the raw value as a string.
    """
    val_str = str(val)
    if not translations:
        return val_str
    values_for_col = translations.get("values", {}).get(col_name)
    if values_for_col:
        return values_for_col.get(val_str, val_str)
    return val_str


def humanize_col(col_name: str, translations: Optional[Translations] = None) -> str:
    """
    Translates a column name to a "Display Name (raw_name)" label, if a
    `translations` table is supplied (see `Translations` above). With no
    `translations` (the default), returns `col_name` unchanged.
    """
    if not translations:
        return col_name
    display_name = translations.get("columns", {}).get(col_name, col_name)
    return f"{display_name} ({col_name})" if display_name != col_name else col_name


def catalog_from_dataframe(df, translations: Optional[Translations] = None) -> List[Dict]:
    """
    Builds the `{id, label, criteria: [{id, label}, ...]}` catalog listing
    — every column of `df` paired with its observed unique values — shared
    by the live server's `/api/columns` endpoint and `vsf.dashboard`'s
    static catalog tree, so both stay byte-for-byte consistent instead of
    hand-rolling this loop twice. `df` is any `pandas.DataFrame`; humanized
    via `humanize_val`/`humanize_col` when `translations` is supplied, and
    falls back to raw column/value strings with none, matching every other
    function in this module.
    """
    catalog = []
    n_rows = int(len(df))
    for col in df.columns:
        label = humanize_col(col, translations)
        # Values ordered by how much of the column they occupy (ties by
        # string, for a stable order), each with its share of ALL rows:
        # the number the search direction and the admissible tau depend
        # on (`vsf.avr.base_rate_reason`), shown in the catalog before a
        # value is picked rather than after. Shares are of `n_rows`, so a
        # column with missing values sums to less than 1.
        counts = df[col].dropna().astype(str).value_counts()
        ordered = sorted(
            counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))
        )
        criteria = [
            {
                "id": val,
                "label": humanize_val(col, val, translations),
                "count": int(cnt),
                "share": (int(cnt) / n_rows) if n_rows else 0.0,
            }
            for val, cnt in ordered
        ]
        catalog.append({"id": col, "label": label, "criteria": criteria})
    return catalog


def _axis_fallback_indices(selected_idx: List[int], n_features: int, count: int) -> List[int]:
    """
    Returns `count` distinct feature-column indices to use as display axes:
    the first `count` entries of `selected_idx` (in AVR's chosen order),
    padded out with the lowest-index columns NOT already used, when AVR
    selected fewer than `count` features (d* < count, e.g. d* = 1 while the
    caller wants 3 scatter axes).

    An earlier version padded with the raw literal indices 1 and 2
    (`selected_idx[1] if len(selected_idx) > 1 else 1`, `... else 2`)
    regardless of what `selected_idx[0]` actually was. Whenever the single
    selected feature happened to BE column 1 or column 2, the "fallback"
    axis silently duplicated the primary axis — e.g. `selected_idx = [1]`
    produced `x_col_idx = y_col_idx = 1`, collapsing the Y axis onto the X
    axis and rendering a degenerate 2D plane instead of a 3D scatter, with
    no indication to the user that this had happened.
    """
    idx_list: List[int] = []
    used = set()
    for idx in selected_idx:
        if len(idx_list) >= count:
            break
        idx_list.append(int(idx))
        used.add(int(idx))

    candidate = 0
    while len(idx_list) < count and candidate < n_features:
        if candidate not in used:
            idx_list.append(candidate)
            used.add(candidate)
        candidate += 1

    # Degenerate case: fewer feature columns exist than axes requested.
    # Repeat the last valid index rather than raising, since the caller
    # (a 3D scatter builder) must always receive `count` indices.
    while len(idx_list) < count:
        idx_list.append(idx_list[-1] if idx_list else 0)

    return idx_list


def target_conditioned_sort(
    x_vals: np.ndarray,
    z_vals: np.ndarray,
    col_name: str = "",
    translations: Optional[Translations] = None,
):
    """
    Sorts categories of x_vals based on their association with the target z_vals.
    Uses deterministic tie-breaking (by frequency and lexical order) to prevent spatial warping.
    Returns:
        x_num: Integer coordinates for x_vals
        x_sorted_labels: List of human-readable category string labels in sorted order (for axis ticks)
    """
    _, z_idx = np.unique(z_vals, return_inverse=True)
    x_unique, x_counts = np.unique(x_vals, return_counts=True)

    cat_stats = []
    for c, cnt in zip(x_unique, x_counts):
        mask = (x_vals == c)
        score = float(np.mean(z_idx[mask])) if np.any(mask) else 0.0
        cat_stats.append((score, -int(cnt), str(c), c))

    cat_stats.sort(key=lambda item: (item[0], item[1], item[2]))
    x_sorted = [item[3] for item in cat_stats]

    x_to_num = {val: i for i, val in enumerate(x_sorted)}
    x_num = np.array([x_to_num[val] for val in x_vals])

    # Translate tick labels via the supplied translation table, if any.
    x_labels = [humanize_val(col_name, v, translations) for v in x_sorted]
    return x_num, x_labels


def target_class_labels(
    unique_values: np.ndarray,
    col_name: str,
    display_name: str,
    translations: Optional[Translations] = None,
    indicator: bool = False,
) -> List[str]:
    """
    Human-readable labels for a target's distinct values.

    `col_name` is the RAW column name, used to look the values up in
    `translations`; `display_name` is its already-humanized form, used only
    for the indicator special-case below.

    `indicator=True` marks the One-vs-Rest case produced by `/api/analyze`'s
    `criterion` path, where the target vector is literally 0/1 and
    `display_name` already reads "Column = value". Running those through
    `humanize_val` yields the labels "0" and "1" -- which is what the legend
    showed, and what the v2.1 per-class breakdown would otherwise have shown
    too; they are relabelled as the criterion and its negation instead. The
    flag is passed explicitly by the caller that BUILT the indicator rather
    than sniffed from the values, because a genuine 0/1 data column is
    indistinguishable from a derived indicator by inspection and must keep
    its own labels.
    """
    arr = np.asarray(unique_values)
    is_indicator = (
        indicator
        and arr.dtype.kind in ("b", "i", "u")
        and set(int(v) for v in arr.tolist()) <= {0, 1}
    )
    if is_indicator:
        return [display_name if int(v) == 1 else f"not {display_name}" for v in arr.tolist()]
    return [humanize_val(col_name, v, translations) for v in arr.tolist()]


def _cell_keys(columns: Sequence[np.ndarray]) -> List[tuple]:
    """Row-wise hashable cell keys from a list of equal-length raw columns."""
    as_str = [np.asarray(c).astype(str) for c in columns]
    return list(zip(*as_str))


class _FullCellStatistics:
    """
    Per-cell (n, k) counts over the FULL dataset, plus the certificate.

    Why the full dataset and not the rendered subsample: `build_grid` groups
    the up-to-`max_display_samples` rows that survive subsampling, so before
    v2.2 the panel's "Total Samples: 10 000" sat next to statistics computed
    by `vsf.avr` on all 32 561 rows, and the cell purities drawn on screen
    were computed on a 30 % sample of each cell. Both numbers were defensible
    in isolation and their combination was not reproducible: on
    the UCI Adult / Census Income dataset with target `occupation = Armed-Forces`, three of
    the nine positives are absent from the default subsample entirely, so the
    displayed purity of the cell holding them is a different quantity from
    the one behind the reported metric.

    The scatter points still come from the subsample (drawing 32 561 spheres
    is a rendering decision), but every NUMBER attached to a cell - its count,
    its purity, its confidence bound, whether it is certified - is computed
    here on all rows.
    """

    def __init__(
        self,
        columns: Sequence[np.ndarray],
        positive_mask: np.ndarray,
        spec: CenterSpec,
        row_mask: Optional[np.ndarray] = None,
    ) -> None:
        # Cell coding is done by `np.unique(..., axis=0)` on the stacked string
        # columns rather than by building N Python tuples: this runs once per
        # view dimensionality per branch, plus once per 4-D slice, on the FULL
        # table, and the tuple-and-dict version spent most of a click's
        # latency there. The lookup dict is built from the DISTINCT rows only
        # (hundreds) instead of from all N of them.
        pos = np.asarray(positive_mask).astype(bool)
        str_cols = [np.asarray(c).astype(str) for c in columns]
        if row_mask is not None:
            keep = np.asarray(row_mask).astype(bool)
            str_cols = [c[keep] for c in str_cols]
            pos = pos[keep]
        n_rows = int(str_cols[0].shape[0]) if str_cols else 0
        if n_rows == 0:
            uniq_rows: List[tuple] = []
            codes = np.zeros(0, dtype=np.int64)
        else:
            # Row-wise `np.unique(matrix, axis=0)` sorts N string tuples;
            # coding each column with its own (sorted) `np.unique` and
            # combining the per-column codes mixed-radix gives the SAME
            # ordering of the distinct rows - lexicographic on the string
            # tuple, because per-column codes are monotone in the strings -
            # so the cell numbering, and with it every downstream index, is
            # unchanged, at the cost of one integer sort instead of a string
            # one.
            levels: List[np.ndarray] = []
            flat = None
            for c in str_cols:
                lv, code = np.unique(c, return_inverse=True)
                code = np.asarray(code, dtype=np.int64).ravel()
                levels.append(lv)
                flat = code if flat is None else flat * int(lv.shape[0]) + code
            assert flat is not None
            uniq_flat, codes = np.unique(flat, return_inverse=True)
            codes = np.asarray(codes, dtype=np.int64).ravel()
            # Decode the distinct flat codes back into their string tuples.
            digits: List[np.ndarray] = []
            rest = uniq_flat.copy()
            for lv in reversed(levels):
                radix = int(lv.shape[0])
                digits.append(lv[rest % radix])
                rest //= radix
            digits.reverse()
            uniq_rows = list(zip(*[d.tolist() for d in digits]))
        index: Dict[tuple, int] = {row: i for i, row in enumerate(uniq_rows)}
        n_cells = len(index)
        self.index = index
        self.n_per_cell = np.bincount(codes, minlength=n_cells).astype(np.int64)
        self.k_per_cell = np.rint(
            np.bincount(codes, weights=pos.astype(np.float64), minlength=n_cells)
        ).astype(np.int64)
        self.n_samples = int(self.n_per_cell.sum())
        self.n_positive = int(self.k_per_cell.sum())
        self.spec = spec
        self.certified, self.alpha_effective = select_centers(
            self.k_per_cell, self.n_per_cell, spec
        )
        self.lower, self.upper = purity_bounds(
            self.k_per_cell, self.n_per_cell, self.alpha_effective, spec.method
        )

    def lookup(self, key: tuple) -> Dict[str, float]:
        """Full-data statistics for one displayed cell, by its raw-value key."""
        i = self.index.get(key)
        if i is None:  # a cell that exists only in the render subsample
            return {
                "n": 0, "k": 0, "purity": 0.0, "purity_lower": 0.0,
                "purity_upper": 1.0, "certified": False,
            }
        n = int(self.n_per_cell[i])
        k = int(self.k_per_cell[i])
        return {
            "n": n,
            "k": k,
            "purity": (k / n) if n > 0 else 0.0,
            "purity_lower": float(self.lower[i]),
            "purity_upper": float(self.upper[i]),
            "certified": bool(self.certified[i]),
        }

    def summary(self) -> Dict[str, float]:
        """Coverage / K / pooled purity for this partition, over all rows."""
        mask = self.certified
        k_sel = int(self.k_per_cell[mask].sum())
        n_sel = int(self.n_per_cell[mask].sum())
        prevalence = self.n_positive / self.n_samples if self.n_samples else 0.0
        purity = (k_sel / n_sel) if n_sel else 0.0
        return {
            "n_centers": int(mask.sum()),
            "coverage": (k_sel / self.n_positive) if self.n_positive else 0.0,
            "purity_pooled": purity,
            "mass": (n_sel / self.n_samples) if self.n_samples else 0.0,
            "lift": (purity / prevalence) if prevalence > 0 else 0.0,
            "n_cells_occupied": int(np.count_nonzero(self.n_per_cell > 0)),
            "max_purity_point": float(
                np.max(self.k_per_cell / np.maximum(self.n_per_cell, 1))
            ) if self.n_per_cell.size else 0.0,
            "max_purity_lower": float(self.lower.max()) if self.lower.size else 0.0,
        }


def prepare_visualization_payload(
    branch: BranchResult,
    X_matrix: np.ndarray,
    Z_target: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_display_samples: int = 10000,
    target_name: str = "class",
    target_is_indicator: bool = False,
    sort_Z: Optional[np.ndarray] = None,
    translations: Optional[Translations] = None,
    positive_value: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    indicator_labels: Optional[Tuple[str, str]] = None,
) -> Dict:
    """
    Prepares a structured visualization payload with axis titles, category
    labels, color mappings, and cluster occupancy density counts, for ONE
    independently-discovered branch (`vsf.avr.BranchResult`) — call this
    once per branch the caller wants to render (typically once per
    dimensionality 1..4 returned by `vsf.avr.discover_branches`).

    `translations` is an optional dataset-specific display table (see the
    `Translations` type alias above) — pass it to get human-readable column
    and value labels; with no `translations`, every label in the payload is
    the raw column/value string as it appears in the input data. This
    function has no built-in knowledge of any particular dataset.

    `positive_value` is the target value that cell purity is measured
    against. Defaults to `np.unique(Z_target)[-1]`, which is the literal 1 of
    `/api/analyze`'s One-vs-Rest `criterion` path and the class this module
    has always labelled as positive; pass it explicitly for a K-class target,
    where the highest `np.unique` code is positive only by accident of
    ordering.

    `center_spec` is the certificate (`vsf.centers.CenterSpec`) applied to
    every displayed cell: the purity floor `tau`, the simultaneous error rate
    `alpha`, and the multiplicity policy. It drives the `certified` flag the
    frontend colours by, and the coverage / K / purity triple the panel
    reports. Defaults to tau = 0.90, alpha = 0.05, Bonferroni across the
    displayed partition's occupied cells.

    NOTE on which partition is certified: the certificate here is computed on
    the DISPLAYED partition — the cells the user can see and click — over ALL
    rows of `X_matrix`, not the render subsample. `vsf.avr` certifies the
    SEARCH partition, which is the PMD-discretised one. The two coincide
    whenever no displayed feature was binned or adaptively coarsened (i.e.
    for any all-categorical dataset); when they do not, `metrics.centers`
    carries the search-partition numbers alongside and
    `metrics.certificate.partition_matches_search` is False. The panel shows
    the displayed-partition numbers, because those describe the object on
    screen.
    """
    X_arr = np.asarray(X_matrix)
    Z_arr = np.asarray(Z_target).ravel()
    n_samples, n_features = X_arr.shape

    if feature_names is None:
        feature_names = [f"Feature_{j+1}" for j in range(n_features)]

    if n_samples > max_display_samples:
        indices = np.random.default_rng(42).choice(n_samples, size=max_display_samples, replace=False)
        X_sub = X_arr[indices]
        Z_sub = Z_arr[indices]
    else:
        indices = np.arange(n_samples)
        X_sub = X_arr
        Z_sub = Z_arr

    if sort_Z is not None:
        sort_Z_arr = np.asarray(sort_Z).ravel()
        sort_Z_sub = sort_Z_arr[indices] if len(sort_Z_arr) == n_samples else sort_Z_arr
    else:
        sort_Z_sub = Z_sub

    selected_idx = branch.selected_features
    branch_d = branch.d

    x_col_idx, y_col_idx, z_col_idx = _axis_fallback_indices(selected_idx, n_features, 3)

    # 4th dimension (slice axis) — extracted when this branch is 4D. Unlike
    # v1.0's `d_star >= 4` (a variable stopping point of one greedy chain),
    # `branch.d` is always exactly `len(branch.selected_features)` by
    # construction (see `discover_branches`), so the two conditions below
    # are equivalent; the explicit length check is kept as a defensive
    # invariant check, not because it can diverge in practice.
    has_4d = branch_d >= 4 and len(selected_idx) >= 4
    w_col_idx = selected_idx[3] if has_4d else None

    x_name = feature_names[x_col_idx]
    y_name = feature_names[y_col_idx]
    z_name = feature_names[z_col_idx]
    w_name = feature_names[w_col_idx] if has_4d else None

    x_vals = X_sub[:, x_col_idx]
    y_vals = X_sub[:, y_col_idx]
    z_vals = X_sub[:, z_col_idx]
    w_vals = X_sub[:, w_col_idx] if has_4d else None

    # Human-readable value strings
    x_human = [humanize_val(x_name, v, translations) for v in x_vals]
    y_human = [humanize_val(y_name, v, translations) for v in y_vals]
    z_human = [humanize_val(z_name, v, translations) for v in z_vals]
    # NOT built: a `w_human` per-point label for the 4th axis. Every point
    # rendered under a given 4D slice tab already shares that slice's single
    # w-value (the tab itself displays it, e.g. "4D Slice: sex -> Female"),
    # so a per-point line would only repeat on-screen information -- the same
    # redundant-info reasoning already applied to the certified-centre hover
    # box below (its 2026-09 trim; see `hov` in `build_grid`).
    # One label map, built from the FULL target vector and reused for the
    # per-sample labels, the legend and the per-class breakdown, so the three
    # can never disagree.
    _target_display = humanize_col(target_name, translations)
    _all_target_values = np.unique(np.asarray(Z_target).ravel())
    _target_label_map = dict(zip(
        [_v.item() if hasattr(_v, "item") else _v for _v in _all_target_values],
        (
            # Explicit (label for 0, label for 1) - used by the absence
            # search, whose indicator is the COMPLEMENT of the criterion:
            # 1 must read "not Column = value" and 0 "Column = value".
            [indicator_labels[int(v)] for v in _all_target_values.tolist()]
            if (indicator_labels is not None and target_is_indicator
                and set(int(v) for v in _all_target_values.tolist()) <= {0, 1})
            else target_class_labels(
                _all_target_values, target_name, _target_display, translations,
                indicator=target_is_indicator,
            )
        ),
    ))

    def _label_of(value) -> str:
        key = value.item() if hasattr(value, "item") else value
        return _target_label_map.get(key, humanize_val(target_name, value, translations))

    z_target_human = [_label_of(v) for v in Z_sub]

    # Target-Conditioned Categorical Ordering using canonical sort_Z_sub
    x_num, x_ticks = target_conditioned_sort(x_vals, sort_Z_sub, col_name=x_name, translations=translations)
    y_num, y_ticks = target_conditioned_sort(y_vals, sort_Z_sub, col_name=y_name, translations=translations)
    z_num, z_ticks = target_conditioned_sort(z_vals, sort_Z_sub, col_name=z_name, translations=translations)
    if has_4d:
        w_num, w_ticks = target_conditioned_sort(w_vals, sort_Z_sub, col_name=w_name, translations=translations)
    else:
        w_num, w_ticks = None, None

    unique_targets, color_num = np.unique(Z_sub, return_inverse=True)
    unique_target_labels = [_label_of(t) for t in unique_targets]

    # Group sample indices by cell for 3D Voxel Crystal Lattice Packing
    from collections import defaultdict, Counter
    cell_groups = defaultdict(list)
    for idx_in_sub, (cx, cy, cz) in enumerate(zip(x_num, y_num, z_num)):
        cell_groups[(cx, cy, cz)].append(idx_in_sub)

    x_cube = np.zeros(len(x_num), dtype=float)
    y_cube = np.zeros(len(y_num), dtype=float)
    z_cube = np.zeros(len(z_num), dtype=float)

    for (cx, cy, cz), cell_indices in cell_groups.items():
        # Sort samples within cell by target class Z so colors form clean stratified layers in the cube
        cell_indices.sort(key=lambda idx: color_num[idx])
        N_cell = len(cell_indices)
        
        # Grid edge dimension S (cube root)
        S = int(np.ceil(N_cell ** (1.0 / 3.0)))
        if S <= 1:
            step = 0.0
        else:
            # Clean spacing so individual spheres are clearly visible with gaps
            step = min(0.052, 0.78 / max(S - 1, 1))

        # 3D lattice indices (i, j, k) of every sample's rank within the
        # cell, vectorised per cell: the same integer and floating-point
        # expressions as the previous per-sample loop, evaluated on arrays
        # (an int64 - float subtraction and a float multiply are the same
        # IEEE operations either way, and `np.round` is the same ufunc).
        rank = np.arange(N_cell, dtype=np.int64)
        i = rank % S
        j = (rank // S) % S
        k = rank // (S * S)
        half = (S - 1) / 2.0
        members = np.asarray(cell_indices, dtype=np.int64)
        x_cube[members] = np.round(cx + (i - half) * step, 4)
        y_cube[members] = np.round(cy + (j - half) * step, 4)
        z_cube[members] = np.round(cz + (k - half) * step, 4)

    coords = list(zip(x_num, y_num, z_num))
    cell_counts = Counter(coords)

    target_display_name = humanize_col(target_name, translations)

    hover_texts = [
        f"<b>🍄 Sample #{idx+1}</b><br>"
        f"🎯 <b>{target_display_name}:</b> {z_target_human[i]}<br>"
        f"📍 <b>{humanize_col(x_name, translations)}:</b> {x_human[i]}<br>"
        f"📍 <b>{humanize_col(y_name, translations)}:</b> {y_human[i]}<br>"
        f"📍 <b>{humanize_col(z_name, translations)}:</b> {z_human[i]}<br>"
        f"📦 <b>Cube Volume (Density):</b> {cell_counts[(x_num[i], y_num[i], z_num[i])]} samples in cell"
        for i, idx in enumerate(indices)
    ]

    # ---- full-data cell statistics (v2.2) --------------------------------
    # Computed on every row of X_matrix, independently of the render
    # subsample; see `_FullCellStatistics` for why the previous
    # subsample-based purities were not reproducible against the reported
    # metrics.
    spec = center_spec if center_spec is not None else CenterSpec()
    _all_targets_sorted = np.unique(Z_arr)
    if positive_value is None:
        positive_value = (
            _all_targets_sorted[-1] if _all_targets_sorted.size else None
        )
    positive_mask_full = (
        np.asarray(Z_arr).astype(str) == str(positive_value)
        if positive_value is not None
        else np.zeros(n_samples, dtype=bool)
    )
    x_full = X_arr[:, x_col_idx]
    y_full = X_arr[:, y_col_idx]
    z_full = X_arr[:, z_col_idx]
    w_full = X_arr[:, w_col_idx] if has_4d else None
    _full_cols = {1: [x_full], 2: [x_full, y_full], 3: [x_full, y_full, z_full]}
    cell_stats: Dict[str, _FullCellStatistics] = {
        str(dim): _FullCellStatistics(cols, positive_mask_full, spec)
        for dim, cols in _full_cols.items()
    }

    def build_grid(dim, slice_mask=None, stats_key=None):
        """
        Build discrete center grid. If slice_mask is provided, only include
        samples where slice_mask[i] is True (subsample indexing).

        `stats_key` names the entry of `cell_stats` whose FULL-DATA counts,
        bounds and certification decorate each cell. Geometry and the drawn
        point cloud come from the subsample; every reported number comes from
        `cell_stats`.
        """
        stats = cell_stats[stats_key if stats_key is not None else str(dim)]
        g_groups = defaultdict(list)
        for idx_in_sub, (cx, cy, cz) in enumerate(zip(x_num, y_num, z_num)):
            if slice_mask is not None and not slice_mask[idx_in_sub]:
                continue
            _cy = cy if dim >= 2 else -0.5
            _cz = cz if dim >= 3 else -0.5
            g_groups[(cx, _cy, _cz)].append(idx_in_sub)
            
        gx, gy, gz, gop, gpur, gsz, ghov, gdata = [], [], [], [], [], [], [], []
        glow, gup, gcert, gk = [], [], [], []
        cell_keys = {}
        for (cx, cy, cz), c_idx in g_groups.items():
            i0 = c_idx[0]
            key = [str(x_vals[i0])]
            if dim >= 2:
                key.append(str(y_vals[i0]))
            if dim >= 3:
                key.append(str(z_vals[i0]))
            cell_keys[(cx, cy, cz)] = tuple(key)
        m_N = max(
            [stats.lookup(cell_keys[k])["n"] for k in g_groups] or [1]
        ) or 1
        
        for (cx, cy, cz), c_idx in g_groups.items():
            cell = stats.lookup(cell_keys[(cx, cy, cz)])
            N_c = cell["n"]
            n_rendered = len(c_idx)
            # Share of the designated "positive" class (the highest-index
            # class, `unique_targets[-1]`) among this cell's samples -- this
            # is what the hover's Confidence bound is a bound ON (the point
            # estimate itself is no longer shown in the hover text, 2026-09
            # trim, but this value still drives `gpur`/cell colour and the
            # certification decision below), for any number of target
            # classes K.
            #
            # The prior formula, `mean(c_cols) / max(K - 1, 1)`, is the
            # cell's AVERAGE class INDEX normalized to [0, 1] — that only
            # coincides with "share of the positive class" when K = 2 (where
            # mean(c_cols) IS already the fraction of class-index 1). For
            # K = 3 a cell that is 100% the MIDDLE class (index 1, not the
            # positive class at index 2) averaged to 1 / 2 = 0.5, reporting
            # "50% positive" for a cell containing ZERO positive-class
            # samples — not a purity measure at all for K > 2, and
            # incompatible with the frontend's discrete 4-zone purity color
            # banding (`getColorIndexForCell` in webapp/static/js/app.js, see
            # Project_Master_Document.md Section 5.3), which assumes this
            # value IS a positive-class probability.
            #
            # v2.2: `pur` is now the FULL-DATA share of `positive_value`, and
            # is accompanied by its simultaneous confidence bounds and the
            # certification flag the renderer colours by. The point estimate
            # alone is what made a cell holding one sample of one class draw
            # as a 100 %-pure "green centre"; the lower bound of that cell is
            # `alpha_effective`, so it can never be certified, and the
            # "minimum samples" slider that used to hide it is redundant.
            pur = float(cell["purity"])
            norm_d = 0.2 + 0.8 * (np.sqrt(N_c) / np.sqrt(m_N)) if N_c > 0 else 0.2

            gx.append(float(cx))
            gy.append(float(cy))
            gz.append(float(cz))
            gop.append(float(norm_d))
            gpur.append(float(pur))
            gsz.append(int(N_c))
            glow.append(float(cell["purity_lower"]))
            gup.append(float(cell["purity_upper"]))
            gcert.append(bool(cell["certified"]))
            gk.append(int(cell["k"]))

            
            # v2.2 hover trim (2026-09, user request): title, the CERTIFIED
            # verdict, and X/Y/Z/4D-slice were dropped as redundant with
            # information already on screen when this box is open -- cell
            # colour already encodes certified/mixed/low, and the axis
            # labels plus the currently selected slice tab already identify
            # which cell this is. The point-estimate purity line was
            # dropped in that same trim but restored below (2026-09,
            # follow-up user request): the confidence bound alone makes the
            # user do the (lower+upper)/2 arithmetic themselves to see what
            # share of the data they care about actually sits in this
            # centre -- showing `pur` directly answers that in one read.
            # `n_rendered` is appended only when the plot is subsampling for
            # render (it then differs from `N_c`, the full-data count
            # everything else here is computed on) -- in the common
            # unsampled case the two are equal and repeating both is
            # exactly the kind of duplication this trim removes.
            hov = (
                f"🎯 <b>Purity:</b> {pur*100:.1f}%<br>"
                f"📏 <b>Confidence bound:</b> [{cell['purity_lower']*100:.1f}%, "
                f"{cell['purity_upper']*100:.1f}%]<br>"
                f"📦 <b>Objects:</b> {N_c} pcs."
            )
            if n_rendered != N_c:
                hov += f" ({n_rendered} drawn)"
            ghov.append(hov)
            
            # Use raw values for API queries so filtering works, cast to standard types for JSON
            def _cast(val):
                if hasattr(val, 'item'): return val.item()
                return val
            
            rx = _cast(x_vals[c_idx[0]])
            ry = _cast(y_vals[c_idx[0]]) if dim >= 2 else None
            rz = _cast(z_vals[c_idx[0]]) if dim >= 3 else None
            rw = _cast(w_vals[c_idx[0]]) if (has_4d and w_vals is not None and slice_mask is not None) else None
            
            cdata = {
                "N": int(N_c),
                "k": int(cell["k"]),
                "pur": float(pur),
                "lower": float(cell["purity_lower"]),
                "upper": float(cell["purity_upper"]),
                "certified": bool(cell["certified"]),
                "coords": {x_name: rx},
            }
            if dim >= 2: cdata["coords"][y_name] = ry
            if dim >= 3: cdata["coords"][z_name] = rz
            if rw is not None: cdata["coords"][w_name] = rw
            gdata.append(cdata)
            
        return {
            "x": gx, "y": gy, "z": gz, "opacity": gop, "purity": gpur,
            "purity_lower": glow, "purity_upper": gup, "certified": gcert,
            "positives": gk, "sizes": gsz, "hover_text": ghov,
            "customdata": gdata, "summary": stats.summary(),
        }

    # Global max points per cell, from the 1-D FULL-data counts, so marker
    # scaling is a property of the dataset rather than of which rows the
    # renderer happened to sample.
    global_max_n = (
        int(cell_stats["1"].n_per_cell.max())
        if cell_stats["1"].n_per_cell.size
        else 1
    )

    grids = {
        "1": build_grid(1),
        "2": build_grid(2),
        "3": build_grid(3),
    }

    # Generate per-slice grids for 4D visualization
    slice_axis_info = None
    if has_4d and w_num is not None and w_ticks is not None and w_full is not None:
        # Map the raw 4th-axis value onto its slice index using the
        # subsample's own ordering, then apply that map to the FULL column so
        # each slice's certificate is computed over every row in that slice,
        # not over the sampled ones. A raw value absent from the subsample has
        # no slice tab and is therefore in no slice, which is the same set the
        # user can navigate.
        slice_of_value = {
            str(v): int(i) for v, i in zip(w_vals, w_num)
        }
        w_full_str = np.asarray(w_full).astype(str)
        slice_index_full = np.array(
            [slice_of_value.get(v, -1) for v in w_full_str.tolist()], dtype=np.int64
        )
        slice_counts = []
        for sv_idx in range(len(w_ticks)):
            mask = (w_num == sv_idx)
            key = f"4_{sv_idx}"
            cell_stats[key] = _FullCellStatistics(
                [x_full, y_full, z_full],
                positive_mask_full,
                spec,
                row_mask=(slice_index_full == sv_idx),
            )
            grids[key] = build_grid(3, slice_mask=mask, stats_key=key)
            slice_counts.append(int(np.sum(slice_index_full == sv_idx)))
        cell_stats["4_all"] = cell_stats["3"]
        grids["4_all"] = build_grid(3, stats_key="4_all")
        # The genuine 4-D partition: (x, y, z, w) jointly. This is what the
        # branch's own centres are counted over when d = 4 -- the per-slice
        # grids ARE this partition, split for display, and `4_all` is its
        # 3-D marginal. Reporting the marginal as "the branch's coverage"
        # would describe a coarser partition than the one on screen: on
        # `relationship = Husband` at tau = 1.0 the 3-D marginal reads
        # K = 25, coverage 1.3 %, while the 4-D partition the slices actually
        # show reads K = 75, coverage 19.2 %.
        cell_stats["4"] = _FullCellStatistics(
            [x_full, y_full, z_full, w_full], positive_mask_full, spec
        )
        slice_axis_info = {
            "name": humanize_col(w_name, translations),
            "ticks": w_ticks,
            "counts": slice_counts,
        }

    # Which partition the panel's headline describes: the branch's own, i.e.
    # the joint partition over all `branch.d` displayed axes. `_headline_grid_key`
    # is the grid whose cell list backs the "top centres" listing; for a 4-D
    # branch that listing is taken from the 3-D marginal because the per-slice
    # grids are separate objects, while the COUNTS come from the joint
    # partition -- the listing is illustrative, the counts are the report.
    _headline_key = "4" if ("4" in cell_stats and branch.d >= 4) else str(min(branch.d, 3))
    _headline_grid_key = str(min(branch.d, 3))

    return {
        "x": x_num.tolist(),
        "y": y_num.tolist(),
        "z": z_num.tolist(),
        "x_jitter": x_cube.tolist(),
        "y_jitter": y_cube.tolist(),
        "z_jitter": z_cube.tolist(),
        "grids": grids,
        "global_max_n": global_max_n,
        "grid_sizes": grids["3"]["sizes"],
        "grid_x": grids["3"]["x"],
        "grid_y": grids["3"]["y"],
        "grid_z": grids["3"]["z"],
        "grid_opacity": grids["3"]["opacity"],
        "grid_purity": grids["3"]["purity"],
        "grid_purity_lower": grids["3"]["purity_lower"],
        "grid_certified": grids["3"]["certified"],
        "grid_hover_text": grids["3"]["hover_text"],
        "grid_customdata": grids["3"].get("customdata", []),
        "color": color_num.tolist(),
        "target_labels": z_target_human,
        "unique_target_classes": unique_target_labels,
        "raw_target_classes": [str(t) for t in unique_targets],
        "target_name": target_display_name,
        "hover_text": hover_texts,
        "axis_names": {
            "x": humanize_col(x_name, translations),
            "y": humanize_col(y_name, translations),
            "z": humanize_col(z_name, translations),
        },
        "axis_ticks": {
            "x": {"vals": list(range(len(x_ticks))), "text": x_ticks},
            "y": {"vals": list(range(len(y_ticks))), "text": y_ticks},
            "z": {"vals": list(range(len(z_ticks))), "text": z_ticks},
        },
        "slice_axis": slice_axis_info,
        # v2.2: `total_samples` is the number of rows every reported STATISTIC
        # is computed on. `rendered_samples` is how many of them are drawn as
        # individual spheres. Before v2.2 this key held the latter while the
        # panel beside it displayed metrics computed on the former, and the
        # two were 10 000 and 32 561 on the reference dataset.
        "total_samples": int(n_samples),
        "rendered_samples": int(len(indices)),
        "all_feature_names": [humanize_col(fn, translations) for fn in feature_names],
        "raw_feature_names": feature_names,
        "selected_features": [humanize_col(sfn, translations) for sfn in branch.selected_feature_names],
        # Column indices of the branch's axes into the feature matrix (the
        # dataset minus the target, in order): what `/api/analyze`'s
        # `features` takes, so a schema can be re-opened or matched in the
        # landscape's cell listings.
        "selected_feature_indices": [int(j) for j in branch.selected_features],
        # `d` is this branch's dimensionality, not a globally "optimal" d*
        # chosen by the algorithm — the caller (or user) picked which branch
        # to render. `search_centers`/`centers` below carry the statistics
        # the branch is ranked and displayed by.
        #
        # The frontend's WITHIN-branch dimensionality collapse (Section 5.6,
        # case 2) must NOT read this dict for a view dimensionality other
        # than `branch.d` itself; see `view_metrics` below for that.
        "metrics": {
            "d": branch.d,
        },
        # v2.2 headline. `certificate` describes the rule; `centers` is what
        # it produced on the DISPLAYED partition over all rows;
        # `search_centers` is `vsf.avr`'s own value on the SEARCH partition,
        # carried so a discrepancy is visible rather than silently averaged
        # away (see this function's docstring).
        "certificate": {
            "tau": float(spec.tau),
            "alpha": float(spec.alpha),
            "rule": spec.rule,
            "min_samples": int(spec.min_samples),
            "method": spec.method,
            "multiplicity": spec.multiplicity,
            "alpha_effective": float(cell_stats[_headline_key].alpha_effective),
            "positive_value": None if positive_value is None else str(positive_value),
            "positive_label": (
                _label_of(positive_value) if positive_value is not None else None
            ),
            "partition_matches_search": (
                branch.centers is not None
                and branch.centers.n_cells_occupied
                == cell_stats[_headline_key].summary()["n_cells_occupied"]
            ),
        },
        "centers": {
            **cell_stats[_headline_key].summary(),
            "n_positive": int(cell_stats[_headline_key].n_positive),
            "prevalence": float(
                cell_stats[_headline_key].n_positive
                / max(1, cell_stats[_headline_key].n_samples)
            ),
            "top": [
                {
                    "n": int(c["N"]), "k": int(c["k"]), "purity": float(c["pur"]),
                    "lower": float(c["lower"]), "coords": c["coords"],
                }
                for c in sorted(
                    (
                        cd for cd in grids[_headline_grid_key]["customdata"]
                        if cd["certified"]
                    ),
                    key=lambda cd: (-cd["k"], -cd["lower"]),
                )[:10]
            ],
        },
        "search_centers": (
            None
            if branch.centers is None
            else {
                "n_centers": int(branch.centers.n_centers),
                "coverage": float(branch.centers.coverage),
                "coverage_lower": float(branch.centers.coverage_lower),
                "purity_pooled": float(branch.centers.purity_pooled),
                "mass": float(branch.centers.mass),
                "lift": float(branch.centers.lift),
                "max_purity_point": float(branch.centers.max_purity_point),
                "max_purity_lower": float(branch.centers.max_purity_lower),
                "n_positive": int(branch.centers.n_positive),
                "coverage_cv": (
                    None
                    if branch.centers.coverage_cv is None
                    else {
                        "mean": float(branch.centers.coverage_cv.mean),
                        "se": float(branch.centers.coverage_cv.se),
                        "n_splits": int(branch.centers.coverage_cv.n_splits),
                        "n_repeats": int(branch.centers.coverage_cv.n_repeats),
                    }
                ),
                "coverage_p_value": branch.centers.coverage_p_value,
                "coverage_p_value_familywise": branch.coverage_p_value_familywise,
                "undetermined_reason": branch.centers.undetermined_reason,
            }
        ),
        # Per-collapsed-view centre statistics for THIS branch's own axes,
        # keyed by view dimensionality "1".."4" as strings (JSON object keys
        # are always strings). The frontend's dimensionality toggle
        # (`setDimensionality`/`updateHUDForDimension`) must read THESE
        # values for the currently viewed d, not a fixed branch-level
        # scalar — using a full-branch number for a collapsed (d < branch.d)
        # view would silently overstate what the visible axes alone carry.
        # v2.3 dropped mi_by_d/mi_adj_by_d/u_adj_by_d along with
        # `BranchResult.mi_adj_by_prefix_d`/`u_adj_by_prefix_d` (see
        # `vsf.avr`'s module docstring) — coverage_by_d below is the
        # equivalent for the statistic the branch is actually ranked by.
        "view_metrics": {
            # Centre statistics for each collapsed view, computed on the SAME
            # displayed partition the user is looking at (so a 1-D view's
            # coverage is the coverage of the 1-D lattice on screen, not a
            # projection bookkept elsewhere). Defined for every view the
            # renderer can show, including views wider than `branch.d`, where
            # the extra axes are `_axis_fallback_indices` padding.
            # Keyed "1".."4". The "4" entry comes from the joint (x, y, z, w)
            # partition rather than from any single grid, because a 4-D view
            # is drawn as a stack of 3-D slices and its cells are exactly the
            # cells of that joint partition.
            "coverage_by_d": {
                k: float(v.summary()["coverage"])
                for k, v in cell_stats.items() if k in ("1", "2", "3", "4")
            },
            "n_centers_by_d": {
                k: int(v.summary()["n_centers"])
                for k, v in cell_stats.items() if k in ("1", "2", "3", "4")
            },
            "purity_by_d": {
                k: float(v.summary()["purity_pooled"])
                for k, v in cell_stats.items() if k in ("1", "2", "3", "4")
            },
            "max_purity_lower_by_d": {
                k: float(v.summary()["max_purity_lower"])
                for k, v in cell_stats.items() if k in ("1", "2", "3", "4")
            },
            # Share of ALL rows inside the view's certified cells. The
            # headline of an absence search ("how much of the data is
            # certified free of the value"); secondary under presence.
            "mass_by_d": {
                k: float(v.summary()["mass"])
                for k, v in cell_stats.items() if k in ("1", "2", "3", "4")
            },
        },
    }

