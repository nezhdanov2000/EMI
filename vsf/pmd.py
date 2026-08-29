"""
VSF PMD Module: Perceptually-Matched Discretization
Implements rate-distortion optimal quantization, visual channel limits,
and grid capacity protection against Miller-Madow bias.
"""

import warnings

import numpy as np

from .math import normalized_mutual_information

# 7 Visual Channels Limits (Lv) from VSF Spec Table Section 2.2
CHANNEL_LIMITS: dict[str, int] = {
    "position_x": 500,
    "position_y": 500,
    "position_z": 20,
    "color_hue": 12,
    "color_saturation": 7,
    "color_lightness": 9,
    "motion_time": 200,
    "default": 200,
}


def freedman_diaconis_bins(X: np.ndarray, max_bins: int = 200) -> int:
    """Computes recommended number of bins using Freedman-Diaconis rule."""
    arr = np.asarray(X, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 2
        
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = q75 - q25
    
    if iqr == 0:
        # Fallback to Sturges rule if IQR is 0
        n_bins = int(np.ceil(np.log2(len(arr)) + 1))
    else:
        bin_width = 2.0 * iqr / (len(arr) ** (1 / 3))
        if bin_width <= 0:
            n_bins = 10
        else:
            data_range = np.ptp(arr)
            n_bins = int(np.ceil(data_range / bin_width))
            
    return max(2, min(max_bins, n_bins))


def discretize_feature(
    X: np.ndarray,
    n_bins: int | None = None,
    channel_name: str = "default",
    strategy: str = "quantile",
    user_bins: list[float] | None = None,
) -> tuple[np.ndarray, int, float]:
    """
    Discretizes a 1D continuous feature X into k discrete bins.
    
    Returns:
        (discrete_X, k_actual, distortion D_j)
        where D_j = 1 - NMI(X_discrete; X_original_quantized_fine)
    """
    raw_arr = np.asarray(X)
    if raw_arr.ndim > 1:
        raw_arr = raw_arr.ravel()
        
    is_str_or_bool = False
    if raw_arr.dtype.kind in ('U', 'S', 'b'):
        is_str_or_bool = True
    elif raw_arr.dtype.kind == 'O':
        elem = raw_arr.ravel()[0] if len(raw_arr) > 0 else None
        if isinstance(elem, (str, bool)):
            is_str_or_bool = True

    if is_str_or_bool:
        _, discrete_x = np.unique(raw_arr, return_inverse=True)
        k_actual = len(np.unique(discrete_x))
        return discrete_x.astype(int), k_actual, 0.0

    arr = np.asarray(raw_arr, dtype=float)
    l_v = CHANNEL_LIMITS.get(channel_name, CHANNEL_LIMITS["default"])

    # Human-in-the-loop custom user bins
    if user_bins is not None:
        bins = np.sort(np.unique(user_bins))
        if len(bins) > l_v + 1:
            warnings.warn(
                f"User defined bins ({len(bins)-1}) exceed channel capacity limit L_v ({l_v})."
            )
        discrete_x = np.digitize(arr, bins) - 1
        k_actual = len(bins) - 1
    elif len(np.unique(arr)) <= l_v:
        # Low-cardinality numeric feature (e.g. a binary 0/1 indicator, or a
        # small integer count taking few distinct values): map directly to
        # its own distinct values rather than approximating via
        # percentile-based quantile bins.
        #
        # This is not just a simplification, it fixes a real correctness
        # bug the quantile path below has for exactly this case: for a
        # feature with few unique values (e.g. 2), `np.percentile` at
        # `k_actual + 1` evenly-spaced quantile points routinely returns
        # only those 2 endpoint values for MOST of the requested quantiles
        # once k_actual+1 > (number of unique values), so
        # `np.unique(bins)` collapses to length 2 — which passes the
        # `len(bins) < 2` guard below unmodified, yet `bins[1:-1]` (the
        # digitize inner-edge array) comes out EMPTY, so every sample digitizes
        # into bin 0 and a fully informative binary feature silently
        # becomes a CONSTANT (k_actual=1, zero downstream signal). Direct
        # unique-value mapping has no such failure mode: it is exact, not
        # an approximation, whenever the raw cardinality already fits
        # within this channel's capacity `l_v`.
        unique_vals = np.unique(arr)
        discrete_x = np.searchsorted(unique_vals, arr)
        k_actual = len(unique_vals)
    else:
        if n_bins is None:
            k_target = freedman_diaconis_bins(arr, max_bins=l_v)
        else:
            k_target = n_bins
            
        k_actual = min(l_v, max(2, k_target))
        
        if strategy == "quantile":
            quantiles = np.linspace(0, 100, k_actual + 1)
            bins = np.percentile(arr, quantiles)
            bins = np.unique(bins)  # remove duplicate quantiles
            if len(bins) < 2:
                bins = np.linspace(arr.min(), arr.max(), k_actual + 1)
            discrete_x = np.digitize(arr, bins[1:-1])
            k_actual = len(bins) - 1
        else:
            # Equal width
            _, bin_edges = np.histogram(arr, bins=k_actual)
            discrete_x = np.digitize(arr, bin_edges[1:-1])
            
    # Calculate Distortion D_j = 1 - NMI(X_discrete; X_fine)
    # Fine reference quantization with 200 bins
    _, fine_edges = np.histogram(arr, bins=min(200, len(np.unique(arr))))
    fine_x = np.digitize(arr, fine_edges[1:-1])
    
    nmi_val = normalized_mutual_information(discrete_x, fine_x)
    distortion = max(0.0, 1.0 - nmi_val)
    
    return discrete_x, k_actual, distortion


def check_grid_capacity(bin_counts: list[int], n_samples: int) -> bool:
    """
    Checks if hypervolume grid capacity prod(k_j) <= N / 10 (Section 2.3).
    Returns True if grid capacity limit is satisfied, False if exceeded.
    """
    total_cells = int(np.prod(bin_counts))
    max_allowed = max(1, n_samples // 10)
    return total_cells <= max_allowed


def adaptively_coarsen_bins(
    discrete_features: np.ndarray, n_samples: int, target_max_cells: int | None = None
) -> np.ndarray:
    """
    If hypervolume grid prod(k_j) > N/10, adaptively coarsens discrete bin levels
    for high-dimensional joint entropy calculation to prevent Miller-Madow bias.
    """
    arr = np.asarray(discrete_features)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        
    n_rows, n_cols = arr.shape
    if target_max_cells is None:
        target_max_cells = max(1, n_samples // 10)
        
    # Get current bin counts per column
    k_counts = [len(np.unique(arr[:, j])) for j in range(n_cols)]
    prod_k = np.prod(k_counts)
    
    if prod_k <= target_max_cells:
        return arr
        
    # Determine max bins per dimension k_max = floor( (N/10) ^ (1/d) )
    k_max_per_dim = max(1, int(np.floor(target_max_cells ** (1.0 / n_cols))))
    if (k_max_per_dim ** n_cols) > target_max_cells and k_max_per_dim > 1:
        k_max_per_dim = max(1, k_max_per_dim - 1)
    
    coarsened = np.zeros_like(arr)
    for j in range(n_cols):
        col = arr[:, j]
        unique_vals = np.sort(np.unique(col))
        if len(unique_vals) > k_max_per_dim:
            if k_max_per_dim == 1:
                coarsened[:, j] = 0
            else:
                # Map values into k_max_per_dim equal groups
                groups = np.array_split(unique_vals, k_max_per_dim)
                val_map = {}
                for group_idx, grp in enumerate(groups):
                    for val in grp:
                        val_map[val] = group_idx
                coarsened[:, j] = np.vectorize(val_map.get)(col)
        else:
            coarsened[:, j] = col
            
    return coarsened


def discretize_dataset(
    X_matrix: np.ndarray,
    feature_channels: list[str] | None = None,
    strategy: str = "quantile",
) -> tuple[np.ndarray, list[int], list[float]]:
    """
    Discretizes a full feature matrix X (N x M) into a discrete integer matrix.
    
    Returns:
        (X_discrete, bin_counts, distortions)
    """
    X_arr = np.asarray(X_matrix, dtype=object)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
        
    n_rows, n_cols = X_arr.shape
    if feature_channels is None:
        feature_channels = ["default"] * n_cols
        
    discrete_cols = []
    bin_counts = []
    distortions = []
    
    for j in range(n_cols):
        col = X_arr[:, j]
        ch = feature_channels[j] if j < len(feature_channels) else "default"
        disc_col, k_act, dist = discretize_feature(col, channel_name=ch, strategy=strategy)
        discrete_cols.append(disc_col)
        bin_counts.append(k_act)
        distortions.append(dist)
        
    X_discrete = np.column_stack(discrete_cols).astype(int)
    return X_discrete, bin_counts, distortions
