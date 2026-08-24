"""
VSF Core Math Module: Information-Theoretic Calculations
Implements Shannon Entropy, Joint Entropy, Discrete Mutual Information, and NMI.
"""


import numpy as np

from .backend import as_backend, as_numpy, get_backend


def _encode_to_int(arr) -> np.ndarray:
    """Safely converts string or object arrays to integer indices for unique counting."""
    arr_np = as_numpy(arr)
    if arr_np.dtype.kind in ('U', 'S', 'O', 'b'):
        if arr_np.ndim > 1:
            cols = []
            for j in range(arr_np.shape[1]):
                _, col_idx = np.unique(arr_np[:, j], return_inverse=True)
                cols.append(col_idx)
            return np.column_stack(cols)
        else:
            _, idx = np.unique(arr_np, return_inverse=True)
            return idx
    return arr


def _flatten_2d_to_1d(arr, xp):
    """Flattens 2D discrete columns to 1D integers for high-performance 1D unique."""
    if arr.ndim <= 1:
        return arr.ravel()
    n_cols = arr.shape[1]
    if n_cols == 1:
        return arr[:, 0]
    
    mins = arr.min(axis=0)
    shifted = arr - mins
    maxs = shifted.max(axis=0) + 1
    maxs_np = as_numpy(maxs)
    
    prod = 1
    for m in maxs_np:
        prod *= int(m)
        if prod >= (1 << 62):
            break
            
    if prod < (1 << 62):
        flat = shifted[:, 0].astype(xp.int64)
        for j in range(1, n_cols):
            flat = flat * int(maxs_np[j]) + shifted[:, j]
        return flat
    return arr


def shannon_entropy(X: np.ndarray | list) -> float:
    """
    Computes Shannon Entropy H(X) in bits.
    
    H(X) = - sum_{x in X} p(x) * log2(p(x))
    """
    # Fallback encoding is always CPU bound for strings/objects
    arr = _encode_to_int(X)
    
    if arr.size == 0:
        return 0.0
        
    xp = get_backend()
    arr = as_backend(arr)
    
    if arr.ndim > 1:
        flat_arr = _flatten_2d_to_1d(arr, xp)
        if flat_arr.ndim == 1:
            _, counts = xp.unique(flat_arr, return_counts=True)
        else:
            _, counts = xp.unique(arr, axis=0, return_counts=True)
    else:
        _, counts = xp.unique(arr, return_counts=True)
        
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-xp.sum(probs * xp.log2(probs)))


def joint_entropy(X: np.ndarray | list, Y: np.ndarray | list) -> float:
    """
    Computes Joint Entropy H(X, Y) in bits.
    
    H(X, Y) = - sum_{x, y} p(x, y) * log2(p(x, y))
    """
    xp = get_backend()
    arr_x = as_backend(X)
    arr_y = as_backend(Y)
    
    if arr_x.shape[0] != arr_y.shape[0]:
        raise ValueError(f"Sample count mismatch: {arr_x.shape[0]} vs {arr_y.shape[0]}")
    
    if arr_x.ndim == 1:
        arr_x = arr_x.reshape(-1, 1)
    if arr_y.ndim == 1:
        arr_y = arr_y.reshape(-1, 1)
        
    joint_arr = xp.hstack([arr_x, arr_y])
    return shannon_entropy(joint_arr)


def _mi_with_entropies(
    Z: np.ndarray | list, X_S: np.ndarray | list
) -> tuple[float, float, float]:
    """
    Internal helper: computes H(Z), H(X_S) and I(Z; X_S) with each entropy
    evaluated exactly once. mutual_information and normalized_mutual_information
    both need H(Z) and H(X_S) individually as well as the combined MI value;
    routing them through this single helper avoids recomputing the same
    entropy (each an O(N log N) unique/sort pass) twice per call.

    Returns:
        (mi, h_z, h_xs)
    """
    h_z = shannon_entropy(Z)
    h_xs = shannon_entropy(X_S)
    h_z_xs = joint_entropy(Z, X_S)

    mi = max(0.0, float(h_z + h_xs - h_z_xs))  # Floating point accuracy guard
    return mi, h_z, h_xs


def mutual_information(Z: np.ndarray | list, X_S: np.ndarray | list) -> float:
    """
    Computes Discrete Mutual Information I(Z; X_S) in bits.

    I(Z; X_S) = H(Z) + H(X_S) - H(Z, X_S)
    """
    mi, _, _ = _mi_with_entropies(Z, X_S)
    return mi


def normalized_mutual_information(Z: np.ndarray | list, X_S: np.ndarray | list) -> float:
    """
    Computes Normalized Mutual Information NMI(Z; X_S).

    NMI(Z; X_S) = I(Z; X_S) / min(H(Z), H(X_S))

    Uses min(H(Z), H(X_S)) normalization as defined in VSF spec Section 3.2.
    """
    mi, h_z, h_xs = _mi_with_entropies(Z, X_S)
    if mi <= 0.0:
        return 0.0

    denom = min(h_z, h_xs)
    if denom <= 1e-12:
        return 0.0

    nmi = mi / denom
    return min(1.0, max(0.0, float(nmi)))
