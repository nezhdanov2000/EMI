"""
VSF Core Math Module: Information-Theoretic Calculations
Implements Shannon Entropy, Joint Entropy, Discrete Mutual Information, and NMI.

These are low-level PLUG-IN (maximum-likelihood) estimators. They are correct
implementations of their textbook definitions and nothing more: every one of
them is positively biased on finite samples, by an amount that grows with the
cardinality of the joint support (see `vsf.metrics` for the exact expectation
under the permutation null and for the bias-corrected quantities that VSF
reports and ranks by). Do not surface a number produced by this module to a
user, and do not compare two numbers produced by this module across feature
subsets of different cardinality, without correcting it first.
"""


import numpy as np

from .metrics import (
    cell_codes,
    contingency_from_codes,
    entropy_bits_from_counts,
    mutual_information_bits,
)

def _encode_to_int(arr) -> np.ndarray:
    """Safely converts string or object arrays to integer indices for unique counting."""
    arr_np = np.asarray(arr)
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


def _flatten_2d_to_1d(arr):
    """Flattens 2D discrete columns to 1D integers for high-performance 1D unique."""
    if arr.ndim <= 1:
        return arr.ravel()
    n_cols = arr.shape[1]
    if n_cols == 1:
        return arr[:, 0]
    
    mins = arr.min(axis=0)
    shifted = arr - mins
    maxs = shifted.max(axis=0) + 1
    maxs_np = np.asarray(maxs)
    
    prod = 1
    for m in maxs_np:
        prod *= int(m)
        if prod >= (1 << 62):
            break
            
    if prod < (1 << 62):
        flat = shifted[:, 0].astype(np.int64)
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
        
    arr = np.asarray(arr)
    
    if arr.ndim > 1:
        flat_arr = _flatten_2d_to_1d(arr)
        if flat_arr.ndim == 1:
            _, counts = np.unique(flat_arr, return_counts=True)
        else:
            _, counts = np.unique(arr, axis=0, return_counts=True)
    else:
        _, counts = np.unique(arr, return_counts=True)
        
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def joint_entropy(X: np.ndarray | list, Y: np.ndarray | list) -> float:
    """
    Computes Joint Entropy H(X, Y) in bits.
    
    H(X, Y) = - sum_{x, y} p(x, y) * log2(p(x, y))
    """
    arr_x = np.asarray(X)
    arr_y = np.asarray(Y)
    
    if arr_x.shape[0] != arr_y.shape[0]:
        raise ValueError(f"Sample count mismatch: {arr_x.shape[0]} vs {arr_y.shape[0]}")
    
    if arr_x.ndim == 1:
        arr_x = arr_x.reshape(-1, 1)
    if arr_y.ndim == 1:
        arr_y = arr_y.reshape(-1, 1)
        
    joint_arr = np.hstack([arr_x, arr_y])
    return shannon_entropy(joint_arr)


def _mi_with_entropies(
    Z: np.ndarray | list, X_S: np.ndarray | list
) -> tuple[float, float, float]:
    """
    Internal helper: computes H(Z), H(X_S) and I(Z; X_S) from ONE joint
    contingency table, so both marginals and the MI come out of a single
    O(N log N) coding pass.

    I is evaluated directly as sum_ij (n_ij/N) log2(N n_ij / (a_i b_j)) rather
    than as H(Z) + H(X_S) - H(Z, X_S). The subtractive form loses precision
    exactly where VSF needs it most: for a micro-class target the three
    entropies are O(1)-O(10) bits while their combination is O(1e-3), so the
    difference discards roughly four significant digits and can go negative,
    which is why the old implementation needed a `max(0.0, ...)` clamp. The
    table form is a sum of non-negative-weighted terms and needs no clamp.

    Returns:
        (mi, h_z, h_xs)
    """
    z_codes, n_rows = cell_codes(Z)
    x_codes, n_cols = cell_codes(X_S)
    if z_codes.shape[0] != x_codes.shape[0]:
        raise ValueError(f"Sample count mismatch: {z_codes.shape[0]} vs {x_codes.shape[0]}")
    if z_codes.size == 0:
        return 0.0, 0.0, 0.0

    table = contingency_from_codes(z_codes, x_codes, n_rows, n_cols)
    h_z = entropy_bits_from_counts(table.sum(axis=1))
    h_xs = entropy_bits_from_counts(table.sum(axis=0))
    mi = mutual_information_bits(table)
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

    RETAINED AS A PRIMITIVE ONLY - NOT A REPORTABLE METRIC. VSF no longer
    displays or ranks by this quantity, and no caller inside the package uses
    it. It is kept so that the comparison table in Project_Master_Document.md
    Section 3.2 stays reproducible from the library.

    The defect: for a micro-class target the denominator collapses to a
    near-zero H(Z) while the plug-in numerator retains its full positive bias,
    so NMI_min saturates toward 100 % on pure noise. Measured on N = 32 561
    with a 7-member target class and 3 256 occupied cells, an independently
    generated feature scores NMI_min = 66 %. Substituting a geometric mean
    sqrt(H(Z) H(X_S)) does not fix this - it rescales noise (1.0 %) and a
    deterministic relation (1.6 %) by the same factor and destroys the
    contrast between them. The bias lives in the estimator, not the
    denominator; use `vsf.metrics.adjusted_uncertainty_coefficient`.
    """
    mi, h_z, h_xs = _mi_with_entropies(Z, X_S)
    if mi <= 0.0:
        return 0.0

    denom = min(h_z, h_xs)
    if denom <= 1e-12:
        return 0.0

    nmi = mi / denom
    return min(1.0, max(0.0, float(nmi)))
