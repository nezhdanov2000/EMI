"""
VSF Core Math Module: Information-Theoretic Calculations
Implements Shannon Entropy, Joint Entropy, Discrete Mutual Information, and NMI.
"""

import numpy as np
from typing import Union, List, Tuple


def _encode_to_int(arr: np.ndarray) -> np.ndarray:
    """Safely converts string or object arrays to integer indices for np.unique."""
    arr = np.asarray(arr)
    if arr.dtype.kind in ('U', 'S', 'O', 'b'):
        if arr.ndim > 1:
            cols = []
            for j in range(arr.shape[1]):
                _, col_idx = np.unique(arr[:, j], return_inverse=True)
                cols.append(col_idx)
            return np.column_stack(cols)
        else:
            _, idx = np.unique(arr, return_inverse=True)
            return idx
    return arr


def shannon_entropy(X: Union[np.ndarray, List]) -> float:
    """
    Computes Shannon Entropy H(X) in bits.
    
    H(X) = - sum_{x in X} p(x) * log2(p(x))
    """
    arr = _encode_to_int(X)
    if arr.size == 0:
        return 0.0
    
    # If 2D array with multiple columns, flatten unique tuples
    if arr.ndim > 1:
        _, counts = np.unique(arr, axis=0, return_counts=True)
    else:
        _, counts = np.unique(arr, return_counts=True)
        
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def joint_entropy(X: Union[np.ndarray, List], Y: Union[np.ndarray, List]) -> float:
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


def mutual_information(Z: Union[np.ndarray, List], X_S: Union[np.ndarray, List]) -> float:
    """
    Computes Discrete Mutual Information I(Z; X_S) in bits.
    
    I(Z; X_S) = H(Z) + H(X_S) - H(Z, X_S)
    """
    h_z = shannon_entropy(Z)
    h_xs = shannon_entropy(X_S)
    h_z_xs = joint_entropy(Z, X_S)
    
    mi = h_z + h_xs - h_z_xs
    return max(0.0, float(mi))  # Floating point accuracy guard


def normalized_mutual_information(Z: Union[np.ndarray, List], X_S: Union[np.ndarray, List]) -> float:
    """
    Computes Normalized Mutual Information NMI(Z; X_S).
    
    NMI(Z; X_S) = I(Z; X_S) / min(H(Z), H(X_S))
    
    Uses min(H(Z), H(X_S)) normalization as defined in VSF spec Section 3.2.
    """
    mi = mutual_information(Z, X_S)
    if mi <= 0.0:
        return 0.0
        
    h_z = shannon_entropy(Z)
    h_xs = shannon_entropy(X_S)
    
    denom = min(h_z, h_xs)
    if denom <= 1e-12:
        return 0.0
        
    nmi = mi / denom
    return min(1.0, max(0.0, float(nmi)))
