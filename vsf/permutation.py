"""
VSF Permutation Engine: Statistical Significance Testing
Implements Marginal Permutation Test (Definition 3) and Conditional Stratified
Permutation Test (Definition 4) for Mutual Information feature selection.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from .math import mutual_information


def marginal_permutation_test(
    Z: np.ndarray,
    X_j: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 0.01,
    random_state: Optional[int] = None,
) -> Tuple[float, float, bool]:
    """
    Performs Marginal Permutation Test for H0: I(Z; X_j) = 0 (Definition 3).
    
    Permutes target vector Z to construct empirical null distribution.
    
    Returns:
        (i_obs, p_value, is_significant)
    """
    z_arr = np.asarray(Z).ravel()
    x_arr = np.asarray(X_j).ravel()
    
    if len(z_arr) != len(x_arr):
        raise ValueError("Z and X_j must have the same length")
        
    rng = np.random.default_rng(random_state)
    i_obs = mutual_information(z_arr, x_arr)
    
    if i_obs <= 1e-12:
        return 0.0, 1.0, False
        
    count_exceed = 0
    z_perm = z_arr.copy()
    
    for _ in range(n_permutations):
        rng.shuffle(z_perm)
        i_perm = mutual_information(z_perm, x_arr)
        if i_perm >= i_obs:
            count_exceed += 1
            
    p_value = (1 + count_exceed) / (1 + n_permutations)
    is_significant = p_value < alpha
    return float(i_obs), float(p_value), is_significant


def _stratified_permute_z(
    Z: np.ndarray,
    X_S: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permutes vector Z within strata defined by unique combinations of X_S.
    Merges sparse strata (< 2 elements) to allow valid within-strata permutation.
    """
    z_arr = np.asarray(Z).ravel()
    x_s_arr = np.asarray(X_S)
    if x_s_arr.ndim == 1:
        x_s_arr = x_s_arr.reshape(-1, 1)
        
    n_samples = len(z_arr)
    z_perm = z_arr.copy()
    
    # Get unique strata indices
    _, strata_indices = np.unique(x_s_arr, axis=0, return_inverse=True)
    
    unique_strata, counts = np.unique(strata_indices, return_counts=True)
    sparse_strata = set(unique_strata[counts < 2])
    
    if len(sparse_strata) > 0:
        # Merge all sparse strata into a single fallback stratum index -1
        merged_strata = np.array(
            [-1 if s in sparse_strata else s for s in strata_indices]
        )
    else:
        merged_strata = strata_indices
        
    # Permute Z within each stratum group
    for stratum_id in np.unique(merged_strata):
        mask = merged_strata == stratum_id
        idx = np.where(mask)[0]
        if len(idx) > 1:
            z_perm[idx] = rng.permutation(z_arr[idx])
            
    return z_perm


def conditional_permutation_test(
    Z: np.ndarray,
    X_j: np.ndarray,
    X_S: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 0.01,
    random_state: Optional[int] = None,
) -> Tuple[float, float, bool]:
    """
    Performs Conditional Stratified Permutation Test for H0: Delta I(j | S) = 0 (Definition 4).
    
    Tests if adding feature X_j to subset S brings statistically significant new information about Z.
    
    Returns:
        (delta_i_obs, p_value, is_significant)
    """
    z_arr = np.asarray(Z).ravel()
    x_j_arr = np.asarray(X_j).ravel()
    x_s_arr = np.asarray(X_S)
    if x_s_arr.ndim == 1:
        x_s_arr = x_s_arr.reshape(-1, 1)
        
    rng = np.random.default_rng(random_state)
    
    x_s_plus_j = np.column_stack([x_s_arr, x_j_arr])
    
    i_base = mutual_information(z_arr, x_s_arr)
    i_combined = mutual_information(z_arr, x_s_plus_j)
    delta_i_obs = max(0.0, i_combined - i_base)
    
    if delta_i_obs <= 1e-12:
        return 0.0, 1.0, False
        
    count_exceed = 0
    
    for _ in range(n_permutations):
        z_perm = _stratified_permute_z(z_arr, x_s_arr, rng)
        i_base_perm = mutual_information(z_perm, x_s_arr)
        i_comb_perm = mutual_information(z_perm, x_s_plus_j)
        delta_i_perm = max(0.0, i_comb_perm - i_base_perm)
        
        if delta_i_perm >= delta_i_obs:
            count_exceed += 1
            
    p_value = (1 + count_exceed) / (1 + n_permutations)
    is_significant = p_value < alpha
    return float(delta_i_obs), float(p_value), is_significant
