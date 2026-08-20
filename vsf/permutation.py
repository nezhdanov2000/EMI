"""
VSF Permutation Engine: Statistical Significance Testing
Implements Marginal Permutation Test (Definition 3) and Conditional Stratified
Permutation Test (Definition 4) for Mutual Information feature selection.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from .math import mutual_information, shannon_entropy, joint_entropy
from .backend import get_backend, as_backend, as_numpy


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
    xp = get_backend()
    z_arr = as_backend(Z).ravel()
    x_arr = as_backend(X_j).ravel()
    
    if len(z_arr) != len(x_arr):
        raise ValueError("Z and X_j must have the same length")
        
    if xp.__name__ == 'cupy':
        rng = xp.random.RandomState(random_state)
    else:
        rng = np.random.default_rng(random_state)
        
    h_z = shannon_entropy(z_arr)
    h_x = shannon_entropy(x_arr)
    h_z_x = joint_entropy(z_arr, x_arr)
    i_obs = max(0.0, float(h_z + h_x - h_z_x))
    
    if i_obs <= 1e-12:
        return 0.0, 1.0, False
        
    count_exceed = 0
    z_perm = z_arr.copy()
    
    for _ in range(n_permutations):
        rng.shuffle(z_perm)
        h_z_x_perm = joint_entropy(z_perm, x_arr)
        i_perm = max(0.0, float(h_z + h_x - h_z_x_perm))
        if i_perm >= i_obs:
            count_exceed += 1
            
    p_value = (1 + count_exceed) / (1 + n_permutations)
    is_significant = p_value < alpha
    return float(i_obs), float(p_value), is_significant


def _get_strata_groups(X_S):
    """Precomputes strata indices on CPU to avoid GPU-CPU kernel launch overhead."""
    x_s_arr = np.asarray(as_numpy(X_S))
    if x_s_arr.ndim == 1:
        x_s_arr = x_s_arr.reshape(-1, 1)
        
    _, strata_indices = np.unique(x_s_arr, axis=0, return_inverse=True)
    unique_strata, counts = np.unique(strata_indices, return_counts=True)
    sparse_strata = set(unique_strata[counts < 2])
    
    groups = {}
    fallback = []
    for idx, s in enumerate(strata_indices):
        if s in sparse_strata:
            fallback.append(idx)
        else:
            if s not in groups:
                groups[s] = []
            groups[s].append(idx)
            
    group_list = [np.array(indices, dtype=np.int64) for indices in groups.values() if len(indices) > 1]
    if len(fallback) > 1:
        group_list.append(np.array(fallback, dtype=np.int64))
    return group_list


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
    xp = get_backend()
    z_arr = as_backend(Z).ravel()
    x_j_arr = as_backend(X_j).ravel()
    x_s_arr = as_backend(X_S)
    
    if x_s_arr.ndim == 1:
        x_s_arr = x_s_arr.reshape(-1, 1)
        
    cpu_rng = np.random.default_rng(random_state)
    z_np = np.asarray(as_numpy(z_arr)).copy()
    strata_groups = _get_strata_groups(x_s_arr)
    
    if x_j_arr.ndim == 1:
        x_j_arr_2d = x_j_arr.reshape(-1, 1)
    else:
        x_j_arr_2d = x_j_arr
    x_s_plus_j = xp.hstack([x_s_arr, x_j_arr_2d])
    
    h_z = shannon_entropy(z_arr)
    h_xs = shannon_entropy(x_s_arr)
    h_xs_plus_j = shannon_entropy(x_s_plus_j)
    
    h_z_xs = joint_entropy(z_arr, x_s_arr)
    h_z_xs_plus_j = joint_entropy(z_arr, x_s_plus_j)
    
    i_base = max(0.0, float(h_z + h_xs - h_z_xs))
    i_combined = max(0.0, float(h_z + h_xs_plus_j - h_z_xs_plus_j))
    delta_i_obs = max(0.0, float(i_combined - i_base))
    
    if delta_i_obs <= 1e-12:
        return 0.0, 1.0, False
        
    count_exceed = 0
    z_perm_np = z_np.copy()
    
    for _ in range(n_permutations):
        for grp in strata_groups:
            z_perm_np[grp] = cpu_rng.permutation(z_np[grp])
            
        z_perm = as_backend(z_perm_np)
        h_z_xs_perm = joint_entropy(z_perm, x_s_arr)
        h_z_xs_plus_j_perm = joint_entropy(z_perm, x_s_plus_j)
        
        i_base_perm = max(0.0, float(h_z + h_xs - h_z_xs_perm))
        i_comb_perm = max(0.0, float(h_z + h_xs_plus_j - h_z_xs_plus_j_perm))
        delta_i_perm = max(0.0, float(i_comb_perm - i_base_perm))
        
        if delta_i_perm >= delta_i_obs:
            count_exceed += 1
            
    p_value = (1 + count_exceed) / (1 + n_permutations)
    is_significant = p_value < alpha
    return float(delta_i_obs), float(p_value), is_significant
