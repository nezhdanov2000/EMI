import numpy as np
import pandas as pd
from typing import Dict, List, Any
from .math import shannon_entropy, joint_entropy

def nmi_miller_madow_corrected(Z: np.ndarray, phi: np.ndarray) -> float:
    """
    Calculates Local NMI with Miller-Madow correction for finite sample bias.
    Z and phi should be 1D arrays of the same length (N_c).
    """
    N_c = len(Z)
    if N_c <= 1:
        return 0.0
        
    h_z = shannon_entropy(Z)
    h_phi = shannon_entropy(phi)
    h_z_phi = joint_entropy(Z, phi)
    
    i_plugin = max(0.0, float(h_z + h_phi - h_z_phi))
    
    k_z = len(np.unique(Z))
    k_phi = len(np.unique(phi))
    
    # Miller-Madow correction
    correction = ((k_z - 1) * (k_phi - 1)) / (2 * N_c * np.log(2))
    i_corrected = max(0.0, i_plugin - correction)
    
    denom = min(h_z, h_phi)
    if denom <= 1e-12:
        return 0.0
        
    return min(1.0, max(0.0, float(i_corrected / denom)))

def get_reliability_indicator(n: int) -> str:
    if n > 100:
        return "🟢"
    elif n >= 30:
        return "🟡"
    else:
        return "🔴"

def _evaluate_filter(
    mask_phi: np.ndarray, 
    Z_c: np.ndarray, 
    N_c: int, 
    I_Z_X_F: float, 
    p_c: float
) -> Dict[str, Any]:
    N_phi = int(np.sum(mask_phi))
    
    # Purity positive & negative
    if N_phi > 0:
        purity_pos = float(np.mean(Z_c[mask_phi]))
    else:
        purity_pos = 0.0
        
    N_neg = N_c - N_phi
    if N_neg > 0:
        purity_neg = float(np.mean(Z_c[~mask_phi]))
    else:
        purity_neg = 0.0

    # NMI local
    nmi_local = nmi_miller_madow_corrected(Z_c, mask_phi)
    
    # Delta VIR
    if I_Z_X_F > 1e-12:
        # I(Z; phi | c) is basically the mutual information between Z_c and mask_phi
        h_z_c = shannon_entropy(Z_c)
        h_phi = shannon_entropy(mask_phi)
        h_z_phi = joint_entropy(Z_c, mask_phi)
        i_z_phi_c = max(0.0, float(h_z_c + h_phi - h_z_phi))
        
        delta_vir = (p_c * i_z_phi_c) / I_Z_X_F
    else:
        delta_vir = 0.0
        
    return {
        "n_pos": N_phi,
        "n_neg": N_neg,
        "purity_pos": purity_pos,
        "purity_neg": purity_neg,
        "nmi_local": float(nmi_local),
        "delta_vir": float(delta_vir),
        "reliability": get_reliability_indicator(N_phi)
    }

def mine_dirty_center(
    X_df: pd.DataFrame, 
    Z_target: np.ndarray, 
    center_mask: np.ndarray, 
    I_Z_X_F: float, 
    max_depth: int = 3,
    top_t1: int = 10
) -> List[Dict[str, Any]]:
    """
    Mines conjunctive filters for a dirty center using greedy expansion.
    """
    N_total = len(Z_target)
    N_c = int(np.sum(center_mask))
    if N_c == 0:
        return []
        
    p_c = N_c / N_total
    min_support = max(30, int(0.1 * N_c))
    
    X_c = X_df[center_mask]
    Z_c = Z_target[center_mask]
    
    features = list(X_c.columns)
    
    # Phase 1: Single features (k=1)
    candidates_k1 = []
    
    for f in features:
        unique_vals = X_c[f].unique()
        for v in unique_vals:
            mask_phi = (X_c[f] == v).values
            N_phi = int(np.sum(mask_phi))
            
            if N_phi < min_support:
                continue
                
            eval_res = _evaluate_filter(mask_phi, Z_c, N_c, I_Z_X_F, p_c)
            candidates_k1.append({
                "conditions": [{"col": f, "val": v}],
                "mask": mask_phi,
                **eval_res
            })
            
    # Sort k=1 by NMI local descending and take top_t1
    candidates_k1.sort(key=lambda x: (x["nmi_local"], x["delta_vir"]), reverse=True)
    top_k1 = candidates_k1[:top_t1]
    
    all_candidates = list(top_k1)
    
    # Phase 2: Greedy expansion (k=2..max_depth)
    current_level = top_k1
    
    for depth in range(2, max_depth + 1):
        next_level = []
        for base_cand in current_level:
            used_features = {cond["col"] for cond in base_cand["conditions"]}
            base_mask = base_cand["mask"]
            
            for f in features:
                if f in used_features:
                    continue
                    
                unique_vals = X_c[base_mask][f].unique()
                for v in unique_vals:
                    new_mask = base_mask & (X_c[f] == v).values
                    N_phi = int(np.sum(new_mask))
                    
                    if N_phi < min_support:
                        continue
                        
                    eval_res = _evaluate_filter(new_mask, Z_c, N_c, I_Z_X_F, p_c)
                    
                    if eval_res["nmi_local"] > base_cand["nmi_local"]:
                        new_cand = {
                            "conditions": base_cand["conditions"] + [{"col": f, "val": v}],
                            "mask": new_mask,
                            **eval_res
                        }
                        next_level.append(new_cand)
                        all_candidates.append(new_cand)
                        
        # Keep the top T1 to expand further
        next_level.sort(key=lambda x: (x["nmi_local"], x["delta_vir"]), reverse=True)
        current_level = next_level[:top_t1]
        
    # Phase 3: Ranking and Validation
    # Deduplicate candidates (by condition set)
    unique_candidates = {}
    for cand in all_candidates:
        # Create a canonical key for the conditions
        key = tuple(sorted([(c["col"], str(c["val"])) for c in cand["conditions"]]))
        if key not in unique_candidates or cand["nmi_local"] > unique_candidates[key]["nmi_local"]:
            unique_candidates[key] = cand
            
    final_list = list(unique_candidates.values())
    final_list.sort(key=lambda x: (x["nmi_local"], x["delta_vir"]), reverse=True)
    
    # Remove masks before returning
    for item in final_list:
        del item["mask"]
        
    return final_list[:3]

