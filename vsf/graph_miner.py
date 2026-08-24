import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .math import mutual_information, shannon_entropy
from .vis import humanize_col, humanize_val

def compute_predictiveness_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Computes Asymmetric NMI (Uncertainty Coefficient) U(Y|X) = I(X, Y) / H(Y).
    This strictly obeys the Data Processing Inequality for multiplicative chains.
    """
    cols = df.columns.tolist()
    entropies = {c: shannon_entropy(df[c].values) for c in cols}
    u_matrix = {c: {} for c in cols}
    
    for c1 in cols:
        for c2 in cols:
            if c1 == c2:
                continue
            mi = mutual_information(df[c1].values, df[c2].values)
            if entropies[c2] > 0:
                u_matrix[c1][c2] = float(mi / entropies[c2])
            else:
                u_matrix[c1][c2] = 0.0
    return u_matrix

def mine_strong_links(
    df: pd.DataFrame, 
    target: str = "class", 
    min_nmi: float = 0.20, 
    max_depth: int = 3,
    only_winning: bool = True
) -> Dict[str, Any]:
    """
    Mines all mediator chains where the Indirect Chain Predictiveness is strictly BETTER or EQUAL 
    to the direct predictiveness (Chain Score >= Direct Score, Gain >= 0).
    Uses Asymmetric NMI (Uncertainty Coefficient) to avoid false positives on near-zero entropy nodes.
    """
    cols = df.columns.tolist()
    nmi = compute_predictiveness_matrix(df)
    
    # Target columns to consider: if target is "all", search across entire dataset, else for specific target
    targets_to_search = [target] if target != "all" else cols
    
    winning_chains = []
    
    for tgt in targets_to_search:
        for inp in cols:
            if inp == tgt:
                continue
            direct_nmi = float(nmi[inp].get(tgt, 0.0))
            
            # 1. 2-step: inp -> z1 -> tgt
            for z1 in cols:
                if z1 == inp or z1 == tgt:
                    continue
                nmi_inp_z1 = float(nmi[inp].get(z1, 0.0))
                nmi_z1_tgt = float(nmi[z1].get(tgt, 0.0))
                chain_2 = nmi_inp_z1 * nmi_z1_tgt
                
                # Check condition: chain is at least min_nmi AND (if only_winning, chain >= direct)
                is_win = chain_2 >= direct_nmi
                if chain_2 >= min_nmi and (not only_winning or is_win):
                    gain = chain_2 - direct_nmi
                    ratio = chain_2 / max(0.001, direct_nmi)
                    
                    winning_chains.append({
                        "type": "mediator_1",
                        "type_label": "2 шага (1 медиатор)",
                        "input": inp,
                        "mediators": [z1],
                        "target": tgt,
                        "path": [inp, z1, tgt],
                        "path_labels": [humanize_col(inp), humanize_col(z1), humanize_col(tgt)],
                        "chain_score": float(chain_2),
                        "direct_nmi": float(direct_nmi),
                        "gain": float(gain),
                        "ratio": float(ratio),
                        "is_superior": is_win and gain > 0.001
                    })
                    
            # 2. 3-step: inp -> z1 -> z2 -> tgt
            if max_depth >= 3:
                for z1 in cols:
                    if z1 == inp or z1 == tgt:
                        continue
                    nmi_inp_z1 = float(nmi[inp].get(z1, 0.0))
                    if nmi_inp_z1 < min_nmi:
                        continue
                        
                    for z2 in cols:
                        if z2 == inp or z2 == tgt or z2 == z1:
                            continue
                        nmi_z1_z2 = float(nmi[z1].get(z2, 0.0))
                        nmi_z2_tgt = float(nmi[z2].get(tgt, 0.0))
                        chain_3 = nmi_inp_z1 * nmi_z1_z2 * nmi_z2_tgt
                        
                        is_win = chain_3 >= direct_nmi
                        if chain_3 >= min_nmi and (not only_winning or is_win):
                            gain = chain_3 - direct_nmi
                            ratio = chain_3 / max(0.001, direct_nmi)
                            
                            winning_chains.append({
                                "type": "mediator_2",
                                "type_label": "3 шага (2 медиатора)",
                                "input": inp,
                                "mediators": [z1, z2],
                                "target": tgt,
                                "path": [inp, z1, z2, tgt],
                                "path_labels": [humanize_col(inp), humanize_col(z1), humanize_col(z2), humanize_col(tgt)],
                                "chain_score": float(chain_3),
                                "direct_nmi": float(direct_nmi),
                                "gain": float(gain),
                                "ratio": float(ratio),
                                "is_superior": is_win and gain > 0.001
                            })

    # Sort chains by Gain descending first, then by chain score
    winning_chains.sort(key=lambda x: (x["gain"], x["chain_score"]), reverse=True)
    
    # Deduplicate: keep top 3 best paths per (input, target) pair
    seen_pairs = {}
    deduped = []
    for ch in winning_chains:
        pair_key = (ch["input"], ch["target"])
        if pair_key not in seen_pairs:
            seen_pairs[pair_key] = 0
        if seen_pairs[pair_key] < 2:
            seen_pairs[pair_key] += 1
            deduped.append(ch)
            
    return {
        "target": target,
        "target_label": humanize_col(target) if target != "all" else "Весь датасет (Все пары)",
        "min_nmi": min_nmi,
        "total_found": len(deduped),
        "top_chains": deduped[:50]
    }
