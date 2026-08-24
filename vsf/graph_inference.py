import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from .graph_miner import compute_significant_edges
from .math import normalized_mutual_information
from .vis import humanize_col, humanize_val, MUSHROOM_TRANSLATIONS


def compute_nmi_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Computes pairwise NMI between all columns in the dataframe.
    Returns a dictionary of dictionaries: nmi[col1][col2] = float.
    """
    cols = df.columns.tolist()
    nmi_matrix = {c: {} for c in cols}
    
    for i, c1 in enumerate(cols):
        val1 = df[c1].values
        nmi_matrix[c1][c1] = 1.0
        for j in range(i + 1, len(cols)):
            c2 = cols[j]
            val2 = df[c2].values
            nmi_val = normalized_mutual_information(val1, val2)
            nmi_matrix[c1][c2] = nmi_val
            nmi_matrix[c2][c1] = nmi_val
            
    return nmi_matrix


def run_graph_inference(
    df: pd.DataFrame,
    inputs: Dict[str, str],
    target: str = "class",
    target_criterion: Optional[str] = None,
    nmi_threshold: float = 0.10,
    fdr_q: float = 0.05,
    n_permutations: int = 200,
    random_state: int | None = 42,
) -> Dict[str, Any]:
    """
    Runs graph inference reasoning:
    1. Computes full NMI matrix.
    2. Discovers multi-step reasoning chains from inputs -> Z_1 -> Z_2 -> ... -> target (no direct shortcut).
    3. Allocates nodes into clean non-overlapping layers.
    4. Computes prior & posterior belief distributions for each node given inputs.
    5. Formulates step-by-step reasoning narrative.

    Every edge considered for a reasoning path must clear BOTH an effect-size
    floor (`nmi_threshold`, unchanged) and a Benjamini-Hochberg-corrected
    permutation-test significance bar (new — see
    `vsf.graph_miner.compute_significant_edges`); a path's overall score is
    the MINIMUM (bottleneck) NMI among its edges, not their product. See
    `vsf.graph_miner.mine_strong_links` for why: the product of differently-
    normalized NMI values is not a validated information-theoretic quantity
    despite an earlier version of this function labeling it a "DPI compliant
    multiplicative cascade", and thresholding purely on raw NMI magnitude
    (with no significance test at all) does not control the false discovery
    rate across the O(M^3) candidate paths this search enumerates.
    """
    cols = df.columns.tolist()
    nmi_matrix = compute_nmi_matrix(df)
    _, significant_edges = compute_significant_edges(
        df, fdr_q=fdr_q, n_permutations=n_permutations, random_state=random_state
    )

    def edge_ok(a: str, b: str) -> bool:
        return significant_edges.get(frozenset((a, b)), False)

    # 1. Base Filter Mask for Inputs
    base_mask = np.ones(len(df), dtype=bool)
    for col, val in inputs.items():
        if col in df.columns:
            base_mask &= (df[col].astype(str) == str(val))
            
    n_match = int(base_mask.sum())
    total_rows = len(df)
    
    input_cols = list(inputs.keys())
    
    # 2. Find Best Paths from any Input to Target (Path Search without direct input->target edge)
    # Graph edges with NMI >= nmi_threshold
    # Direct edge input->target is explicitly forbidden
    active_edges = []
    
    # We will search for all paths of length 2 and 3:
    # Length 2: Input -> Z1 -> Target
    # Length 3: Input -> Z1 -> Z2 -> Target
    valid_paths = []
    
    for inp in input_cols:
        for z1 in cols:
            if z1 in input_cols or z1 == target:
                continue
            nmi_inp_z1 = nmi_matrix[inp].get(z1, 0.0)
            if nmi_inp_z1 < nmi_threshold or not edge_ok(inp, z1):
                continue

            # Check 2-step: Input -> Z1 -> Target
            nmi_z1_tgt = nmi_matrix[z1].get(target, 0.0)
            if nmi_z1_tgt >= nmi_threshold and edge_ok(z1, target):
                # Bottleneck score: the weakest edge caps the chain's
                # transmitted information, consistent with how information
                # can only be lost (never manufactured) by an intermediate
                # step — NOT the product of the two edges' NMI values.
                score = float(min(nmi_inp_z1, nmi_z1_tgt))
                valid_paths.append({
                    "path": [inp, z1, target],
                    "score": score,
                    "nmis": [nmi_inp_z1, nmi_z1_tgt]
                })

            # Check 3-step: Input -> Z1 -> Z2 -> Target
            for z2 in cols:
                if z2 in input_cols or z2 == target or z2 == z1:
                    continue
                nmi_z1_z2 = nmi_matrix[z1].get(z2, 0.0)
                if nmi_z1_z2 < nmi_threshold or not edge_ok(z1, z2):
                    continue
                nmi_z2_tgt = nmi_matrix[z2].get(target, 0.0)
                if nmi_z2_tgt >= nmi_threshold and edge_ok(z2, target):
                    score = float(min(nmi_inp_z1, nmi_z1_z2, nmi_z2_tgt))
                    valid_paths.append({
                        "path": [inp, z1, z2, target],
                        "score": score,
                        "nmis": [nmi_inp_z1, nmi_z1_z2, nmi_z2_tgt]
                    })
                    
    # Sort paths by score descending
    valid_paths.sort(key=lambda x: x["score"], reverse=True)
    
    # Select top paths to form the clean reasoning subgraph
    top_paths = valid_paths[:6] if valid_paths else []
    
    # Assign layers based on top paths
    # Layer 0: Inputs
    # Layer 1: First mediators
    # Layer 2: Second mediators (if 3-step)
    # Layer Target: Target
    node_layer_map = {}
    for inp in input_cols:
        node_layer_map[inp] = 0
        
    has_layer_2 = False
    for p_info in top_paths:
        p = p_info["path"]
        if len(p) == 3: # inp -> z1 -> target
            z1 = p[1]
            if z1 not in node_layer_map:
                node_layer_map[z1] = 1
        elif len(p) == 4: # inp -> z1 -> z2 -> target
            z1, z2 = p[1], p[2]
            if z1 not in node_layer_map:
                node_layer_map[z1] = 1
            if z2 not in node_layer_map:
                node_layer_map[z2] = 2
                has_layer_2 = True
                
    target_layer_idx = 3 if has_layer_2 else 2
    node_layer_map[target] = target_layer_idx
    
    # If no multi-step path found, fallback to BFS
    if not top_paths:
        for z1 in cols:
            if z1 in input_cols or z1 == target:
                continue
            for inp in input_cols:
                if nmi_matrix[inp].get(z1, 0.0) >= nmi_threshold:
                    node_layer_map[z1] = 1
                    break
        node_layer_map[target] = 2
        
    # Build edges from the active reasoning nodes
    edge_set = set()
    edges = []
    
    for p_info in top_paths:
        p = p_info["path"]
        for i in range(len(p) - 1):
            u, v = p[i], p[i+1]
            edge_key = (u, v)
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges.append({
                    "source": u,
                    "target": v,
                    "nmi": float(nmi_matrix[u].get(v, 0.0))
                })
                
    if not edges:
        # Fallback edges
        for u, l_u in node_layer_map.items():
            for v, l_v in node_layer_map.items():
                if l_v == l_u + 1 and not (l_u == 0 and v == target):
                    n_val = nmi_matrix[u].get(v, 0.0)
                    if n_val >= nmi_threshold:
                        edges.append({
                            "source": u,
                            "target": v,
                            "nmi": float(n_val)
                        })
                        
    # 3. Calculate Node Belief Distributions and Top Predictions
    nodes = []
    for col in cols:
        is_input = col in input_cols
        is_target = col == target
        layer = node_layer_map.get(col, -1)
        
        # Prior & Posterior
        value_counts = df[col].value_counts(normalize=True).to_dict()
        posterior_counts = {}
        if n_match > 0:
            filtered_df = df[base_mask]
            posterior_counts = filtered_df[col].value_counts(normalize=True).to_dict()
            
        distribution = []
        unique_vals = df[col].dropna().unique()
        for v in unique_vals:
            v_str = str(v)
            human_val_label = humanize_val(col, v_str)
            prior_p = float(value_counts.get(v, 0.0))
            post_p = float(posterior_counts.get(v, 0.0))
            distribution.append({
                "value": v_str,
                "label": human_val_label,
                "prior": prior_p,
                "posterior": post_p
            })
            
        distribution.sort(key=lambda x: x["posterior"], reverse=True)
        top_prediction = distribution[0] if distribution else {"value": "-", "label": "-", "posterior": 0.0}
        
        col_label = humanize_col(col)
        
        nodes.append({
            "id": col,
            "label": col_label,
            "raw_name": col,
            "is_input": is_input,
            "is_target": is_target,
            "layer": layer,
            "top_prediction": top_prediction,
            "distribution": distribution
        })
        
    # 4. Direct NMI vs Chain NMI
    direct_nmi = 0.0
    for inp in input_cols:
        direct_nmi = max(direct_nmi, nmi_matrix[inp].get(target, 0.0))
        
    chain_score = top_paths[0]["score"] if top_paths else 0.0
    
    # 5. Target Criterion Probability
    target_node = next((n for n in nodes if n["id"] == target), None)
    target_probability = 0.0
    target_criterion_label = ""
    
    if target_node:
        if target_criterion:
            for item in target_node["distribution"]:
                if str(item["value"]) == str(target_criterion):
                    target_probability = item["posterior"]
                    target_criterion_label = item["label"]
                    break
        else:
            if target_node["distribution"]:
                target_probability = target_node["distribution"][0]["posterior"]
                target_criterion_label = target_node["distribution"][0]["label"]
                target_criterion = target_node["distribution"][0]["value"]
                
    # 6. Step-by-step Reasoning Chain Text
    reasoning_steps = []
    if top_paths:
        best_path = top_paths[0]["path"]
        for i, node_id in enumerate(best_path):
            node_obj = next((n for n in nodes if n["id"] == node_id), None)
            if i == 0:
                inp_val = inputs.get(node_id, "")
                val_lbl = humanize_val(node_id, inp_val)
                reasoning_steps.append({
                    "step": 1,
                    "type": "input",
                    "title": f"Input Observation: {node_obj['label']}",
                    "detail": f"{val_lbl} ({inp_val})",
                    "nmi_next": float(top_paths[0]["nmis"][0]) if top_paths[0]["nmis"] else 0.0
                })
            elif i == len(best_path) - 1:
                reasoning_steps.append({
                    "step": i + 1,
                    "type": "target",
                    "title": f"Target Conclusion: {node_obj['label']}",
                    "detail": f"{target_criterion_label} ({target_criterion})",
                    "confidence": f"{target_probability * 100:.1f}%"
                })
            else:
                pred = node_obj["top_prediction"]
                nmi_next = float(top_paths[0]["nmis"][i]) if i < len(top_paths[0]["nmis"]) else 0.0
                reasoning_steps.append({
                    "step": i + 1,
                    "type": "mediator",
                    "layer": node_obj["layer"],
                    "title": f"Layer {node_obj['layer']} (Mediator): {node_obj['label']}",
                    "detail": f"{pred['label']} ({pred['value']}) — probability {pred['posterior']*100:.1f}%",
                    "nmi_next": nmi_next
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "direct_nmi": float(direct_nmi),
        "chain_score": float(chain_score),
        "n_match": n_match,
        "total_rows": total_rows,
        "target_criterion": target_criterion,
        "target_criterion_label": target_criterion_label,
        "target_probability": float(target_probability),
        "reasoning_steps": reasoning_steps,
        "best_paths": top_paths
    }
