import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .math import joint_entropy, shannon_entropy
from .permutation import marginal_permutation_test
from .stats import benjamini_hochberg


def nmi_miller_madow_corrected(Z: np.ndarray, phi: np.ndarray) -> float:
    """
    Calculates Local NMI with Miller-Madow correction for finite sample bias.
    Z and phi should be 1D arrays of the same length (N_c).

    The Miller (1955) plugin entropy estimator is negatively biased:
    E[H_plugin(X)] ~= H_true(X) - (K_X - 1) / (2N), where K_X is the number
    of observed categories. Correcting each of the three plugin entropy
    estimates in I(Z;phi) = H(Z) + H(phi) - H(Z,phi) by adding back its own
    bias term gives:

        I_MM = I_plugin + [(K_Z - 1) + (K_phi - 1) - (K_joint - 1)] / (2N ln2)

    (in bits; K_joint is the number of distinct OBSERVED (Z, phi) pairs, not
    K_Z * K_phi). Because K_joint typically grows faster than K_Z + K_phi as
    the joint distribution fills in, this correction term is usually
    negative in practice — i.e. it usually *reduces* the plugin MI, which is
    the expected direction for a finite-sample bias correction, since plugin
    MI is on average upward-biased.

    An earlier version of this function used
    `I_plugin - (K_Z-1)(K_phi-1) / (2N ln2)`: a product of the two marginal
    cardinality offsets with no K_joint term at all. That is not the
    Miller-Madow correction for mutual information found in the cited
    literature (Miller, 1955; Panzeri & Treves, 1996) — it happened to have
    the right sign only in the coincidental case tested by this module's
    binary-phi callers, but does not generalize and does not correspond to
    summing the three entropy terms' individual bias corrections.
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
    joint_rows = np.column_stack([np.asarray(Z).reshape(N_c, -1), np.asarray(phi).reshape(N_c, -1)])
    k_joint = len(np.unique(joint_rows, axis=0))

    correction = ((k_z - 1) + (k_phi - 1) - (k_joint - 1)) / (2 * N_c * np.log(2))
    i_corrected = max(0.0, i_plugin + correction)

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


def _min_support_count(N_c: int, theta_supp: float = 0.03, min_absolute: int = 30) -> int:
    """
    Support threshold per Project_Master_Document.md Section 4.5.2: a
    predicate is admitted if Support(phi) >= theta_supp OR N_pos >= 30 (an
    OR of a relative and an absolute floor). The minimal N_phi satisfying
    *either* clause is the smaller of the two count thresholds — requiring
    the larger of the two (as `max(min_absolute, theta_supp * N_c)` would)
    silently implements an AND instead of the spec's OR, and is strictly
    more conservative than intended for small dirty centers.
    """
    return max(1, min(min_absolute, math.ceil(theta_supp * N_c)))


def _evaluate_filter(
    mask_phi: np.ndarray,
    Z_c: np.ndarray,
    N_c: int,
    I_Z_X_F: float,
    p_c: float,
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
        "reliability": get_reliability_indicator(N_phi),
    }


def mine_dirty_center(
    X_df: pd.DataFrame,
    Z_target: np.ndarray,
    center_mask: np.ndarray,
    I_Z_X_F: float,
    max_depth: int = 3,
    top_t1: int = 10,
    n_permutations: int = 200,
    fdr_q: float = 0.05,
    random_state: int | None = 42,
) -> List[Dict[str, Any]]:
    """
    Mines conjunctive filters for a dirty center using greedy expansion.

    Candidate filters are still ranked by Miller-Madow-corrected local NMI
    (an effect-size measure), but the final reported set is now also gated
    on statistical significance: every deduplicated candidate is tested with
    a marginal permutation test (H0: the filter carries no information about
    Z_c beyond chance) and the raw p-values are corrected jointly via
    Benjamini-Hochberg at `fdr_q`. The greedy search over
    features x values x depth explores a large candidate space (easily
    hundreds of conjunctions for a dataset with many categorical columns),
    so reporting candidates by a fixed, uncorrected NMI cutoff — as this
    function previously did (`nmi_local >= 0.05`, no permutation test at
    all) — does not control the false discovery rate across that search;
    see Project_Master_Document.md Section 4.5.3.
    """
    N_total = len(Z_target)
    N_c = int(np.sum(center_mask))
    if N_c == 0:
        return []

    p_c = N_c / N_total
    min_support = _min_support_count(N_c)

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

    # Phase 3: Deduplicate, then test significance jointly (FDR) before ranking
    unique_candidates = {}
    for cand in all_candidates:
        # Create a canonical key for the conditions
        key = tuple(sorted([(c["col"], str(c["val"])) for c in cand["conditions"]]))
        if key not in unique_candidates or cand["nmi_local"] > unique_candidates[key]["nmi_local"]:
            unique_candidates[key] = cand

    deduped = list(unique_candidates.values())

    if deduped:
        p_values = np.empty(len(deduped), dtype=float)
        for i, cand in enumerate(deduped):
            # H0: this conjunctive filter carries no information about Z_c
            # beyond what a random subset of the same size would.
            _, p_val, _ = marginal_permutation_test(
                Z_c,
                cand["mask"].astype(np.int64),
                n_permutations=n_permutations,
                alpha=fdr_q,
                random_state=random_state,
            )
            p_values[i] = p_val
            cand["p_value"] = float(p_val)

        significant_mask = benjamini_hochberg(p_values, q=fdr_q)
        final_list = [cand for cand, sig in zip(deduped, significant_mask) if sig]
    else:
        final_list = []

    final_list.sort(key=lambda x: (x["nmi_local"], x["delta_vir"]), reverse=True)

    from .vis import humanize_val, MUSHROOM_TRANSLATIONS

    # Format human readable descriptions
    for item in final_list:
        del item["mask"]
        human_parts = []
        for cond in item["conditions"]:
            col = cond["col"]
            val = str(cond["val"])
            en_col = MUSHROOM_TRANSLATIONS.get("columns", {}).get(col, col)
            en_val = humanize_val(col, val)
            cond["human_col"] = en_col
            cond["human_val"] = en_val
            human_parts.append(f"{en_col} = {en_val}")
        item["human_text"] = " ∧ ".join(human_parts)

    return final_list[:5]
