import itertools
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .math import mutual_information, shannon_entropy
from .permutation import marginal_permutation_test
from .stats import benjamini_hochberg
from .vis import humanize_col


def compute_predictiveness_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Computes the Asymmetric NMI (Uncertainty Coefficient) U(Y|X) = I(X, Y) / H(Y)
    between every ordered pair of columns.

    Mutual information is symmetric (I(X,Y) == I(Y,X)), so each unordered
    pair's MI is computed once and reused to derive both U(Y|X) and U(X|Y) —
    an earlier version of this function called `mutual_information` once per
    ORDERED pair, recomputing the same value twice for every (c1, c2).
    """
    cols = df.columns.tolist()
    entropies = {c: shannon_entropy(df[c].values) for c in cols}
    u_matrix: Dict[str, Dict[str, float]] = {c: {} for c in cols}

    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            mi = mutual_information(df[c1].values, df[c2].values)
            u_matrix[c1][c2] = float(mi / entropies[c2]) if entropies[c2] > 0 else 0.0
            u_matrix[c2][c1] = float(mi / entropies[c1]) if entropies[c1] > 0 else 0.0

    return u_matrix


def compute_significant_edges(
    df: pd.DataFrame,
    fdr_q: float = 0.05,
    n_permutations: int = 200,
    random_state: int | None = 42,
) -> Tuple[Dict[frozenset, float], Dict[frozenset, bool]]:
    """
    Runs one marginal permutation test per unordered pair of columns
    (H0: I(c1; c2) = 0) and applies Benjamini-Hochberg correction jointly
    across all M*(M-1)/2 pairs tested.

    Chain-mining over a graph of columns implicitly tests a large number of
    candidate edges/chains at once (every mediator-chain search in
    `mine_strong_links` and `graph_inference.run_graph_inference` draws its
    edges from this same pool). Neither function previously ran any
    significance test at all — edges were included purely on raw point-
    estimate NMI/Uncertainty-Coefficient magnitude against a fixed threshold
    (`min_nmi`), which cannot distinguish a genuine association from a
    small-sample fluctuation, and provides no control over how many of the
    O(M^2) to O(M^3) hypotheses implicitly being screened are false
    discoveries. This is the shared significance layer both callers gate on.

    Returns:
        (p_values, significant) — both dicts keyed by frozenset({c1, c2}).
    """
    cols = df.columns.tolist()
    pairs = list(itertools.combinations(cols, 2))
    p_values: Dict[frozenset, float] = {}

    if not pairs:
        return {}, {}

    raw_p = np.empty(len(pairs), dtype=float)
    for idx, (c1, c2) in enumerate(pairs):
        _, p_val, _ = marginal_permutation_test(
            df[c1].values,
            df[c2].values,
            n_permutations=n_permutations,
            alpha=fdr_q,
            random_state=random_state,
        )
        raw_p[idx] = p_val

    reject = benjamini_hochberg(raw_p, q=fdr_q)

    significant: Dict[frozenset, bool] = {}
    for (c1, c2), p_val, sig in zip(pairs, raw_p, reject):
        key = frozenset((c1, c2))
        p_values[key] = float(p_val)
        significant[key] = bool(sig)

    return p_values, significant


def mine_strong_links(
    df: pd.DataFrame,
    target: str = "class",
    min_nmi: float = 0.20,
    max_depth: int = 3,
    only_winning: bool = True,
    fdr_q: float = 0.05,
    n_permutations: int = 200,
    random_state: int | None = 42,
    translations: dict | None = None,
) -> Dict[str, Any]:
    """
    Mines mediator chains (input -> z1 [-> z2] -> target) whose EVERY edge is
    statistically significant (Benjamini-Hochberg corrected across all edges
    tested, see `compute_significant_edges`) and whose bottleneck strength
    clears `min_nmi`.

    Chain strength is the MINIMUM Uncertainty Coefficient among the chain's
    edges (a conservative "weakest-link" bound), not the PRODUCT of the
    edges' coefficients used by an earlier version of this function. That
    product was described in code comments as "Multiplicative Cascade...
    DPI compliant" — the Data Processing Inequality bounds mutual
    information for a genuine Markov chain (I(X;Y) <= I(X;Z) when X-Z-Y is
    Markov); it says nothing about, and does not justify, multiplying two
    *differently normalized* Uncertainty Coefficients from what may not even
    be a Markov chain in the data. That product-based score could and did
    exceed a weak but real direct association purely from the normalization
    arithmetic, for a mediator variable demonstrably unrelated to the
    input-target relationship (see the project code review's reproduction).
    `only_winning=True` previously meant "chain magnitude >= direct edge
    magnitude"; it now means "every edge in the chain, including the
    comparison direct edge, passed the significance test" — a chain that
    merely LOOKS bigger than a noisy direct estimate is no longer treated as
    a discovery.

    `translations` is an optional dataset-specific display table (see
    `vsf.vis.Translations`) used only for `path_labels`/`target_label`;
    with no `translations`, those fall back to raw column names.
    """
    cols = df.columns.tolist()
    nmi = compute_predictiveness_matrix(df)
    _, significant_edges = compute_significant_edges(
        df, fdr_q=fdr_q, n_permutations=n_permutations, random_state=random_state
    )

    def edge_ok(a: str, b: str) -> bool:
        return significant_edges.get(frozenset((a, b)), False)

    # Target columns to consider: if target is "all", search across entire dataset, else for specific target
    targets_to_search = [target] if target != "all" else cols

    found_chains = []

    for tgt in targets_to_search:
        for inp in cols:
            if inp == tgt:
                continue
            direct_nmi = float(nmi[inp].get(tgt, 0.0))
            direct_significant = edge_ok(inp, tgt)

            # 1. 2-step: inp -> z1 -> tgt
            for z1 in cols:
                if z1 == inp or z1 == tgt:
                    continue
                nmi_inp_z1 = float(nmi[inp].get(z1, 0.0))
                nmi_z1_tgt = float(nmi[z1].get(tgt, 0.0))
                chain_score = min(nmi_inp_z1, nmi_z1_tgt)
                edges_significant = edge_ok(inp, z1) and edge_ok(z1, tgt)
                fully_significant = edges_significant and direct_significant

                if chain_score >= min_nmi and (not only_winning or fully_significant):
                    gain = chain_score - direct_nmi
                    ratio = chain_score / max(0.001, direct_nmi)

                    found_chains.append({
                        "type": "mediator_1",
                        "type_label": "2 шага (1 медиатор)",
                        "input": inp,
                        "mediators": [z1],
                        "target": tgt,
                        "path": [inp, z1, tgt],
                        "path_labels": [humanize_col(inp, translations), humanize_col(z1, translations), humanize_col(tgt, translations)],
                        "chain_score": float(chain_score),
                        "direct_nmi": float(direct_nmi),
                        "gain": float(gain),
                        "ratio": float(ratio),
                        "edges_significant": bool(edges_significant),
                        "direct_significant": bool(direct_significant),
                        "significant": bool(fully_significant),
                    })

            # 2. 3-step: inp -> z1 -> z2 -> tgt
            if max_depth >= 3:
                for z1 in cols:
                    if z1 == inp or z1 == tgt:
                        continue
                    nmi_inp_z1 = float(nmi[inp].get(z1, 0.0))
                    if nmi_inp_z1 < min_nmi or not edge_ok(inp, z1):
                        continue

                    for z2 in cols:
                        if z2 == inp or z2 == tgt or z2 == z1:
                            continue
                        nmi_z1_z2 = float(nmi[z1].get(z2, 0.0))
                        nmi_z2_tgt = float(nmi[z2].get(tgt, 0.0))
                        chain_score = min(nmi_inp_z1, nmi_z1_z2, nmi_z2_tgt)
                        edges_significant = edge_ok(z1, z2) and edge_ok(z2, tgt)  # inp-z1 already checked above
                        fully_significant = edges_significant and direct_significant

                        if chain_score >= min_nmi and (not only_winning or fully_significant):
                            gain = chain_score - direct_nmi
                            ratio = chain_score / max(0.001, direct_nmi)

                            found_chains.append({
                                "type": "mediator_2",
                                "type_label": "3 шага (2 медиатора)",
                                "input": inp,
                                "mediators": [z1, z2],
                                "target": tgt,
                                "path": [inp, z1, z2, tgt],
                                "path_labels": [humanize_col(inp, translations), humanize_col(z1, translations), humanize_col(z2, translations), humanize_col(tgt, translations)],
                                "chain_score": float(chain_score),
                                "direct_nmi": float(direct_nmi),
                                "gain": float(gain),
                                "ratio": float(ratio),
                                "edges_significant": bool(edges_significant),
                                "direct_significant": bool(direct_significant),
                                "significant": bool(fully_significant),
                            })

    # Sort chains by chain_score (bottleneck strength) descending, then by gain
    found_chains.sort(key=lambda x: (x["chain_score"], x["gain"]), reverse=True)

    # Deduplicate: keep top 3 best paths per (input, target) pair
    seen_pairs: Dict[Tuple[str, str], int] = {}
    deduped = []
    for ch in found_chains:
        pair_key = (ch["input"], ch["target"])
        if pair_key not in seen_pairs:
            seen_pairs[pair_key] = 0
        if seen_pairs[pair_key] < 2:
            seen_pairs[pair_key] += 1
            deduped.append(ch)

    return {
        "target": target,
        "target_label": humanize_col(target, translations) if target != "all" else "Весь датасет (Все пары)",
        "min_nmi": min_nmi,
        "fdr_q": fdr_q,
        "total_found": len(deduped),
        "top_chains": deduped[:50]
    }
