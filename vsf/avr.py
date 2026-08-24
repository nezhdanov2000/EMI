"""
VSF AVR Module: Adaptive Visual Routing Engine
Implements 3-Phase feature selection (Noise Filter -> Greedy Forward Selection -> Routing)
and rendering scenario triggers (Scenario A, B, C, D).
"""

import itertools
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .math import mutual_information, normalized_mutual_information
from .permutation import conditional_permutation_test, marginal_permutation_test
from .pmd import adaptively_coarsen_bins, check_grid_capacity, discretize_dataset
from .stats import benjamini_hochberg


class Scenario(str, Enum):
    SCENARIO_A = "SCENARIO_A"  # Minimalist (2D/3D, d* <= 3)
    SCENARIO_B = "SCENARIO_B"  # Full Load (4D-7D, d* in [4,7]); message flags incompleteness if VIR < threshold
    SCENARIO_C = "SCENARIO_C"  # Warning (>7D would be needed: d* == max_d and VIR < threshold)
    SCENARIO_D = "SCENARIO_D"  # Chaos / Block (d* = 0, noise dataset)


@dataclass
class AVRResult:
    d_star: int
    selected_features: list[int]
    selected_feature_names: list[str]
    scenario: Scenario
    vir: float
    l_target: float
    l_feat: float
    nmi_full: float
    xai_message: str
    submodularity_ratio: float | None = None
    selection_history: list[dict] | None = None
    # Number of features that survived Phase 1's FDR-controlled noise filter
    # (i.e. |F|). d_star <= n_significant_features always; the gap between
    # them is how many statistically-significant features Phase 2's greedy
    # search declined to add because their conditional gain didn't clear
    # significance, not "features beyond the 7D cap" — see the Phase 3
    # routing logic in AVREngine.fit for why this distinction matters.
    n_significant_features: int = 0


class AVREngine:
    """
    Adaptive Visual Routing Engine (AVR)

    Parameters:
        alpha: Significance level for the Phase 2 conditional permutation
            test's per-step stopping decision (default 0.01). This is a
            sequential stopping rule (one test per greedy step), not a batch
            of simultaneous hypotheses, so it is deliberately left as a
            fixed-alpha test rather than FDR-corrected — see `fdr_q` for the
            batch test that Phase 1 performs.
        fdr_q: Target False Discovery Rate for Phase 1's marginal noise
            filter (default 0.05, per Project_Master_Document.md Section
            4.5.3: "Benjamini-Hochberg correction (FDR q <= 0.05)"). Phase 1
            tests every input feature against Z independently in one batch,
            so — unlike Phase 2 — its significance decisions are made
            jointly via Benjamini-Hochberg rather than by thresholding each
            feature's raw p-value against `alpha` in isolation; doing the
            latter across M features lets the family-wise false discovery
            rate grow with M instead of staying bounded at `fdr_q`.
        vir_threshold: VIR ratio threshold for Scenario B vs C (default 0.85)
        max_d: Upper perceptual channel limit (default 7)
        n_permutations: Number of iterations B for permutation test (default 1000)
    """

    def __init__(
        self,
        alpha: float = 0.01,
        fdr_q: float = 0.05,
        vir_threshold: float = 0.85,
        max_d: int = 7,
        n_permutations: int = 1000,
        random_state: int | None = 42,
    ):
        self.alpha = alpha
        self.fdr_q = fdr_q
        self.vir_threshold = vir_threshold
        self.max_d = max_d
        self.n_permutations = n_permutations
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        feature_names: list[str] | None = None,
        feature_channels: list[str] | None = None,
        offline_brute_force: bool = False,
    ) -> AVRResult:
        """
        Runs the full 3-Phase AVR algorithm on dataset X and target Z.
        """
        # Let NumPy infer the natural dtype: a homogeneous numeric matrix stays
        # numeric (fast path through PMD/np.unique), and a genuinely mixed
        # input (e.g. a DataFrame.values with both string and numeric columns)
        # already comes back as dtype=object from np.asarray without forcing
        # it — forcing dtype=object unconditionally routed every dataset,
        # numeric or not, through NumPy's much slower per-element object path.
        X_arr = np.asarray(X)
        Z_arr = np.asarray(Z).ravel()
        
        n_samples, n_features = X_arr.shape
        if feature_names is None:
            feature_names = [f"X_{j+1}" for j in range(n_features)]
            
        # Step 1: Discretization via PMD
        X_discrete, bin_counts, _ = discretize_dataset(
            X_arr, feature_channels=feature_channels
        )
        # Discretize Z target
        if Z_arr.dtype.kind in ('U', 'S', 'O', 'b'):
            _, Z_discrete = np.unique(Z_arr, return_inverse=True)
            Z_discrete = Z_discrete.astype(int)
        elif Z_arr.dtype.kind in ('f', 'c') and len(np.unique(Z_arr)) > 20:
            Z_discrete, _, _ = discretize_dataset(Z_arr.reshape(-1, 1))
            Z_discrete = Z_discrete.ravel().astype(int)
        else:
            _, Z_discrete = np.unique(Z_arr, return_inverse=True)
            Z_discrete = Z_discrete.astype(int)
            
        # -------------------------------------------------------------
        # PHASE 1: Noise Filtering (Marginal Permutation Test + FDR Control)
        # -------------------------------------------------------------
        # Run every feature's marginal permutation test first and collect the
        # raw p-values, THEN decide significance jointly via Benjamini-Hochberg
        # at fdr_q. Thresholding each feature's p-value against `alpha`
        # independently (the previous behavior) lets the family-wise false
        # discovery rate grow with n_features instead of staying bounded —
        # exactly the multiple-testing exposure Project_Master_Document.md
        # Section 4.5.3 requires FDR control to prevent.
        marginal_mis: dict[int, float] = {}
        marginal_pvals = np.ones(n_features, dtype=float)

        for j in range(n_features):
            x_j = X_discrete[:, j]
            i_obs, p_val, _ = marginal_permutation_test(
                Z_discrete,
                x_j,
                n_permutations=self.n_permutations,
                alpha=self.alpha,
                random_state=self.random_state,
            )
            marginal_mis[j] = i_obs
            marginal_pvals[j] = p_val

        fdr_reject = (
            benjamini_hochberg(marginal_pvals, q=self.fdr_q)
            if n_features > 0
            else np.zeros(0, dtype=bool)
        )
        significant_features = [j for j in range(n_features) if fdr_reject[j]]

        # Phase 1 Guard: If no features pass noise filter -> Scenario D (Chaos)
        if len(significant_features) == 0:
            xai_msg = (
                f"No statistically significant structure detected in data "
                f"(FDR q > {self.fdr_q} for all {n_features} features under "
                f"Benjamini-Hochberg correction). Visualization aborted."
            )
            return AVRResult(
                d_star=0,
                selected_features=[],
                selected_feature_names=[],
                scenario=Scenario.SCENARIO_D,
                vir=0.0,
                l_target=1.0,
                l_feat=1.0,
                nmi_full=0.0,
                xai_message=xai_msg,
                selection_history=[],
                n_significant_features=0,
            )

        # -------------------------------------------------------------
        # PHASE 2: Greedy Forward Selection
        # -------------------------------------------------------------
        # Sort significant features F by descending marginal MI
        sorted_F = sorted(
            significant_features, key=lambda j: marginal_mis[j], reverse=True
        )
        
        X_F_all = X_discrete[:, sorted_F]
        i_F_all = mutual_information(Z_discrete, X_F_all)
        selection_history = []
        
        # Pick best first feature
        S = [sorted_F[0]]
        i_1d = mutual_information(Z_discrete, X_discrete[:, S])
        nmi_1d = normalized_mutual_information(Z_discrete, X_discrete[:, S])
        
        alt_1d = []
        for j in sorted_F[1:4]:
            i_alt = marginal_mis[j]
            nmi_alt = normalized_mutual_information(Z_discrete, X_discrete[:, [j]])
            alt_1d.append({
                'feature': feature_names[j],
                'mi': float(i_alt),
                'nmi': float(nmi_alt),
                'vir': float(i_alt / i_F_all) if i_F_all > 1e-12 else 1.0,
                'delta_mi': float(i_alt)
            })
            
        selection_history.append({
            'step': 1,
            'feature': feature_names[S[0]],
            'features_so_far': [feature_names[j] for j in S],
            'mi': float(i_1d),
            'nmi': float(nmi_1d),
            'vir': float(i_1d / i_F_all) if i_F_all > 1e-12 else 1.0,
            'delta_mi': float(i_1d),
            'alternatives': alt_1d
        })
        
        # Greedy selection for d = 2 to min(max_d, |F|)
        max_steps = min(self.max_d, len(sorted_F))
        
        for d in range(2, max_steps + 1):
            candidates = [j for j in sorted_F if j not in S]
            if not candidates:
                break
                
            best_candidate = None
            best_delta_i = -1.0
            candidate_scores = []

            X_S_curr = X_discrete[:, S]
            # I(Z; X_S_curr) does not depend on the candidate j — computing it
            # once here instead of inside the loop below avoids |candidates|
            # redundant full mutual-information passes (each an O(N log N)
            # entropy computation) per greedy step.
            i_base = mutual_information(Z_discrete, X_S_curr)

            for j in candidates:
                x_j = X_discrete[:, j]
                x_comb = np.column_stack([X_S_curr, x_j])

                # Check grid capacity protection limit
                k_comb = [len(np.unique(x_comb[:, c])) for c in range(x_comb.shape[1])]
                if not check_grid_capacity(k_comb, n_samples):
                    x_comb = adaptively_coarsen_bins(x_comb, n_samples)
                    i_comb = mutual_information(Z_discrete, x_comb)
                    # x_comb was just coarsened to a coarser bin resolution
                    # than X_S_curr; comparing it against the hoisted i_base
                    # (computed at the ORIGINAL resolution) would score
                    # delta_i against two different binnings. Recompute the
                    # baseline at the matching coarsened resolution for this
                    # candidate only — the common case (no coarsening
                    # triggered) still uses the hoisted i_base below.
                    i_base_for_j = mutual_information(Z_discrete, x_comb[:, :-1])
                else:
                    i_comb = mutual_information(Z_discrete, x_comb)
                    i_base_for_j = i_base

                delta_i = max(0.0, i_comb - i_base_for_j)
                candidate_scores.append((j, delta_i, i_comb))
                
                if delta_i > best_delta_i:
                    best_delta_i = delta_i
                    best_candidate = j
                    
            if best_candidate is None or best_delta_i <= 0.0:
                break
                
            # Perform Conditional Permutation Test for best candidate
            delta_obs, p_val, is_sig = conditional_permutation_test(
                Z_discrete,
                X_discrete[:, best_candidate],
                X_S_curr,
                n_permutations=self.n_permutations,
                alpha=self.alpha,
                random_state=self.random_state,
            )
            
            if not is_sig:
                # Stop selection if marginal gain is not statistically significant
                break
                
            S.append(best_candidate)
            
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            top_alts = [item for item in candidate_scores if item[0] != best_candidate][:3]
            
            alt_d = []
            for j_alt, d_alt, i_alt in top_alts:
                x_alt_comb = np.column_stack([X_S_curr, X_discrete[:, j_alt]])
                nmi_alt = normalized_mutual_information(Z_discrete, x_alt_comb)
                alt_d.append({
                    'feature': feature_names[j_alt],
                    'mi': float(i_alt),
                    'nmi': float(nmi_alt),
                    'vir': float(i_alt / i_F_all) if i_F_all > 1e-12 else 1.0,
                    'delta_mi': float(d_alt)
                })
            
            i_S_curr = mutual_information(Z_discrete, X_discrete[:, S])
            nmi_S_curr = normalized_mutual_information(Z_discrete, X_discrete[:, S])
            selection_history.append({
                'step': d,
                'feature': feature_names[best_candidate],
                'features_so_far': [feature_names[j] for j in S],
                'mi': float(i_S_curr),
                'nmi': float(nmi_S_curr),
                'vir': float(i_S_curr / i_F_all) if i_F_all > 1e-12 else 1.0,
                'delta_mi': float(best_delta_i),
                'alternatives': alt_d
            })
            
        d_star = len(S)
        selected_names = [feature_names[j] for j in S]

        # Submodularity ratio check (Optional Brute Force Benchmark for M <= 20)
        submod_ratio = None
        if offline_brute_force and n_features <= 20:
            best_opt_mi = 0.0
            for d_search in range(1, min(self.max_d, n_features) + 1):
                for comb in itertools.combinations(range(n_features), d_search):
                    mi_comb = mutual_information(Z_discrete, X_discrete[:, list(comb)])
                    best_opt_mi = max(best_opt_mi, mi_comb)
            greedy_mi = mutual_information(Z_discrete, X_discrete[:, S])
            if best_opt_mi > 0:
                submod_ratio = greedy_mi / best_opt_mi

        # -------------------------------------------------------------
        # PHASE 3: Scenario Routing & Loss Calculation
        # -------------------------------------------------------------
        X_S_star = X_discrete[:, S] if S else np.zeros((n_samples, 0))
        
        i_S_star = mutual_information(Z_discrete, X_S_star) if S else 0.0
        
        # Calculate VIR = I(Z; X_S*) / I(Z; X_F)
        if i_F_all > 1e-12:
            vir = float(i_S_star / i_F_all)
        else:
            vir = 1.0
            
        # Target Loss = 1 - NMI(Z; X_S*)
        nmi_S_star = normalized_mutual_information(Z_discrete, X_S_star) if S else 0.0
        l_target = float(1.0 - nmi_S_star)
        
        # Feature Loss = 1 - I(Z; X_S*) / I(Z; X_F) = 1 - VIR
        l_feat = float(1.0 - vir)
        
        # Full NMI of entire dataset
        nmi_full = float(normalized_mutual_information(Z_discrete, X_discrete))

        # Number of Phase-1-significant features that did NOT make it into the
        # final basis S* — the quantity that belongs in an "omitted features"
        # message. This is never negative (d_star = |S| <= |sorted_F| by
        # construction) and is distinct from "features beyond the max_d cap":
        # greedy selection can stop below max_d simply because no remaining
        # candidate's conditional gain was significant, independent of the cap.
        n_unselected_significant = len(sorted_F) - d_star

        # Routing Triggers. This chain is exhaustive over d_star in [1, max_d]
        # (d_star == 0 already returned early as SCENARIO_D above, so d_star
        # >= 1 here always): every combination of d_star and vir lands in
        # exactly one branch below, with no mislabeled catch-all.
        if d_star <= 3:
            scenario = Scenario.SCENARIO_A
            if l_target > 0.70:
                xai_msg = (
                    f"SCENARIO A (Minimalist): Selected d* = {d_star} axes. "
                    f"Projection captures {vir*100:.1f}% of dataset information (VIR). "
                    f"However, target association is weak (NMI = {nmi_S_star*100:.1f}%, Target Loss = {l_target*100:.1f}%), centers are mixed."
                )
            else:
                xai_msg = (
                    f"SCENARIO A (Minimalist): Selected d* = {d_star} axes. "
                    f"Target structure is well-explained by {d_star} features (VIR = {vir*100:.1f}%, NMI = {nmi_S_star*100:.1f}%)."
                )
        elif vir >= self.vir_threshold:
            # 4 <= d_star <= max_d and VIR clears the threshold: genuine Full Load.
            scenario = Scenario.SCENARIO_B
            xai_msg = (
                f"SCENARIO B (Full Load): Selected d* = {d_star} axes. "
                f"VIR = {vir*100:.1f}% (>= {self.vir_threshold*100:.0f}%), NMI = {nmi_S_star*100:.1f}%. High-dimensional structure rendered."
            )
        elif d_star >= self.max_d:
            # Genuinely hit the perceptual channel cap (d_star == max_d) and
            # VIR is still below threshold: more axes would be needed than
            # the display supports. This is the only case that should be
            # framed as a ">max_d" warning.
            scenario = Scenario.SCENARIO_C
            xai_msg = (
                f"SCENARIO C (Warning: >{self.max_d}D): Feature Projection Loss = {l_feat*100:.1f}%. "
                f"Current visualization is incomplete: {n_unselected_significant} additional "
                f"statistically significant feature(s) beyond the {self.max_d}-axis display cap "
                f"were not rendered."
            )
        else:
            # 4 <= d_star < max_d and VIR < threshold: greedy selection
            # plateaued (no remaining candidate cleared the conditional
            # significance test) before reaching either full coverage or the
            # dimensionality cap. Still a Full Load axis count, but the
            # message must say so honestly instead of claiming a ">max_d"
            # situation that isn't what happened.
            scenario = Scenario.SCENARIO_B
            xai_msg = (
                f"SCENARIO B (Full Load - Incomplete): Selected d* = {d_star} axes, below the "
                f"{self.max_d}-axis cap. VIR = {vir*100:.1f}% (< {self.vir_threshold*100:.0f}% target). "
                f"Selection plateaued: {n_unselected_significant} additional statistically "
                f"significant feature(s) did not clear the conditional permutation test and "
                f"were left out, so this projection is missing information rather than being "
                f"cap-limited."
            )

        return AVRResult(
            d_star=d_star,
            selected_features=S,
            selected_feature_names=selected_names,
            scenario=scenario,
            vir=vir,
            l_target=l_target,
            l_feat=l_feat,
            nmi_full=float(nmi_full),
            xai_message=xai_msg,
            submodularity_ratio=submod_ratio,
            selection_history=selection_history,
            n_significant_features=len(sorted_F),
        )
