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


class Scenario(str, Enum):
    SCENARIO_A = "SCENARIO_A"  # Minimalist (2D/3D, d* <= 3)
    SCENARIO_B = "SCENARIO_B"  # Full Load (4D-7D, d* in [4,7] and VIR >= 0.85)
    SCENARIO_C = "SCENARIO_C"  # Warning (>7D, d* = 7 and VIR < 0.85)
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


class AVREngine:
    """
    Adaptive Visual Routing Engine (AVR)
    
    Parameters:
        alpha: Significance level for permutation tests (default 0.01)
        vir_threshold: VIR ratio threshold for Scenario B vs C (default 0.85)
        max_d: Upper perceptual channel limit (default 7)
        n_permutations: Number of iterations B for permutation test (default 1000)
    """

    def __init__(
        self,
        alpha: float = 0.01,
        vir_threshold: float = 0.85,
        max_d: int = 7,
        n_permutations: int = 1000,
        random_state: int | None = 42,
    ):
        self.alpha = alpha
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
        X_arr = np.asarray(X, dtype=object)
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
        # PHASE 1: Noise Filtering (Marginal Permutation Test)
        # -------------------------------------------------------------
        significant_features = []
        marginal_mis = {}
        
        for j in range(n_features):
            x_j = X_discrete[:, j]
            i_obs, p_val, is_sig = marginal_permutation_test(
                Z_discrete,
                x_j,
                n_permutations=self.n_permutations,
                alpha=self.alpha,
                random_state=self.random_state,
            )
            if is_sig:
                significant_features.append(j)
                marginal_mis[j] = i_obs

        # Phase 1 Guard: If no features pass noise filter -> Scenario D (Chaos)
        if len(significant_features) == 0:
            xai_msg = (
                f"No statistically significant structure detected in data "
                f"(p > {self.alpha} for all features). Visualization aborted."
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
                selection_history=[]
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
            
            for j in candidates:
                x_j = X_discrete[:, j]
                x_comb = np.column_stack([X_S_curr, x_j])
                
                # Check grid capacity protection limit
                k_comb = [len(np.unique(x_comb[:, c])) for c in range(x_comb.shape[1])]
                if not check_grid_capacity(k_comb, n_samples):
                    x_comb = adaptively_coarsen_bins(x_comb, n_samples)
                    
                i_base = mutual_information(Z_discrete, X_S_curr)
                i_comb = mutual_information(Z_discrete, x_comb)
                delta_i = max(0.0, i_comb - i_base)
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

        # Routing Triggers:
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
        elif 4 <= d_star <= 7 and vir >= self.vir_threshold:
            scenario = Scenario.SCENARIO_B
            xai_msg = (
                f"SCENARIO B (Full Load): Selected d* = {d_star} axes. "
                f"VIR = {vir*100:.1f}% (>= {self.vir_threshold*100:.0f}%), NMI = {nmi_S_star*100:.1f}%. High-dimensional structure rendered."
            )
        elif d_star == 7 and vir < self.vir_threshold:
            scenario = Scenario.SCENARIO_C
            xai_msg = (
                f"SCENARIO C (Warning: >7D): Feature Projection Loss = {l_feat*100:.1f}%. "
                f"Current visualization is incomplete. {n_features - 7} significant features omitted."
            )
        else:
            # Fallback for boundary combinations
            scenario = Scenario.SCENARIO_B if vir >= self.vir_threshold else Scenario.SCENARIO_C
            xai_msg = f"Routing completed: d* = {d_star}, VIR = {vir*100:.1f}%."

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
            selection_history=selection_history
        )
