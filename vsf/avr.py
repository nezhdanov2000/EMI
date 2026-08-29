"""
VSF Independent Branch Discovery (IBD) — v2.0 "Clean Core"

Replaces the v1.0 Adaptive Visual Routing (AVR) engine (greedy forward
selection + permutation-test stopping + Benjamini-Hochberg FDR control +
four rendering scenarios A/B/C/D). See Project_Master_Document.md Section 0
for the full revision history and Sections 4-4.6 for the formal spec this
module implements.

Core idea: for a chosen target Z, independently find, for every
dimensionality d in {1, 2, 3, 4}, the single best-scoring feature subset of
exactly that size — by HONEST, EXHAUSTIVE enumeration of every
C(M, d) combination, ranked by raw (unnormalized) mutual information. The
four resulting "branches" are NOT required to be nested: the winning pair
for d=2 need not contain either feature from the winning singleton at d=1.
This is intentional, not a bug — mutual information is not submodular in
general (Krause & Guestrin, 2005), so a greedy nested chain (the v1.0
approach) can provably miss a synergistic combination that an independent
per-d search finds. See `discover_branches`'s docstring for the formal
statement.

Explicitly and deliberately absent from this module (Project_Master_Document.md
Section 4.5, "Methodological Trade-off"): permutation-test significance
gating, Benjamini-Hochberg FDR correction, VIR thresholds, and Scenario
A/B/C/D routing. Ranking purely by raw MI reintroduces a real risk — the
"look-elsewhere effect" of scanning up to C(M,1)+...+C(M,4) candidate
combinations with no correction for multiple comparisons, and no
normalization for the differing degrees of freedom between combinations of
unequal category cardinality. This is a knowing, product-level trade-off,
not an oversight; it is documented here and in the master spec so nothing
downstream misrepresents a branch's raw MI as a validated, significant
result.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .math import mutual_information, normalized_mutual_information
from .pmd import adaptively_coarsen_bins, check_grid_capacity, discretize_dataset

# Hard ceiling on branch dimensionality. Matches the display's actual spatial
# encoding (3 coordinate axes + 1 time/frame axis, see vsf.vis's module
# docstring and Project_Master_Document.md Section 1.3) — there is no 5th/6th
# channel to search for, unlike v1.0's d_max=7 (which searched further than
# the display could ever show).
MAX_BRANCH_D = 4


@dataclass
class BranchResult:
    """
    One independently-discovered feature subset for a single dimensionality
    `d`. Deliberately does NOT carry v1.0's `scenario`/`vir`/`l_target`/
    `l_feat`/`xai_message`/`selection_history`/`n_significant_features`
    fields — none of those concepts exist in the v2.0 algorithm (see this
    module's docstring). `mi`/`nmi` are reported as plain numbers with no
    accompanying significance claim.
    """

    d: int
    selected_features: List[int]
    selected_feature_names: List[str]
    mi: float
    nmi: float


def _discretize_target(Z_arr: np.ndarray) -> np.ndarray:
    """
    Discretizes the target vector to integer codes, identical in behavior to
    v1.0's inline target-discretization block: string/object/bool targets
    and low-cardinality numeric targets are integer-coded directly via
    `np.unique`; high-cardinality continuous targets are PMD-binned first.
    """
    if Z_arr.dtype.kind in ("U", "S", "O", "b"):
        _, Z_discrete = np.unique(Z_arr, return_inverse=True)
    elif Z_arr.dtype.kind in ("f", "c") and len(np.unique(Z_arr)) > 20:
        Z_discrete, _, _ = discretize_dataset(Z_arr.reshape(-1, 1))
        Z_discrete = Z_discrete.ravel()
    else:
        _, Z_discrete = np.unique(Z_arr, return_inverse=True)
    return Z_discrete.astype(int)


def discover_branches(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[List[str]] = None,
    feature_channels: Optional[List[str]] = None,
    max_d: int = MAX_BRANCH_D,
) -> Dict[int, BranchResult]:
    """
    Independent Branch Discovery (Project_Master_Document.md Section 4.3).

    For every d in {1, ..., min(max_d, M)}, independently computes

        S*_d = argmax_{S subset of {1,...,M}, |S|=d}  I(Z; X_S)

    by exhaustively enumerating every C(M, d) combination and evaluating its
    mutual information with Z — never a greedy approximation, never a
    predfiltered candidate pool. Returns up to `max_d` branches, keyed by
    dimensionality; a key is missing only when `M < d` (not enough columns
    in `X` to form a subset of that size).

    Why exhaustive, not greedy: I(Z; X_S) is not submodular in general
    (Krause & Guestrin, 2005). A greedy chain that commits to the single
    best feature at d=1 and only ever ADDS to it can never discover a pair
    {a, b} that is jointly informative while neither a nor b is
    individually strong — exactly the synergy case Project_Master_Document.md
    Section 4.4 requires each branch to be capable of finding. Running d=1,
    2, 3, 4 as four fully independent exhaustive searches is the only way to
    guarantee finding the true S*_d at every dimensionality without assuming
    nesting.

    Cost: see Project_Master_Document.md Section 4.6 for the full
    complexity table. This function makes no attempt to bound the number of
    combinations evaluated — by explicit product decision, "always honest
    full enumeration," regardless of M. Callers with very wide datasets
    (M gtrsim 50-100) should expect this to take minutes; that is the
    accepted cost of never approximating, not a defect to silently work
    around here.

    Ranking is by RAW (unnormalized) mutual information, per the product
    spec — NOT by NMI. See this module's docstring for why that is a
    conscious, documented trade-off rather than an oversight.
    """
    if max_d < 1 or max_d > MAX_BRANCH_D:
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")

    X_arr = np.asarray(X)
    Z_arr = np.asarray(Z).ravel()
    n_samples, n_features = X_arr.shape

    if feature_names is None:
        feature_names = [f"X_{j + 1}" for j in range(n_features)]

    branches: Dict[int, BranchResult] = {}
    if n_features == 0:
        return branches

    X_discrete, bin_counts, _ = discretize_dataset(X_arr, feature_channels=feature_channels)
    Z_discrete = _discretize_target(Z_arr)

    effective_max_d = min(max_d, n_features)
    for d in range(1, effective_max_d + 1):
        best_combo: Optional[tuple] = None
        best_mi = -1.0

        for combo in itertools.combinations(range(n_features), d):
            X_S = X_discrete[:, combo]
            k_comb = [bin_counts[j] for j in combo]
            if not check_grid_capacity(k_comb, n_samples):
                # Grid Capacity Limit (Project_Master_Document.md Section 2.4):
                # coarsen just this combination's joint code before scoring
                # it, so a sparse joint table doesn't distort the ranking
                # against combinations that didn't need coarsening.
                X_S = adaptively_coarsen_bins(X_S, n_samples)

            mi = mutual_information(Z_discrete, X_S)
            if mi > best_mi:
                best_mi = mi
                best_combo = combo

        if best_combo is None:
            continue

        X_Sstar = X_discrete[:, list(best_combo)]
        nmi = normalized_mutual_information(Z_discrete, X_Sstar)
        branches[d] = BranchResult(
            d=d,
            selected_features=list(best_combo),
            selected_feature_names=[feature_names[j] for j in best_combo],
            mi=float(best_mi),
            nmi=float(nmi),
        )

    return branches


class BranchEngine:
    """
    Thin, stateless wrapper around `discover_branches` matching v1.0's
    `AVREngine(...).fit(...)` call shape, so callers (server handlers,
    `vsf.dashboard`, tests) construct-then-fit as before. Carries no
    significance-testing parameters (`alpha`, `fdr_q`, `n_permutations`,
    `random_state` from v1.0 are all gone — this algorithm is fully
    deterministic and uses no randomness at all) and no `vir_threshold`
    (there is no scenario routing to threshold).
    """

    def __init__(self, max_d: int = MAX_BRANCH_D):
        if max_d < 1 or max_d > MAX_BRANCH_D:
            raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
        self.max_d = max_d

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        feature_names: Optional[List[str]] = None,
        feature_channels: Optional[List[str]] = None,
    ) -> Dict[int, BranchResult]:
        return discover_branches(
            X, Z, feature_names=feature_names, feature_channels=feature_channels, max_d=self.max_d
        )
