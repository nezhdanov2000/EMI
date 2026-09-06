"""
VSF Complement Branch Discovery (CBD).

Extension to Independent Branch Discovery for finding feature combinations
where the TARGET ABSENCE (not presence) is concentrated. This enables
analysis of high-base-rate targets by inverting the perspective.

When you search for discover_branches(Z), you find where positive_class is
concentrated. When you search discover_complement_branches(Z), you find where
NOT positive_class is concentrated - i.e., cells with the lowest target
purity. This is a separate, equally valid analysis.

Mathematical foundation: if z_binary = 1[Z = positive_class], then
z_complement = 1 - z_binary defines a different binary target that the
existing framework handles without modification. All statistics (coverage,
purity, certificates, p-values) are computed on z_complement and are
therefore correct for the complementary question.

Key insight: the exhaustive search itself is unchanged. We only invert the
binary indicator before it enters _exhaustive_search, and then all metrics
automatically measure the opposite phenomenon. No duplication, no special
cases.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .avr import (
    BranchResult,
    _CandidateFactory,
    _discretize_target,
    _exhaustive_search,
    _report_branches,
    _resolve_positive_indicator,
    MAX_BRANCH_D,
    DEFAULT_N_PERMUTATIONS,
)
from .centers import CenterSpec
from .metrics import cell_codes
from .pmd import discretize_dataset


def discover_complement_branches(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_d: int = MAX_BRANCH_D,
    random_state: Optional[int] = 0,
    progress: Optional[Callable[[int, int], None]] = None,
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    n_permutations_centers: int = DEFAULT_N_PERMUTATIONS,
    n_permutations_familywise_coverage: int = 0,
    cv_splits: int = 5,
    cv_repeats: int = 5,
) -> Dict[int, BranchResult]:
    """
    Independent Branch Discovery on the COMPLEMENT of the target.

    For every d in {1, ..., min(max_d, M)}, independently computes

        S*_d = argmax_{S subset of {1,...,M}, |S| = d}
               coverage_score(1 - Z_pos; X_S)

    by exhaustively enumerating every C(M, d) combination. This finds feature
    subsets whose discrete centres are concentrated in the ABSENCE of the
    positive class value, rather than its presence.

    Key use case: when positive_class has high base rate (e.g., 90%+ of data),
    the standard `discover_branches()` is uninformative because almost any
    combination trivially contains the target. Searching the complement finds
    cells where the target value is ABSENT or rare - the inverse finding.

    All other aspects match `discover_branches()` exactly:
    - Same exhaustive, not approximate, search
    - Same ranking by coverage_score (coverage, -K, -mass, best Wilson lower)
    - Same centre definitions (purity floor tau, occupancy minimum m)
    - Same statistics: cross-validation, permutation nulls, familywise control
    - Same BranchResult output type

    The only semantic shift is in interpretation:
    - `discover_branches(Z, positive_class=v)`: "Find cells concentrated in v"
    - `discover_complement_branches(Z, positive_class=v)`: "Find cells
      concentrated in NOT-v", i.e. highest absence of v.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    Z : np.ndarray
        Target vector, shape (n_samples,).
    feature_names : list of str, optional
        Column names for X. Defaults to ['X_1', 'X_2', ...].
    max_d : int, default 4
        Maximum dimensionality to search.
    random_state : int, optional
        Seed for permutation nulls; makes the search reproducible.
    progress : callable, optional
        Callback (done, total) invoked during familywise null computation.
    positive_class : object, optional
        The raw target value whose ABSENCE is being concentrated. Required
        unless Z has exactly two distinct values (then the higher one is
        used, matching the complement of `discover_branches()`'s default).
    center_spec : CenterSpec, optional
        Purity floor (tau), error rate (alpha), multiplicity policy.
        Defaults to tau=0.90, alpha=0.05, Bonferroni.
    n_permutations_centers : int, default 999
        Replicates for uncorrected per-branch coverage p-value.
    n_permutations_familywise_coverage : int, default 0
        Replicates for familywise max-statistic p-value. Off by default
        (costs B × the entire candidate family); any published claim must set
        this to > 0.
    cv_splits : int, default 5
        Number of folds in stratified K-fold cross-validation.
    cv_repeats : int, default 5
        Number of repetitions of the K-fold. Set to 0 to skip out-of-sample
        coverage estimation.

    Returns
    -------
    dict[int, BranchResult]
        Keyed by dimensionality (1, 2, 3, 4). Each value is a BranchResult
        holding the optimal feature subset, its centres, coverage of the
        COMPLEMENT (absence of positive_class), and statistics.

    Raises
    ------
    ValueError
        If positive_class cannot be resolved (K-class target with no explicit
        positive_class), or if other parameters are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from vsf import discover_complement_branches
    >>> # High base rate target: 90% cases are "infected"
    >>> Z = np.random.binomial(1, 0.90, 1000)  # mostly 1s
    >>> X = np.random.randn(1000, 10)
    >>> # Find cells where infection is RARE
    >>> branches = discover_complement_branches(
    ...     X, Z, positive_class=1,
    ...     center_spec=CenterSpec(tau=0.90)
    ... )
    >>> # branches[2].centers.coverage is now the fraction of the
    >>> # NON-infected (Z=0) rows localised in high-purity NOT-infected cells

    Notes
    -----
    Mathematically, this is equivalent to calling:
        discover_branches(X, 1 - Z, positive_class=1)
    but we invert Z internally, so the user passes the original target and
    only changes the function name.

    The binarization logic is identical to `discover_branches()`: the target
    is categorically encoded, positive_class is resolved to one distinct raw
    value, and a 0/1 indicator is formed. The ONLY difference is the
    inversion of that indicator before the exhaustive search.

    All guarantees of `discover_branches()` hold for the complement:
    - Exact global optimum per dimensionality (no approximation)
    - Independent branches (non-nested, synergy-aware)
    - Honest statistical control via permutation nulls and cross-validation
    - Family-wise correction available (via n_permutations_familywise_coverage)
    """
    if max_d < 1 or max_d > MAX_BRANCH_D:
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
    if n_permutations_centers < 0 or n_permutations_familywise_coverage < 0:
        raise ValueError("permutation counts must be non-negative")

    spec = center_spec if center_spec is not None else CenterSpec()

    X_arr = np.asarray(X)
    Z_arr = np.asarray(Z).ravel()
    n_samples, n_features = X_arr.shape

    if feature_names is None:
        feature_names = [f"X_{j + 1}" for j in range(n_features)]

    branches: Dict[int, BranchResult] = {}
    if n_features == 0:
        return branches

    # ---- Setup identical to discover_branches ----
    X_discrete, bin_counts = discretize_dataset(X_arr)
    Z_discrete = _discretize_target(Z_arr)
    z_codes, n_rows = cell_codes(Z_discrete)

    effective_max_d = min(max_d, n_features)

    # ---- Resolve positive class ----
    z_binary = _resolve_positive_indicator(Z_arr, z_codes, n_rows, positive_class)
    if z_binary is None:
        raise ValueError(
            "discover_complement_branches needs a resolvable positive class: the "
            f"target has {n_rows} distinct values and no positive_class was given. "
            "Pass positive_class explicitly for a target with more than two values."
        )

    # ---- THE KEY DIFFERENCE: invert the binary indicator ----
    # z_binary: 1[Z = positive_class]
    # z_complement: 1[Z ≠ positive_class] = 1 - z_binary
    # All downstream metrics (coverage, purity, etc.) now measure the absence
    z_complement = 1 - z_binary

    # ---- Exhaustive search on the complement ----
    factory = _CandidateFactory(X_discrete, bin_counts, n_samples)
    best_by_d = _exhaustive_search(
        factory, z_complement.astype(np.int64), 2, [1], spec, effective_max_d
    )[0]
    if not best_by_d:
        return branches

    # ---- Report results (all statistics computed on z_complement) ----
    return _report_branches(
        factory,
        z_complement,
        best_by_d,
        feature_names,
        spec,
        effective_max_d,
        random_state=random_state,
        progress=progress,
        n_permutations_centers=n_permutations_centers,
        n_permutations_familywise_coverage=n_permutations_familywise_coverage,
        cv_splits=cv_splits,
        cv_repeats=cv_repeats,
    )
