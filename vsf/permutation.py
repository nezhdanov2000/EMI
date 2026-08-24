"""
VSF Permutation Engine: Statistical Significance Testing
Implements Marginal Permutation Test (Definition 3) and Conditional Stratified
Permutation Test (Definition 4) for Mutual Information feature selection.
"""

import warnings

import numpy as np

from .backend import as_backend, as_numpy, get_backend
from .math import joint_entropy, shannon_entropy

# Above this many (k_a * k_b) joint cells, a dense bincount table is no longer
# a clear win over the general-purpose unique()-based entropy path (memory
# footprint grows linearly with cell count, most of it empty for sparse
# high-cardinality joints), so the fast path bows out and callers fall back.
_MAX_FAST_JOINT_CELLS = 5_000_000


def _integer_encode_1d(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Encodes a 1D array of any dtype to compact int64 codes in [0, k). Returns (codes, k)."""
    arr_np = np.asarray(as_numpy(arr)).ravel()
    _, codes = np.unique(arr_np, return_inverse=True)
    codes = codes.astype(np.int64, copy=False)
    k = int(codes.max()) + 1 if codes.size else 0
    return codes, k


def _integer_encode_rows(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Encodes the rows of a (N, D) array to compact int64 codes in [0, k). Returns (codes, k)."""
    arr_np = np.asarray(as_numpy(arr))
    if arr_np.ndim == 1:
        arr_np = arr_np.reshape(-1, 1)
    _, codes = np.unique(arr_np, axis=0, return_inverse=True)
    codes = codes.astype(np.int64, copy=False)
    k = int(codes.max()) + 1 if codes.size else 0
    return codes, k


def _bincount_entropy(codes: np.ndarray, minlength: int) -> float:
    """O(N) Shannon entropy in bits of a pre-encoded int64 code array via bincount."""
    counts = np.bincount(codes, minlength=minlength)
    counts = counts[counts > 0]
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(-np.sum(probs * np.log2(probs)))


def _fast_joint_entropy(
    codes_a: np.ndarray, k_a: int, codes_b: np.ndarray, k_b: int
) -> float | None:
    """
    O(N) joint entropy H(A, B) in bits from two pre-encoded int64 code arrays,
    computed via a single bincount over the combined code `a * k_b + b`
    instead of sorting the joint rows (as the generic shannon_entropy/
    joint_entropy path does via np.unique). Used inside permutation loops
    where one side is re-permuted every iteration but its cardinality and the
    other side's codes are already known and constant.

    Returns None (caller must fall back to the general-purpose path) when the
    dense combined table would exceed _MAX_FAST_JOINT_CELLS, or when either
    side is empty.
    """
    n_cells = k_a * k_b
    if k_a <= 0 or k_b <= 0 or n_cells > _MAX_FAST_JOINT_CELLS:
        return None
    combined = codes_a * k_b + codes_b
    return _bincount_entropy(combined, minlength=n_cells)


def marginal_permutation_test(
    Z: np.ndarray,
    X_j: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 0.01,
    random_state: int | None = None,
) -> tuple[float, float, bool]:
    """
    Performs Marginal Permutation Test for H0: I(Z; X_j) = 0 (Definition 3).

    Permutes target vector Z to construct empirical null distribution.

    Returns:
        (i_obs, p_value, is_significant)

    Note on `is_significant`: this flag compares the test's own raw p-value
    against `alpha` and is provided for standalone/unit-test use. Callers
    that run this test across many features in one screening pass (e.g.
    AVREngine's Phase 1 noise filter) MUST NOT rely on this per-test flag to
    decide significance — with M simultaneous tests at a fixed alpha, the
    family-wise false discovery rate is uncontrolled. Collect the raw
    `p_value` from every test instead and pass the full array through
    `vsf.stats.benjamini_hochberg` to get an FDR-controlled significance
    decision, per Project_Master_Document.md Section 4.5.3.
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

    if xp.__name__ != 'cupy':
        # Fast CPU path: pre-encode both sides to compact integer codes once,
        # then permute the (cheap, integer) Z codes and read off the joint
        # entropy via an O(N) bincount instead of an O(N log N) unique/sort
        # on the raw joint array every one of the n_permutations iterations.
        z_codes, k_z = _integer_encode_1d(z_arr)
        x_codes, k_x = _integer_encode_1d(x_arr)
        z_codes_perm = z_codes.copy()
        rng_np = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(random_state)

        use_fast = (k_z * k_x) <= _MAX_FAST_JOINT_CELLS
        for _ in range(n_permutations):
            rng_np.shuffle(z_codes_perm)
            if use_fast:
                h_z_x_perm = _fast_joint_entropy(z_codes_perm, k_z, x_codes, k_x)
            else:
                h_z_x_perm = None
            if h_z_x_perm is None:
                h_z_x_perm = joint_entropy(z_codes_perm, x_codes)
            i_perm = max(0.0, float(h_z + h_x - h_z_x_perm))
            if i_perm >= i_obs:
                count_exceed += 1
    else:
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


def _get_strata_groups(X_S: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Precomputes strata index-groups for the stratified conditional
    permutation test (Definition 4).

    Only samples that share the SAME X_S conditioning value may be exchanged
    with one another under H0. Strata with fewer than 2 members contain
    nothing to permute against (permuting a single element against itself is
    the identity map), and are therefore EXCLUDED from every permutation
    entirely: their Z-value is held fixed across all n_permutations draws,
    exactly as it must be to remain conditionally exchangeable only within
    its own (unobserved-elsewhere) stratum.

    An earlier version of this function pooled all such singleton/sparse
    strata into one shared "fallback" group and permuted across it — that
    swaps Z-values between samples whose X_S values actually differ, which
    breaks the exchangeability assumption the stratified null relies on and
    biases the resulting p-value. Do not reintroduce that pooling.

    Returns:
        (group_list, n_excluded) — the list of permutable index-groups (each
        of size >= 2, sharing one X_S value), and the count of samples
        excluded from permutation because their stratum was a singleton.
    """
    x_s_arr = np.asarray(as_numpy(X_S))
    if x_s_arr.ndim == 1:
        x_s_arr = x_s_arr.reshape(-1, 1)

    _, strata_indices = np.unique(x_s_arr, axis=0, return_inverse=True)
    unique_strata, counts = np.unique(strata_indices, return_counts=True)
    permutable_strata = set(unique_strata[counts >= 2].tolist())

    groups: dict[int, list[int]] = {}
    n_excluded = 0
    for idx, s in enumerate(strata_indices):
        s_int = int(s)
        if s_int not in permutable_strata:
            n_excluded += 1
            continue
        groups.setdefault(s_int, []).append(idx)

    group_list = [np.array(indices, dtype=np.int64) for indices in groups.values()]
    return group_list, n_excluded


def conditional_permutation_test(
    Z: np.ndarray,
    X_j: np.ndarray,
    X_S: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 0.01,
    random_state: int | None = None,
    max_excluded_fraction: float = 0.5,
) -> tuple[float, float, bool]:
    """
    Performs Conditional Stratified Permutation Test for H0: Delta I(j | S) = 0 (Definition 4).

    Tests if adding feature X_j to subset S brings statistically significant new information about Z.

    Samples whose X_S conditioning value is unique in the data (a singleton
    stratum) carry no information about the within-stratum null and are
    excluded from the permutation (see `_get_strata_groups`). If more than
    `max_excluded_fraction` of the sample is excluded this way — common once
    |S| grows and the X_S grid gets sparse — a UserWarning is raised, since
    the effective sample size backing the null distribution has shrunk
    enough that the test's power should not be assumed to hold.

    Returns:
        (delta_i_obs, p_value, is_significant)

    Note on `is_significant`: see the equivalent note on
    `marginal_permutation_test` — when this test is invoked repeatedly
    across many candidate features/chains in one run, collect raw p-values
    and apply `vsf.stats.benjamini_hochberg` rather than trusting each
    per-test flag in isolation.
    """
    xp = get_backend()
    z_arr = as_backend(Z).ravel()
    x_j_arr = as_backend(X_j).ravel()
    x_s_arr = as_backend(X_S)

    if x_s_arr.ndim == 1:
        x_s_arr = x_s_arr.reshape(-1, 1)

    cpu_rng = np.random.default_rng(random_state)
    z_np = np.asarray(as_numpy(z_arr)).copy()
    n_samples = len(z_np)
    strata_groups, n_excluded = _get_strata_groups(x_s_arr)

    if n_samples > 0 and (n_excluded / n_samples) > max_excluded_fraction:
        warnings.warn(
            f"Conditional permutation test: {n_excluded}/{n_samples} samples "
            f"({n_excluded / n_samples:.1%}) fall into singleton X_S strata and "
            "are excluded from permutation (their Z-value is held fixed). "
            "The effective sample size behind this test's null distribution "
            "is reduced accordingly; treat the resulting p-value as "
            "conservative/low-power rather than as evidence of a null effect.",
            stacklevel=2,
        )

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

    if xp.__name__ != 'cupy':
        # Fast CPU path: encode Z, X_S and X_S+X_j to compact integer codes
        # once, then recompute the two joint entropies each iteration via
        # O(N) bincount instead of two O(N log N) unique/sort passes.
        z_codes, k_z = _integer_encode_1d(z_np)
        xs_codes, k_xs = _integer_encode_rows(as_numpy(x_s_arr))
        xsj_codes, k_xsj = _integer_encode_rows(as_numpy(x_s_plus_j))
        use_fast = (k_z * k_xs) <= _MAX_FAST_JOINT_CELLS and (k_z * k_xsj) <= _MAX_FAST_JOINT_CELLS
        z_codes_perm = z_codes.copy()

        for _ in range(n_permutations):
            for grp in strata_groups:
                z_codes_perm[grp] = cpu_rng.permutation(z_codes[grp])

            if use_fast:
                h_z_xs_perm = _fast_joint_entropy(z_codes_perm, k_z, xs_codes, k_xs)
                h_z_xs_plus_j_perm = _fast_joint_entropy(z_codes_perm, k_z, xsj_codes, k_xsj)
            else:
                h_z_xs_perm = h_z_xs_plus_j_perm = None

            if h_z_xs_perm is None:
                h_z_xs_perm = joint_entropy(z_codes_perm, xs_codes)
            if h_z_xs_plus_j_perm is None:
                h_z_xs_plus_j_perm = joint_entropy(z_codes_perm, xsj_codes)

            i_base_perm = max(0.0, float(h_z + h_xs - h_z_xs_perm))
            i_comb_perm = max(0.0, float(h_z + h_xs_plus_j - h_z_xs_plus_j_perm))
            delta_i_perm = max(0.0, float(i_comb_perm - i_base_perm))

            if delta_i_perm >= delta_i_obs:
                count_exceed += 1
    else:
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
