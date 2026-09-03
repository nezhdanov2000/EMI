"""
VSF Metrics: bias-corrected, decomposable information measures.

Rationale (read this before touching `normalized_mutual_information` in
`vsf.math`)
-----------------------------------------------------------------------
`vsf.math.normalized_mutual_information` reports

    NMI_min = I_hat(Z; X_S) / min(H_hat(Z), H_hat(X_S))

where `I_hat` is the *plug-in* (maximum-likelihood) estimator built from the
empirical joint table. The plug-in estimator is positively biased: under exact
statistical independence its expectation is approximately

    E_0[I_hat] ~ (R - 1)(C - 1) / (2 N ln 2)          [Miller & Madow, 1955]

with R = |support(Z)|, C = |support(X_S)|. In this codebase C is allowed to
grow up to `N // 10` (`vsf.pmd.check_grid_capacity`), so the *noise floor* of
a 4-D branch on N = 32 561 rows is ~0.08 bits of mutual information between
variables that are, by construction, independent. That is an order of
magnitude above the "< 0.01 bits = noise" rule of thumb one is tempted to
apply to raw MI, and it is the single dominant artefact in every number this
package currently reports.

Two consequences that changing the NMI denominator does NOT fix:

1.  `vsf.avr.discover_branches` ranks candidate subsets by *raw* `I_hat`.
    Because the bias term grows with C, the argmax is systematically pulled
    toward the highest-cardinality combination available. On a fully
    synthetic dataset in which every feature is independent of the target,
    the current search still returns a 3-D branch with I_hat = 0.078 bits
    and NMI_min = 9.8 %. This is a *selection* defect, not a display defect.

2.  For a micro-class (H(Z) -> 0) the plug-in MI saturates at its ceiling
    min(H(Z), H(X_S)) = H(Z) under pure noise, so NMI_min -> 100 % with no
    signal whatsoever. Replacing `min` with a geometric mean
    sqrt(H(Z) H(X_S)) does not repair this: it merely divides *every* number
    for that target by the same large constant. Empirically, at N = 32 561
    with 7 positives and C = 3 256 cells, NMI_geo reads 1.1 % for pure noise
    and 1.6 % for a *perfect* predictor of the rare class - i.e. the
    geometric normalisation destroys the signal/noise contrast it is
    supposed to expose. The defect is in the estimator, not in the
    denominator.

What this module provides instead
-----------------------------------------------------------------------
*   `mutual_information_bits` - plug-in MI computed directly from the
    contingency table (no `H(Z) + H(X) - H(Z,X)` cancellation).
*   `expected_mutual_information_bits` - the EXACT expectation of the plug-in
    estimator under the permutation (hypergeometric) null, per Vinh, Epps &
    Bailey, JMLR 11 (2010), 2837-2854. No Monte Carlo, no scipy.
*   `adjusted_mutual_information_bits` - I_hat - E_0[I_hat], the correct
    quantity to RANK candidate subsets by, since it is comparable across
    combinations of different cardinality.
*   `adjusted_uncertainty_coefficient` - (I_hat - E_0) / (H(Z) - E_0), the
    bias-corrected fraction of the target's uncertainty that the display
    actually resolves. This is the "share in percent" a user wants: it is 0 %
    for noise and 100 % for a deterministic relation, *including on
    micro-classes*, which no choice of symmetric denominator achieves.
*   `information_report` - the above plus an exact per-class decomposition
    I(Z; X) = sum_z p(z) * D_KL(p(x|z) || p(x)), which answers the distinct
    question "is this global number driven by the class I care about, or
    entirely by the majority class?"

All estimates here are in-sample. A bias-corrected in-sample statistic
establishes that an association exists in the observed table; it does NOT
establish out-of-sample predictability. Any claim of the latter requires a
held-out split, which is deliberately out of scope for this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final, Hashable, Iterable, Iterator, Literal, Sequence

import numpy as np

__all__ = [
    "Candidate",
    "ClassInfo",
    "FamilywiseNull",
    "InfoReport",
    "adjusted_mutual_information_bits",
    "adjusted_uncertainty_coefficient",
    "benjamini_hochberg",
    "branch_score",
    "cell_codes",
    "contingency_from_codes",
    "contingency_table",
    "entropy_bits_from_counts",
    "expected_mutual_information_bits",
    "familywise_max_null",
    "information_report",
    "miller_madow_bias_bits",
    "mutual_information_bits",
    "permutation_null",
    "permutation_pvalue",
    "specific_surprise_bits",
]

_LN2: Final[float] = float(np.log(2.0))
_EPS: Final[float] = 1e-12

NullMethod = Literal["exact", "miller_madow", "none"]


# --------------------------------------------------------------------------
# Label handling
# --------------------------------------------------------------------------
def _as_cell_codes(A: np.ndarray | Sequence[object]) -> np.ndarray:
    """
    Maps an (N,) or (N, d) array of discrete labels onto dense integer codes
    0..C-1 identifying the distinct joint cells, preserving row identity.

    Unlike `vsf.math._flatten_2d_to_1d` this never falls back to a raw 2-D
    array: the caller of every function in this module requires a 1-D code
    vector, so the mixed-radix path is used when it cannot overflow and an
    exact `np.unique(..., axis=0)` otherwise.
    """
    arr = np.asarray(A)
    if arr.ndim == 1:
        _, codes = np.unique(arr, return_inverse=True)
        return codes.astype(np.int64, copy=False).ravel()
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got ndim={arr.ndim}")
    if arr.shape[1] == 1:
        _, codes = np.unique(arr[:, 0], return_inverse=True)
        return codes.astype(np.int64, copy=False).ravel()

    if arr.dtype.kind in ("U", "S", "O", "b"):
        cols = [np.unique(arr[:, j], return_inverse=True)[1] for j in range(arr.shape[1])]
        arr = np.column_stack(cols)

    arr = arr.astype(np.int64, copy=False)
    radices = (arr.max(axis=0) - arr.min(axis=0) + 1).astype(object)
    total = 1
    for r in radices:
        total *= int(r)
    if total < (1 << 62):
        shifted = arr - arr.min(axis=0)
        flat = shifted[:, 0].copy()
        for j in range(1, arr.shape[1]):
            flat = flat * int(radices[j]) + shifted[:, j]
        _, codes = np.unique(flat, return_inverse=True)
    else:  # pragma: no cover - requires >2^62 nominal cells
        _, codes = np.unique(arr, axis=0, return_inverse=True)
    return codes.astype(np.int64, copy=False).ravel()


def cell_codes(X: np.ndarray | Sequence[object]) -> tuple[np.ndarray, int]:
    """
    Dense integer cell codes for `X` plus the number of occupied cells.

    Exposed because the permutation machinery below re-uses one candidate's
    codes across every shuffle of the target: the codes depend only on the
    features, which permutation leaves untouched, so recomputing the
    O(N log N) `np.unique` pass per shuffle would dominate the cost.
    """
    codes = _as_cell_codes(X)
    n_cells = int(codes.max()) + 1 if codes.size > 0 else 0
    return codes, n_cells


def contingency_from_codes(
    z_codes: np.ndarray, x_codes: np.ndarray, n_rows: int, n_cols: int
) -> np.ndarray:
    """Contingency table from pre-computed dense codes (no `np.unique` pass)."""
    flat = np.bincount(z_codes * n_cols + x_codes, minlength=n_rows * n_cols)
    return flat.reshape(n_rows, n_cols).astype(np.int64, copy=False)


def contingency_table(
    Z: np.ndarray | Sequence[object], X: np.ndarray | Sequence[object]
) -> np.ndarray:
    """
    Dense (R, C) contingency table of counts. Rows index the distinct values
    of `Z` in `np.unique` order; columns index the distinct joint cells of
    `X` (which may be 1-D or (N, d)).
    """
    z = _as_cell_codes(Z)
    x = _as_cell_codes(X)
    if z.shape[0] != x.shape[0]:
        raise ValueError(f"sample count mismatch: {z.shape[0]} vs {x.shape[0]}")
    if z.size == 0:
        return np.zeros((0, 0), dtype=np.int64)
    n_rows = int(z.max()) + 1
    n_cols = int(x.max()) + 1
    flat = np.bincount(z * n_cols + x, minlength=n_rows * n_cols)
    return flat.reshape(n_rows, n_cols).astype(np.int64, copy=False)


# --------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------
def entropy_bits_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy in bits of the empirical distribution `counts / sum`."""
    c = np.asarray(counts, dtype=np.float64).ravel()
    total = c.sum()
    if total <= 0.0:
        return 0.0
    p = c[c > 0] / total
    return float(-np.sum(p * np.log2(p)))


def mutual_information_bits(table: np.ndarray) -> float:
    """
    Plug-in mutual information in bits, evaluated directly on the joint table

        I_hat = sum_ij (n_ij / N) log2( N n_ij / (a_i b_j) ).

    Preferred over `H(Z) + H(X) - H(Z, X)`: that form subtracts three O(10)
    quantities to obtain an O(1e-3) result when the target is a micro-class,
    losing ~4 significant digits to cancellation and requiring the
    `max(0.0, ...)` clamp that `vsf.math._mi_with_entropies` applies.
    """
    t = np.asarray(table, dtype=np.float64)
    n = t.sum()
    if n <= 0.0:
        return 0.0
    a = t.sum(axis=1, keepdims=True)
    b = t.sum(axis=0, keepdims=True)
    mask = t > 0
    if not mask.any():
        return 0.0
    nij = t[mask]
    outer = np.broadcast_to(a, t.shape)[mask] * np.broadcast_to(b, t.shape)[mask]
    return float(np.sum((nij / n) * (np.log(n * nij) - np.log(outer))) / _LN2)


def specific_surprise_bits(table: np.ndarray) -> np.ndarray:
    """
    Per-row specific surprise, D_KL( p(x | z) || p(x) ), in bits.

    Exact decomposition (DeWeese & Meister, 1999):
        I(Z; X) = sum_z p(z) * specific_surprise_bits(table)[z]
    Every entry is non-negative, so the vector `p(z) * surprise[z]` splits the
    reported MI into additive, interpretable per-class contributions.
    """
    t = np.asarray(table, dtype=np.float64)
    n = t.sum()
    out = np.zeros(t.shape[0], dtype=np.float64)
    if n <= 0.0:
        return out
    b = t.sum(axis=0)
    q = b / n
    for i in range(t.shape[0]):
        a_i = t[i].sum()
        if a_i <= 0.0:
            continue
        p = t[i] / a_i
        m = (p > 0) & (q > 0)
        out[i] = float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))) / _LN2)
    return out


# --------------------------------------------------------------------------
# Null model
# --------------------------------------------------------------------------
def _log_factorial(n_max: int) -> np.ndarray:
    """log(k!) for k = 0..n_max, by cumulative summation (float64, exact to ~1e-11 rel.)."""
    lf = np.zeros(n_max + 1, dtype=np.float64)
    if n_max >= 1:
        np.cumsum(np.log(np.arange(1, n_max + 1, dtype=np.float64)), out=lf[1:])
    return lf


def _ragged_arange(starts: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenation of `arange(starts[j], starts[j] + lengths[j])` for all j,
    together with the originating index j of each element. Fully vectorised;
    used to enumerate the admissible overlap counts of every column in one shot.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    idx = np.repeat(np.arange(lengths.shape[0], dtype=np.int64), lengths)
    base = np.repeat(np.cumsum(lengths) - lengths, lengths)
    offs = np.arange(total, dtype=np.int64) - base
    return idx, np.repeat(np.asarray(starts, dtype=np.int64), lengths) + offs


def expected_mutual_information_bits(table: np.ndarray) -> float:
    """
    E_0[I_hat] in bits: the exact expectation of the plug-in MI over all
    permutations of Z that preserve both margins of `table` (the
    hypergeometric null of Vinh, Epps & Bailey, JMLR 2010, eq. 24a).

    This is the correct noise floor for `mutual_information_bits`, and the
    quantity every reported MI in this package must be measured against.

    Complexity: O( sum_ij min(a_i, b_j) ) arithmetic in vectorised numpy, with
    one Python-level iteration per row of the table (i.e. per target class).
    Measured: 2.1 ms median for a 2 x 1379 table at N = 32561, linear in R
    (16 ms at R = 20).
    """
    t = np.asarray(table, dtype=np.int64)
    n = int(t.sum())
    if n <= 1 or t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    a = t.sum(axis=1)
    b = t.sum(axis=0)
    b = b[b > 0]
    a = a[a > 0]
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0

    lf = _log_factorial(n)
    log_n = float(np.log(n))
    const_n = lf[n]
    emi = 0.0
    for a_i in a.tolist():
        starts = np.maximum(1, a_i + b - n)
        stops = np.minimum(a_i, b)
        lens = np.maximum(0, stops - starts + 1)
        j_idx, nij = _ragged_arange(starts, lens)
        if nij.size == 0:
            continue
        b_j = b[j_idx]
        weight = (nij / n) * (log_n + np.log(nij.astype(np.float64))
                              - np.log(float(a_i)) - np.log(b_j.astype(np.float64))) / _LN2
        log_p = (
            lf[a_i] + lf[b_j] + lf[n - a_i] + lf[n - b_j]
            - const_n - lf[nij] - lf[a_i - nij] - lf[b_j - nij] - lf[n - a_i - b_j + nij]
        )
        emi += float(np.sum(weight * np.exp(log_p)))
    return max(0.0, emi)


def miller_madow_bias_bits(n_rows: int, n_cols: int, n_samples: int) -> float:
    """
    Closed-form Miller-Madow approximation to E_0[I_hat],
    (R - 1)(C - 1) / (2 N ln 2).

    Use only where `expected_mutual_information_bits` is too slow to call in a
    tight loop (it over-estimates the floor when many cells are empty, because
    it counts nominal rather than occupied cells). For any number that is
    *reported*, use the exact expectation.
    """
    if n_samples <= 0:
        return 0.0
    return float((n_rows - 1) * (n_cols - 1) / (2.0 * n_samples * _LN2))


def permutation_null(
    Z: np.ndarray | Sequence[object],
    X: np.ndarray | Sequence[object],
    n_permutations: int = 999,
    random_state: int | None = 0,
) -> tuple[float, float, float]:
    """
    Convenience wrapper over `permutation_pvalue` taking raw label arrays
    instead of dense codes. Returns (null_mean, null_std, p_value).
    """
    z, n_rows = cell_codes(Z)
    x, n_cols = cell_codes(X)
    if z.shape[0] != x.shape[0]:
        raise ValueError(f"sample count mismatch: {z.shape[0]} vs {x.shape[0]}")
    p, mean, std = permutation_pvalue(z, x, n_rows, n_cols, n_permutations, random_state)
    return mean, std, p


# --------------------------------------------------------------------------
# Corrected metrics
# --------------------------------------------------------------------------
def adjusted_mutual_information_bits(table: np.ndarray, null: NullMethod = "exact") -> float:
    """
    I_hat - E_0[I_hat], in bits: the excess association over what an
    independent variable of the same margins would produce by chance.

    This, not raw `I_hat`, is the quantity that is comparable ACROSS candidate
    feature subsets of differing cardinality, and therefore the one
    `vsf.avr.discover_branches` must maximise.
    """
    mi = mutual_information_bits(table)
    if null == "none":
        return mi
    t = np.asarray(table)
    if null == "miller_madow":
        floor = miller_madow_bias_bits(t.shape[0], t.shape[1], int(t.sum()))
    elif null == "exact":
        floor = expected_mutual_information_bits(t)
    else:
        raise ValueError(f"unknown null method: {null!r}")
    return mi - floor


def adjusted_uncertainty_coefficient(table: np.ndarray, null: NullMethod = "exact") -> float:
    """
    Bias-corrected fraction of the TARGET's uncertainty resolved by X:

        U_adj = (I_hat - E_0[I_hat]) / (H(Z) - E_0[I_hat])

    Properties, all verified in `tests/test_metrics.py`:
      * 0 in expectation when Z is independent of X, at any cardinality and
        any class balance - including a 7-in-32 561 micro-class, where the
        uncorrected `MI / min(H(Z), H(X))` reads 66 % (U_adj itself reads
        ~2 % there rather than exactly 0 - the adjusted denominator
        H(Z) - E_0 is tiny for a 7-member class, so a micro-class is settled
        by the p-value, not by the effect size; on a balanced target the same
        measurement gives 0.06 %);
      * 1 exactly when Z is a deterministic function of X, again including
        the micro-class, where `MI / sqrt(H(Z) H(X))` reads 1.6 % and is
        therefore unusable;
      * directional: it normalises by H(Z), not by a symmetric average, so it
        answers "what share of the target did we explain", which is the
        question a percentage is being asked for in the first place.

    The uncorrected version of this quantity is Theil's U / the uncertainty
    coefficient; the correction is the Vinh et al. adjustment applied to a
    directional denominator rather than a symmetric one.
    """
    t = np.asarray(table, dtype=np.int64)
    h_z = entropy_bits_from_counts(t.sum(axis=1))
    mi = mutual_information_bits(t)
    if null == "none":
        floor = 0.0
    elif null == "miller_madow":
        floor = miller_madow_bias_bits(t.shape[0], t.shape[1], int(t.sum()))
    elif null == "exact":
        floor = expected_mutual_information_bits(t)
    else:
        raise ValueError(f"unknown null method: {null!r}")
    denom = h_z - floor
    if denom <= _EPS:
        return 0.0
    return float(np.clip((mi - floor) / denom, 0.0, 1.0))


def branch_score(
    Z: np.ndarray | Sequence[object],
    X_S: np.ndarray | Sequence[object],
    null: NullMethod = "miller_madow",
) -> float:
    """
    Ranking score for `vsf.avr.discover_branches`'s exhaustive search loop.

    Drop-in replacement for the current `mutual_information(Z_discrete, X_S)`
    call. Defaults to the closed-form floor because it is evaluated
    C(M, 1) + ... + C(M, 4) times; switch to `null="exact"` for the final
    re-scoring of the winning combination.
    """
    return adjusted_mutual_information_bits(contingency_table(Z, X_S), null=null)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassInfo:
    """Per-target-class decomposition of a single `InfoReport`."""

    label_index: int
    count: int
    prevalence: float
    surprise_bits: float
    contribution_bits: float
    contribution_share: float
    u_adj_one_vs_rest: float


@dataclass(frozen=True)
class InfoReport:
    """
    Complete information-theoretic description of one (target, feature-subset)
    pair. `mi_bits` is directly comparable to `vsf.math.mutual_information`;
    `nmi_min` and `nmi_geo` are reproduced only so that a reader can see how
    far the uncorrected numbers are from `u_adj`.
    """

    n_samples: int
    n_classes: int
    n_cells: int
    samples_per_cell: float
    h_target_bits: float
    h_cells_bits: float
    mi_bits: float
    mi_null_bits: float
    mi_adj_bits: float
    u_raw: float
    u_adj: float
    nmi_min: float
    nmi_geo: float
    p_value: float | None
    per_class: tuple[ClassInfo, ...]

    def is_signal(self, alpha: float = 0.01, min_u_adj: float = 0.02) -> bool:
        """Conservative gate: significant under the permutation null AND non-trivial in size."""
        if self.p_value is not None and self.p_value > alpha:
            return False
        return self.u_adj >= min_u_adj


def information_report(
    Z: np.ndarray | Sequence[object],
    X_S: np.ndarray | Sequence[object],
    null: NullMethod = "exact",
    n_permutations: int = 0,
    random_state: int | None = 0,
) -> InfoReport:
    """
    Computes every metric this module exposes for one (Z, X_S) pair.

    Set `n_permutations > 0` to attach a Monte-Carlo p-value (adds
    `n_permutations` contingency-table builds; 999 is the usual choice for a
    0.001-resolution p-value). The point estimates themselves never need it -
    `null="exact"` corrects the bias analytically.
    """
    table = contingency_table(Z, X_S)
    n = int(table.sum())
    row_counts = table.sum(axis=1)
    col_counts = table.sum(axis=0)
    n_classes = int((row_counts > 0).sum())
    n_cells = int((col_counts > 0).sum())

    h_z = entropy_bits_from_counts(row_counts)
    h_x = entropy_bits_from_counts(col_counts)
    mi = mutual_information_bits(table)

    if null == "none":
        floor = 0.0
    elif null == "miller_madow":
        floor = miller_madow_bias_bits(table.shape[0], table.shape[1], n)
    elif null == "exact":
        floor = expected_mutual_information_bits(table)
    else:
        raise ValueError(f"unknown null method: {null!r}")

    denom_adj = h_z - floor
    u_adj = float(np.clip((mi - floor) / denom_adj, 0.0, 1.0)) if denom_adj > _EPS else 0.0
    u_raw = float(mi / h_z) if h_z > _EPS else 0.0
    nmi_min = float(min(1.0, mi / min(h_z, h_x))) if min(h_z, h_x) > _EPS else 0.0
    nmi_geo = float(mi / np.sqrt(h_z * h_x)) if h_z > _EPS and h_x > _EPS else 0.0

    surprise = specific_surprise_bits(table)
    priors = row_counts / n if n > 0 else np.zeros_like(row_counts, dtype=np.float64)
    contributions = priors * surprise

    per_class: list[ClassInfo] = []
    for i in range(table.shape[0]):
        if row_counts[i] == 0:
            continue
        ovr = np.vstack([table.sum(axis=0) - table[i], table[i]])
        per_class.append(
            ClassInfo(
                label_index=i,
                count=int(row_counts[i]),
                prevalence=float(priors[i]),
                surprise_bits=float(surprise[i]),
                contribution_bits=float(contributions[i]),
                contribution_share=float(contributions[i] / mi) if mi > _EPS else 0.0,
                u_adj_one_vs_rest=adjusted_uncertainty_coefficient(ovr, null=null),
            )
        )

    p_value: float | None = None
    if n_permutations > 0:
        _, _, p_value = permutation_null(Z, X_S, n_permutations, random_state)

    return InfoReport(
        n_samples=n,
        n_classes=n_classes,
        n_cells=n_cells,
        samples_per_cell=float(n / n_cells) if n_cells > 0 else 0.0,
        h_target_bits=h_z,
        h_cells_bits=h_x,
        mi_bits=mi,
        mi_null_bits=floor,
        mi_adj_bits=mi - floor,
        u_raw=u_raw,
        u_adj=u_adj,
        nmi_min=nmi_min,
        nmi_geo=nmi_geo,
        p_value=p_value,
        per_class=tuple(per_class),
    )


# --------------------------------------------------------------------------
# Batched permutation machinery
# --------------------------------------------------------------------------
_CHUNK_BYTES: Final[int] = 64 * 1024 * 1024


def _mi_bits_batch(tables: np.ndarray) -> np.ndarray:
    """
    Plug-in MI in bits for a stack of contingency tables, shape (B, R, C).

    Same estimator as `mutual_information_bits`, vectorised over the leading
    axis so that B permutation replicates cost one pass instead of B.
    """
    t = np.asarray(tables, dtype=np.float64)
    n = t.sum(axis=(1, 2))
    a = t.sum(axis=2)[:, :, None]
    b = t.sum(axis=1)[:, None, :]
    mask = t > 0
    safe_t = np.where(mask, t, 1.0)
    safe_outer = np.where(mask, a * b, 1.0)
    log_n = np.log(np.where(n > 0, n, 1.0))[:, None, None]
    term = np.where(mask, safe_t * (np.log(safe_t) + log_n - np.log(safe_outer)), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = term.sum(axis=(1, 2)) / (n * _LN2)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _permuted_scores(
    z_codes: np.ndarray,
    x_codes: np.ndarray,
    n_rows: int,
    n_cols: int,
    perms: np.ndarray,
) -> np.ndarray:
    """
    Plug-in MI of `x_codes` against every row of `z_codes[perms]`.

    `perms` is a (B, N) index matrix. The B contingency tables are built with
    a SINGLE `np.bincount` over an offset-encoded (replicate, class, cell)
    index, which is what makes a 999-replicate null affordable inside an
    exhaustive C(M, d) search.
    """
    b_size = perms.shape[0]
    if b_size == 0:
        return np.empty(0, dtype=np.float64)
    cells = n_rows * n_cols
    zp = z_codes[perms]
    offsets = (np.arange(b_size, dtype=np.int64) * cells)[:, None]
    flat = (offsets + zp * n_cols + x_codes[None, :]).ravel()
    counts = np.bincount(flat, minlength=b_size * cells)
    return _mi_bits_batch(counts.reshape(b_size, n_rows, n_cols))


def _permutation_chunks(
    n_samples: int, n_permutations: int, table_cells: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """
    Yields (b, N) index matrices whose row-blocks partition `n_permutations`,
    sized so that the peak working set of `_permuted_scores` stays under
    `_CHUNK_BYTES`. Consuming the generator once produces exactly
    `n_permutations` distinct shuffles.

    The 32 bytes/sample/replicate accounts for the four (b, N) temporaries
    that are simultaneously live: the uniform draw, its argsort, the gathered
    target codes, and the offset-encoded bincount index.
    """
    per_row = max(1, n_samples * 32 + max(0, table_cells) * 8)
    block = int(np.clip(_CHUNK_BYTES // per_row, 1, max(1, n_permutations)))
    done = 0
    while done < n_permutations:
        size = min(block, n_permutations - done)
        yield np.argsort(rng.random((size, n_samples)), axis=1)
        done += size


@dataclass(frozen=True)
class FamilywiseNull:
    """
    Distribution of the MAXIMUM adjusted-MI score attained across an entire
    candidate family under the permutation null.

    `max_scores` has one entry per replicate: the best score any candidate in
    the family achieved on that shuffled target. Comparing an observed winner
    against this distribution is the correction for the look-elsewhere effect
    of scanning C(M,1) + ... + C(M,d_max) subsets - an uncorrected per-branch
    p-value does not account for the fact that the branch was CHOSEN as the
    maximum of that scan.
    """

    max_scores: np.ndarray
    n_permutations: int
    n_candidates: int

    def p_value(self, observed: float) -> float:
        """Familywise-corrected p-value, floored at 1 / (B + 1)."""
        if self.n_permutations <= 0:
            return 1.0
        exceed = int(np.sum(self.max_scores >= observed - _EPS))
        return float((1 + exceed) / (self.n_permutations + 1))


Candidate = tuple[Hashable, np.ndarray, int]
"""(key, dense cell codes, n_cells) describing one member of a search family."""


def _sample_null_mi(
    row_counts: np.ndarray,
    col_counts: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
    block: int = 256,
) -> np.ndarray:
    """
    Draws `n_replicates` contingency tables from the permutation null of a
    table with the given margins, and returns their plug-in MI in bits.

    Conditional on both margins, the permutation null of a contingency table
    IS the multiple hypergeometric distribution (this is the null underlying
    Fisher's exact test and the Vinh et al. expectation used elsewhere in this
    module). Sampling it directly is exact, not an approximation of
    permutation, and never touches the N samples: the cost is O(R * C) per
    replicate against O(N) for materialising an explicit shuffle.

    That is a win only while R * C stays well below N, because both paths then
    pay for the MI evaluation itself. Measured at N = 32 561, R = 2, B = 999,
    against the explicit-shuffle path:

        C =   20   80x faster
        C =  100   25x
        C =  500  5.4x
        C = 1379  2.1x
        C = 3256  0.9x   <- slower

    `permutation_pvalue` therefore dispatches on `5 * R * C <= N`, which is
    where the measured crossover sits. Reported here rather than smoothed over
    because the Grid Capacity Limit explicitly allows C up to N / 10, so the
    unfavourable regime is reachable in normal use, not hypothetical.

    For R = 2 the whole replicate set is drawn in one vectorised call. For
    R > 2 the rows are drawn sequentially from the residual column supply,
    which is the standard sequential construction of the same joint law.
    """
    rows = np.asarray(row_counts, dtype=np.int64).ravel()
    cols = np.asarray(col_counts, dtype=np.int64).ravel()
    rows = rows[rows > 0]
    n_rows, n_cols = rows.shape[0], cols.shape[0]
    if n_replicates <= 0 or n_rows < 2 or n_cols < 2:
        return np.zeros(max(0, n_replicates), dtype=np.float64)

    out = np.empty(n_replicates, dtype=np.float64)
    done = 0
    while done < n_replicates:
        size = int(min(block, n_replicates - done))
        tables = np.empty((size, n_rows, n_cols), dtype=np.int64)
        if n_rows == 2:
            tables[:, 0, :] = rng.multivariate_hypergeometric(cols, int(rows[0]), size=size)
            tables[:, 1, :] = cols[None, :] - tables[:, 0, :]
        else:
            for r in range(size):
                remaining = cols.copy()
                for i in range(n_rows - 1):
                    draw = rng.multivariate_hypergeometric(remaining, int(rows[i]))
                    tables[r, i, :] = draw
                    remaining -= draw
                tables[r, n_rows - 1, :] = remaining
        out[done : done + size] = _mi_bits_batch(tables)
        done += size
    return out


def _shuffled_null_mi(
    z_codes: np.ndarray,
    x_codes: np.ndarray,
    n_rows: int,
    n_cols: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permutation null by explicit relabelling: one shuffle of the N target
    codes per replicate. Distributionally identical to `_sample_null_mi` (both
    condition on the same two margins); used when the table is dense enough
    that drawing it directly is no longer the cheaper route - see
    `_sample_null_mi`'s measured crossover.
    """
    out = np.empty(max(0, n_replicates), dtype=np.float64)
    for k in range(n_replicates):
        out[k] = mutual_information_bits(
            contingency_from_codes(rng.permutation(z_codes), x_codes, n_rows, n_cols)
        )
    return out


def permutation_pvalue(
    z_codes: np.ndarray,
    x_codes: np.ndarray,
    n_rows: int,
    n_cols: int,
    n_permutations: int = 999,
    random_state: int | None = 0,
) -> tuple[float, float, float]:
    """
    Uncorrected permutation p-value for a SINGLE, pre-specified (Z, X_S) pair,
    returning (p_value, null_mean, null_std).

    Because the plug-in MI's null mean is available in closed form
    (`expected_mutual_information_bits`), this exists for the p-value and the
    null spread only. It does NOT correct for the fact that a branch reported
    by `vsf.avr.discover_branches` was selected as the argmax of a scan - use
    `familywise_max_null` for that.

    The p-value's floor is 1 / (B + 1), so B must exceed 1/alpha - 1: at the
    default B = 999 no result can be reported below p = 0.001, and alpha
    thresholds tighter than that are not resolvable without raising B.
    """
    table = contingency_from_codes(z_codes, x_codes, n_rows, n_cols)
    observed = mutual_information_bits(table)
    if n_permutations <= 0:
        return 1.0, 0.0, 0.0
    rng = np.random.default_rng(random_state)
    n_samples = int(z_codes.shape[0])
    if 5 * n_rows * n_cols <= n_samples:
        null = _sample_null_mi(table.sum(axis=1), table.sum(axis=0), n_permutations, rng)
    else:
        null = _shuffled_null_mi(z_codes, x_codes, n_rows, n_cols, n_permutations, rng)
    exceed = int(np.sum(null >= observed - _EPS))
    p = float((1 + exceed) / (n_permutations + 1))
    std = float(null.std(ddof=1)) if null.size > 1 else 0.0
    return p, float(null.mean()), std


def familywise_max_null(
    z_codes: np.ndarray,
    candidates: Callable[[], Iterable[Candidate]],
    n_permutations: int = 999,
    random_state: int | None = 0,
    null: NullMethod = "exact",
    progress: Callable[[int, int], None] | None = None,
) -> FamilywiseNull:
    """
    Permutation distribution of max_S (I_hat(Z_pi; X_S) - E_0[I_hat]) over the
    whole candidate family, i.e. the null of the statistic that
    `discover_branches` actually reports.

    `candidates` must be a CALLABLE returning a fresh iterable of
    `(key, cell_codes, n_cells)` triples - it is re-invoked once per
    permutation chunk so that the caller may regenerate codes lazily instead
    of holding every candidate's (N,) code vector in memory at once.

    The per-candidate bias floor E_0 is a function of the two margins alone,
    and permutation preserves both, so it is computed ONCE per candidate and
    cached across all replicates. That is what keeps an exact-null-corrected
    familywise test affordable.

    Cost: O(B * sum_S N) table builds. On the reference 32 561-row, M = 7
    dataset (98 candidates, B = 999) this is a few seconds; it grows linearly
    in the number of candidates, so for M >~ 20 the caller should either lower
    B or run this off the interactive path.
    """
    n_samples = int(z_codes.shape[0])
    n_rows = int(z_codes.max()) + 1 if z_codes.size > 0 else 0
    if n_permutations <= 0 or n_rows < 2:
        return FamilywiseNull(np.empty(0, dtype=np.float64), 0, 0)

    floors: dict[Hashable, float] = {}
    max_scores = np.full(n_permutations, -np.inf, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    n_candidates = 0
    done = 0
    # Conservative cell-count hint: the Grid Capacity Limit
    # (`vsf.pmd.check_grid_capacity`) caps any candidate at N // 10 columns.
    cells_hint = n_rows * max(1, n_samples // 10)
    for perms in _permutation_chunks(n_samples, n_permutations, cells_hint, rng):
        size = perms.shape[0]
        sl = slice(done, done + size)
        n_candidates = 0
        for key, x_codes, n_cols in candidates():
            n_candidates += 1
            if key not in floors:
                table = contingency_from_codes(z_codes, x_codes, n_rows, n_cols)
                if null == "none":
                    floors[key] = 0.0
                elif null == "miller_madow":
                    floors[key] = miller_madow_bias_bits(n_rows, n_cols, n_samples)
                elif null == "exact":
                    floors[key] = expected_mutual_information_bits(table)
                else:
                    raise ValueError(f"unknown null method: {null!r}")
            scores = _permuted_scores(z_codes, x_codes, n_rows, n_cols, perms) - floors[key]
            np.maximum(max_scores[sl], scores, out=max_scores[sl])
        done += size
        if progress is not None:
            progress(done, n_permutations)
    return FamilywiseNull(max_scores, n_permutations, n_candidates)


# --------------------------------------------------------------------------
# Multiplicity control across a family of independent tests
# --------------------------------------------------------------------------
def benjamini_hochberg(p_values: Sequence[float] | np.ndarray, q: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg step-up procedure at false-discovery rate `q`.

    Returns a boolean mask of the hypotheses to reject. Used by the Global
    Pattern Scan, where the family is the set of (column, value) targets
    swept in one run: without it, sweeping k targets at a nominal 0.05 yields
    ~0.05k spurious "patterns" by construction.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = p.shape[0]
    reject = np.zeros(m, dtype=bool)
    if m == 0:
        return reject
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")
    order = np.argsort(p, kind="stable")
    thresholds = q * np.arange(1, m + 1, dtype=np.float64) / m
    passing = np.nonzero(p[order] <= thresholds)[0]
    if passing.size > 0:
        reject[order[: int(passing[-1]) + 1]] = True
    return reject
