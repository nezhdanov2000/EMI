"""
VSF Independent Branch Discovery (IBD) - v2.1 "Corrected Core"

Replaces the v1.0 Adaptive Visual Routing (AVR) engine (greedy forward
selection + permutation-test stopping + Benjamini-Hochberg FDR control +
four rendering scenarios A/B/C/D). See Project_Master_Document.md Section 0
for the full revision history and Sections 4-4.6 for the formal spec this
module implements.

Core idea: for a chosen target Z, independently find, for every
dimensionality d in {1, 2, 3, 4}, the single best-scoring feature subset of
exactly that size - by HONEST, EXHAUSTIVE enumeration of every
C(M, d) combination. The four resulting "branches" are NOT required to be
nested: the winning pair for d=2 need not contain either feature from the
winning singleton at d=1. This is intentional, not a bug - mutual information
is not submodular in general (Krause & Guestrin, 2005), so a greedy nested
chain (the v1.0 approach) can provably miss a synergistic combination that an
independent per-d search finds. See `discover_branches`'s docstring for the
formal statement.

WHAT CHANGED IN v2.1 (and why v2.0's ranking was wrong)
-------------------------------------------------------
v2.0 ranked candidates by RAW plug-in mutual information and documented the
absence of a degrees-of-freedom correction as an accepted product trade-off
(v2.0 module docstring; Project_Master_Document.md Section 4.5). That
trade-off was not survivable. The plug-in estimator's bias under exact
independence is ~(R-1)(C-1) / (2 N ln 2), which grows with the candidate's
cell count C, so `argmax` over raw MI is pulled toward whichever combination
has the most cells - independently of any signal.

Measured on a fully synthetic control in which EVERY feature is generated
independently of the target (N = 32 561; feature cardinalities 2, 3, 5, 10,
50, 200), v2.0's search returned:

    d=1  ['k=200']                       MI = 0.0047   NMI_min =  0.6 %
    d=2  ['k=50', 'k=200']               MI = 0.0677   NMI_min =  8.6 %
    d=3  ['k=3', 'k=5', 'k=200']         MI = 0.0778   NMI_min =  9.8 %

i.e. it selected the highest-cardinality feature at d=1 and reported a
"moderate association" where the true mutual information is exactly zero.
The same control under v2.1 returns MI_adj <= 0.004 bits and U_adj <= 0.5 %
at every d, with no branch significant at alpha = 0.01.

v2.1 therefore ranks by

    MI_adj(S) = I_hat(Z; X_S) - E_0[I_hat(Z; X_S)]

where E_0 is the EXACT expectation of the plug-in estimator over all
permutations of Z that preserve both margins (Vinh, Epps & Bailey, JMLR 11
(2010), 2837-2854), computed by `vsf.metrics.expected_mutual_information_bits`.
E_0 depends only on the two margins, so it is a per-candidate constant that
makes candidates of unequal cardinality directly comparable - which raw MI is
not.

WHAT CHANGED IN v2.2 (the reporting objective)
----------------------------------------------
v2.1 fixed the RANKING statistic and left the REPORTED one wrong for the
product's actual question. MI_adj / U_adj measure whether an association
exists between the target and the cell partition. The display's deliverable
is narrower: a small number of cells that are almost purely the target
value. On a rare target the two answers diverge completely - see
`vsf.centers`' module docstring for the fully reproducible Armed-Forces case
where U_adj reads 41.3 % while the highest cell purity in the entire branch
is 2.42 % and the Bayes rule under 0-1 loss never predicts the class.

v2.2 therefore adds, to every branch, a `CenterReport` (`vsf.centers`):
certified centre count K, coverage (the share of target-value samples inside
certified centres), pooled purity, and their out-of-sample and permutation
counterparts. Nothing about MI_adj ranking changed by default; U_adj is
demoted from headline to diagnostic, and `objective="coverage"` makes the
search optimise the quantity the product reports instead of a correlate of
it. That option is not the default because switching it silently would
invalidate every measured claim in Sections 4.3-4.6 of
Project_Master_Document.md; it is the setting a paper reporting coverage
must use, and the mismatch is stated rather than hidden.

WHAT IS STILL NOT CLAIMED
-------------------------
* Every estimate here is IN-SAMPLE. A significant MI_adj establishes that an
  association exists in the observed table. It does not establish
  out-of-sample predictability; that needs a held-out split, which this module
  deliberately does not perform.
* `p_value` is the uncorrected permutation p-value of one pre-specified
  subset. It is NOT valid for the branch this module returns, because that
  branch was chosen as the maximum of a scan over
  C(M,1) + ... + C(M,d_max) candidates. The look-elsewhere-corrected number is
  `p_value_familywise`, which requires `n_permutations_familywise > 0` and
  costs B times a full search; it is off by default because it is not
  affordable on an interactive path, and any published claim must set it.
* No FDR control is applied WITHIN a single search. It is applied across
  targets by the Global Pattern Scan (`vsf.server`), which is the family in
  which multiplicity actually accumulates for that feature.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .centers import (
    CenterReport,
    CenterSpec,
    CenterRule,
    CVCoverage,
    binarize_target,
    center_report,
    coverage_score,
    familywise_max_coverage_null,
)
from .centers import select_dimensionality as _select_dimensionality
from .metrics import (
    ClassInfo,
    NullMethod,
    cell_codes,
    contingency_from_codes,
    entropy_bits_from_counts,
    expected_mutual_information_bits,
    familywise_max_null,
    information_report,
    miller_madow_bias_bits,
    mutual_information_bits,
    permutation_pvalue,
)
from .pmd import adaptively_coarsen_bins, check_grid_capacity, discretize_dataset

# Hard ceiling on branch dimensionality. Matches the display's actual spatial
# encoding (3 coordinate axes + 1 time/frame axis, see vsf.vis's module
# docstring and Project_Master_Document.md Section 1.3) - there is no 5th/6th
# channel to search for, unlike v1.0's d_max=7 (which searched further than
# the display could ever show).
MAX_BRANCH_D = 4

#: Ranking objective for the exhaustive search.
#:
#: `"auto"` (the default) is `"coverage"` whenever a positive target value is
#: resolvable, and `"mi_adj"` otherwise. It is the default because the
#: product's stated goal is a small set of cells that capture as much of one
#: target value as possible, and that is what `"coverage"` maximises;
#: `"mi_adj"` is a correlate of it, and on a rare value not even a good one.
#: A K-valued target with no declared positive value has no purity and hence
#: no coverage, so the fallback is not a preference but the only defined
#: option.
#:
#: `"mi_adj"` maximises I_hat - E_0[I_hat] (v2.1 behaviour). `"coverage"`
#: maximises the centre coverage the display reports; see
#: `vsf.centers.coverage_score` for the tie-break order, which encodes the
#: secondary product rule "fewer, larger centres at equal coverage". The two
#: are NOT interchangeable: MI_adj rewards association anywhere in the table,
#: including a cell that is purely the NEGATIVE class, which contributes
#: nothing to a target-value centre.
Objective = Literal["auto", "mi_adj", "coverage"]

#: Default number of permutation replicates for a single branch's p-value.
#: 999 gives a resolution of 1e-3, the coarsest that can still express
#: alpha = 0.01 with a margin.
DEFAULT_N_PERMUTATIONS = 999


@dataclass
class BranchResult:
    """
    One independently-discovered feature subset for a single dimensionality
    `d`.

    Field semantics
    ---------------
    `mi`
        Raw plug-in I_hat(Z; X_S) in bits. Retained as a diagnostic and for
        reproducibility of the v2.0 -> v2.1 comparison; it is NOT comparable
        across branches of different `d`, because its bias grows with cell
        count.
    `mi_null`
        E_0[I_hat] in bits for THIS branch's margins - the value the raw `mi`
        would take if Z and X_S were independent. Read `mi` against it: a
        branch with mi = 0.078 and mi_null = 0.074 carries no association at
        all, however impressive 0.078 looks on an absolute scale.
    `mi_adj`
        `mi - mi_null`. The ranking statistic. Comparable across
        cardinalities and across `d`.
    `u_adj`
        `mi_adj / (H(Z) - mi_null)`, clipped to [0, 1]. The reportable
        percentage: the bias-corrected share of the TARGET's uncertainty that
        this branch's axes resolve. Directional by construction (normalized by
        H(Z), not by a symmetric average of H(Z) and H(X_S)), because the
        question a percentage is asked to answer here is "how much of the
        target did we explain", not "how similar are these two partitions".
    `h_target`
        H(Z) in bits. Carried so that a consumer can recover every other
        quantity without re-reading the dataset.
    `p_value`
        Uncorrected permutation p-value for this subset alone, or None when
        `n_permutations <= 0`. See this module's docstring for why it must not
        be quoted as the significance of a DISCOVERED branch.
    `p_value_familywise`
        Permutation p-value against the distribution of the maximum MI_adj
        over the entire C(M,1..d_max) candidate family. This is the valid
        number for a discovered branch. None unless
        `n_permutations_familywise > 0`.
    `per_class`
        Exact additive decomposition of `mi` over the target's classes,
        I(Z; X) = sum_z p(z) D_KL(p(x|z) || p(x)). Answers the question no
        scalar can: whether a headline number is carried by the class the user
        cares about or entirely by the majority class. For a 0.1 %-prevalence
        class, `contribution_share` is typically < 1 % however large the
        headline is.
    `centers`
        The v2.2 reporting layer (`vsf.centers.CenterReport`) for THIS
        branch's full cell partition: how many cells are certified to be at
        least `tau` pure in the target value, what share of all target-value
        samples they contain (`coverage`), how clean they are, and what
        survives cross-validation and the permutation null. `None` when the
        target is not binary and the caller named no `positive_class` - a
        purity has no meaning without a designated positive value, and
        guessing one (as v2.1's renderer did, by taking the highest class
        index) silently reports a different quantity than the panel claims.
    `coverage_by_prefix_d` / `n_centers_by_prefix_d` / `purity_by_prefix_d`
        The centre statistics under this branch's own first-k axes, indexed
        as the `*_by_prefix_d` MI series above and subject to the same
        warning: the frontend's within-branch collapse must read the entry
        for the VIEWED dimensionality, not the full-branch scalar. In-sample
        only (no cross-validation per prefix - that is 25 refits per prefix
        per branch and does not belong on an interactive path); use
        `select_branch_dimensionality` for the out-of-sample answer.
    `coverage_p_value_familywise`
        Look-elsewhere-corrected p-value of `centers.coverage` against
        `vsf.centers.familywise_max_coverage_null`. This, not
        `centers.coverage_p_value`, is the number a paper quotes for a
        DISCOVERED branch, for exactly the reason `p_value_familywise`
        exists for MI_adj.
    `mi_by_prefix_d` / `mi_adj_by_prefix_d` / `u_adj_by_prefix_d`
        NOT a second independent search - they answer a different question
        from `discover_branches`'s own per-d optimization.
        `discover_branches` finds, for every d, the BEST-scoring subset of
        that size, independently (branches need not nest - see the module
        docstring). These lists instead fix THIS branch's own
        `selected_features` ordering and report the metric for
        I(Z; X_{S[:k]}) - i.e. what this SAME branch would read if only its
        first k axes were displayed. This is the quantity the frontend's
        within-branch dimensionality collapse/split
        (Project_Master_Document.md Section 5.6, case 2) must show; showing
        the fixed full-branch value instead while the view is collapsed to
        k < d axes silently overstates the association carried by those k axes
        alone. The final entry always equals (`mi`, `mi_adj`, `u_adj`)
        exactly, since all are computed by the same `_prefix_metric_series`
        call. Empty for a `BranchResult` built by hand rather than through
        `discover_branches` (e.g. in tests); callers must treat that as "no
        per-view breakdown available" and fall back to the scalars.
    """

    d: int
    selected_features: List[int]
    selected_feature_names: List[str]
    mi: float
    mi_null: float
    mi_adj: float
    u_adj: float
    h_target: float
    mi_by_prefix_d: List[float] = field(default_factory=list)
    mi_adj_by_prefix_d: List[float] = field(default_factory=list)
    u_adj_by_prefix_d: List[float] = field(default_factory=list)
    p_value: Optional[float] = None
    p_value_familywise: Optional[float] = None
    per_class: Tuple[ClassInfo, ...] = ()
    centers: Optional[CenterReport] = None
    coverage_by_prefix_d: List[float] = field(default_factory=list)
    n_centers_by_prefix_d: List[int] = field(default_factory=list)
    purity_by_prefix_d: List[float] = field(default_factory=list)
    coverage_p_value_familywise: Optional[float] = None

    def is_significant(self, alpha: float = 0.01, min_u_adj: float = 0.02) -> bool:
        """
        Conservative gate for "this branch is worth showing as a finding".

        Prefers `p_value_familywise` when it was computed, because that is the
        only p-value valid for a branch that was SELECTED as an argmax. Falls
        back to the uncorrected `p_value`, which is anti-conservative here; a
        branch with no p-value at all is judged on effect size alone and can
        only ever be a weak claim.
        """
        p = self.p_value_familywise if self.p_value_familywise is not None else self.p_value
        if p is not None and p > alpha:
            return False
        return self.u_adj >= min_u_adj

    def has_certified_centers(self, alpha: float = 0.01) -> bool:
        """
        Gate for "this branch produces the deliverable the display promises".

        Deliberately NOT the same test as `is_significant`: a branch can be
        overwhelmingly significant as an association and still certify no
        centre at all (`data/adult_census.csv`, target income = ">50K",
        3-D branch `education + occupation + relationship`: U_adj = 34.0 %
        at p = 0.001, zero certified centres, highest per-cell lower bound
        0.804 against tau = 0.90). Reporting
        such a branch as a finding is the failure mode v2.2 exists to stop.
        """
        if self.centers is None or self.centers.n_centers == 0:
            return False
        p = (
            self.coverage_p_value_familywise
            if self.coverage_p_value_familywise is not None
            else self.centers.coverage_p_value
        )
        return p is None or p <= alpha


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


def _subset_codes(
    X_discrete: np.ndarray,
    bin_counts: Sequence[int],
    n_samples: int,
    combo: Tuple[int, ...],
) -> Tuple[np.ndarray, int]:
    """
    Dense joint cell codes for one feature subset, applying the Grid Capacity
    Limit (`vsf.pmd.check_grid_capacity`) and, when it is exceeded, the same
    adaptive coarsening (`vsf.pmd.adaptively_coarsen_bins`) as the search.

    Single source of truth for how a combination becomes a cell partition:
    the exhaustive search, the prefix breakdown and the familywise null must
    all score the IDENTICAL partition or their numbers are not comparable.
    """
    X_S = X_discrete[:, combo]
    if not check_grid_capacity([bin_counts[j] for j in combo], n_samples):
        X_S = adaptively_coarsen_bins(X_S, n_samples)
    return cell_codes(X_S)


def _iter_candidates(
    X_discrete: np.ndarray,
    bin_counts: Sequence[int],
    n_samples: int,
    max_d: int,
) -> Iterator[Tuple[Tuple[int, ...], np.ndarray, int]]:
    """Every (combo, cell codes, n_cells) the exhaustive search will score."""
    n_features = X_discrete.shape[1]
    for d in range(1, max_d + 1):
        for combo in itertools.combinations(range(n_features), d):
            codes, n_cells = _subset_codes(X_discrete, bin_counts, n_samples, combo)
            yield combo, codes, n_cells


def _null_floor(table: np.ndarray, n_samples: int, null: NullMethod) -> float:
    """E_0[I_hat] for one candidate under the requested null model."""
    if null == "none":
        return 0.0
    if null == "miller_madow":
        return miller_madow_bias_bits(table.shape[0], table.shape[1], n_samples)
    if null == "exact":
        return expected_mutual_information_bits(table)
    raise ValueError(f"unknown null method: {null!r}")


def _score_candidate(
    z_codes: np.ndarray,
    n_rows: int,
    x_codes: np.ndarray,
    n_cells: int,
    h_target: float,
    null: NullMethod,
) -> Tuple[float, float, float, float]:
    """Returns (mi, mi_null, mi_adj, u_adj) for one candidate partition."""
    n_samples = int(z_codes.shape[0])
    table = contingency_from_codes(z_codes, x_codes, n_rows, n_cells)
    mi = mutual_information_bits(table)
    floor = _null_floor(table, n_samples, null)
    mi_adj = mi - floor
    denom = h_target - floor
    u_adj = float(np.clip(mi_adj / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
    return mi, floor, mi_adj, u_adj


def _prefix_metric_series(
    z_codes: np.ndarray,
    n_rows: int,
    X_discrete: np.ndarray,
    bin_counts: Sequence[int],
    ordered_features: Sequence[int],
    h_target: float,
    null: NullMethod,
) -> Tuple[List[float], List[float], List[float]]:
    """
    For a FIXED feature ordering (one branch's own `selected_features`, in the
    order they are displayed as axes: X, Y, Z, slice), computes
    (mi, mi_adj, u_adj) for every prefix length k = 1..len(ordered_features).

    This is a marginal/projection of ONE already-chosen joint distribution
    onto a subset of its own axes - not `discover_branches`'s per-d
    independent re-optimization (that would generally pick a DIFFERENT subset
    for size k). Each prefix goes through `_subset_codes`, so the
    k = len(ordered_features) entry reproduces the search's own score for the
    winning combination exactly.
    """
    n_samples = int(z_codes.shape[0])
    mi_by_k: List[float] = []
    mi_adj_by_k: List[float] = []
    u_adj_by_k: List[float] = []
    for k in range(1, len(ordered_features) + 1):
        prefix = tuple(ordered_features[:k])
        codes, n_cells = _subset_codes(X_discrete, bin_counts, n_samples, prefix)
        mi, _, mi_adj, u_adj = _score_candidate(
            z_codes, n_rows, codes, n_cells, h_target, null
        )
        mi_by_k.append(mi)
        mi_adj_by_k.append(mi_adj)
        u_adj_by_k.append(u_adj)
    return mi_by_k, mi_adj_by_k, u_adj_by_k


def _resolve_positive_indicator(
    Z_arr: np.ndarray,
    z_codes: np.ndarray,
    n_rows: int,
    positive_class: Optional[object],
) -> Optional[np.ndarray]:
    """
    The 0/1 indicator the centre layer is defined against, or None when the
    target admits no unambiguous positive value.

    Resolution order, deliberately explicit:

    1.  `positive_class` given -> it must match exactly one distinct value of
        the RAW target, and the raw values must correspond 1:1 to the integer
        codes (i.e. the target was not PMD-binned). Anything else raises,
        rather than silently reporting centres of a different class.
    2.  No `positive_class` and a two-valued target -> the higher code, which
        is `np.unique(Z)[-1]`. This is the value `vsf.vis` labels as the
        positive class and the literal 1 of `/api/analyze`'s One-vs-Rest
        `criterion` path, so the three agree by construction.
    3.  Otherwise -> None. A K-class target has no single purity; the caller
        must say which value it means.
    """
    if positive_class is not None:
        uniq = np.unique(Z_arr)
        if uniq.shape[0] != n_rows:
            raise ValueError(
                "positive_class cannot be resolved: the target was discretised "
                f"into {n_rows} bins from {uniq.shape[0]} distinct raw values, so "
                "raw values no longer correspond 1:1 to class codes"
            )
        matches = np.nonzero(uniq == positive_class)[0]
        if matches.shape[0] != 1:
            raise ValueError(
                f"positive_class={positive_class!r} matches {matches.shape[0]} "
                "distinct target values; exactly one is required"
            )
        return (z_codes == int(matches[0])).astype(np.int8)
    if n_rows == 2:
        return (z_codes == 1).astype(np.int8)
    return None


def _center_prefix_series(
    z_binary: np.ndarray,
    X_discrete: np.ndarray,
    bin_counts: Sequence[int],
    ordered_features: Sequence[int],
    spec: CenterSpec,
) -> Tuple[List[float], List[int], List[float]]:
    """
    In-sample (coverage, K, pooled purity) for every prefix of one branch's
    own axis ordering - the centre-layer analogue of `_prefix_metric_series`,
    and subject to the identical caveat: these are projections of THIS
    branch's partition onto its own first k axes, not `discover_branches`'s
    independently optimal k-dimensional branch.
    """
    n_samples = int(z_binary.shape[0])
    coverage: List[float] = []
    n_centers: List[int] = []
    purity: List[float] = []
    for k in range(1, len(ordered_features) + 1):
        codes, n_cells = _subset_codes(
            X_discrete, bin_counts, n_samples, tuple(ordered_features[:k])
        )
        report = center_report(
            z_binary, codes, n_cells, spec, n_repeats=0, n_permutations=0
        )
        coverage.append(report.coverage)
        n_centers.append(report.n_centers)
        purity.append(report.purity_pooled)
    return coverage, n_centers, purity


def select_branch_dimensionality(
    branches: Dict[int, BranchResult], t_threshold: float = 2.0
) -> Optional[int]:
    """
    "How many characteristics does it take to describe the target value?" -
    the smallest branch dimensionality whose OUT-OF-SAMPLE certified coverage
    is positive and beats every lower dimensionality by more than
    `t_threshold` corrected standard errors of the paired difference.

    Returns None when no dimensionality achieves positive cross-validated
    coverage, which is the correct answer for a target that has no certifiable
    centres at all, and must be surfaced as "none" rather than collapsed to
    d = 1. Also None when the branches carry no `CenterReport` (non-binary
    target with no declared positive class) or no cross-validated estimate
    (fewer than `vsf.centers.MIN_POSITIVES_FOR_CV` positives).

    The comparison is valid only because every branch's `CenterReport` was
    cross-validated through `vsf.centers.stratified_repeated_kfold` with the
    same target vector and seed, so the per-fold coverages are paired; that
    is guaranteed by `discover_branches`, which passes one `random_state` to
    all of them.
    """
    cv_by_d: Dict[int, CVCoverage] = {
        d: br.centers.coverage_cv
        for d, br in branches.items()
        if br.centers is not None and br.centers.coverage_cv is not None
    }
    return _select_dimensionality(cv_by_d, t_threshold=t_threshold)


def discover_branches(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[List[str]] = None,
    feature_channels: Optional[List[str]] = None,
    max_d: int = MAX_BRANCH_D,
    null: NullMethod = "exact",
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    n_permutations_familywise: int = 0,
    random_state: Optional[int] = 0,
    progress: Optional[Callable[[int, int], None]] = None,
    objective: Objective = "auto",
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    n_permutations_centers: int = DEFAULT_N_PERMUTATIONS,
    n_permutations_familywise_coverage: int = 0,
    cv_splits: int = 5,
    cv_repeats: int = 5,
) -> Dict[int, BranchResult]:
    """
    Independent Branch Discovery (Project_Master_Document.md Section 4.3).

    For every d in {1, ..., min(max_d, M)}, independently computes

        S*_d = argmax_{S subset of {1,...,M}, |S| = d}  MI_adj(Z; X_S)

    with MI_adj = I_hat - E_0[I_hat], by exhaustively enumerating every
    C(M, d) combination - never a greedy approximation, never a prefiltered
    candidate pool. Returns up to `max_d` branches, keyed by dimensionality; a
    key is missing only when `M < d`.

    Why exhaustive, not greedy: I(Z; X_S) is not submodular in general
    (Krause & Guestrin, 2005). A greedy chain that commits to the single best
    feature at d=1 and only ever ADDS to it can never discover a pair {a, b}
    that is jointly informative while neither a nor b is individually strong -
    exactly the synergy case Project_Master_Document.md Section 4.4 requires
    each branch to be capable of finding.

    Why MI_adj, not raw I_hat: see this module's docstring. Ranking by raw
    I_hat selects on cell count rather than on association and, on a control
    dataset with no signal at all, returns branches reporting up to 9.8 %
    NMI_min.

    Parameters
    ----------
    null
        Bias model subtracted from every candidate. `"exact"` (default) uses
        the exact permutation expectation - O(R * C) per candidate, ~2.1 ms
        median for a 2 x 1379 table and linear in R (16 ms at R = 20), i.e.
        ~0.2 s for the whole M = 7 reference search. `"miller_madow"` uses the closed form and is ~1000x cheaper
        but counts nominal rather than occupied cells, so it over-corrects on
        sparse grids; use it only when profiling shows the exact null
        dominates. `"none"` reproduces v2.0's (incorrect) raw-MI ranking and
        exists solely so the regression can be demonstrated.
    n_permutations
        Replicates for each winning branch's UNCORRECTED `p_value`. Set to 0
        to skip (4 x B table builds; ~0.5 s at B = 999 on the reference
        dataset).
    n_permutations_familywise
        Replicates for the look-elsewhere-corrected `p_value_familywise`.
        Costs B x (the whole candidate family), so it is 0 by default. This is
        the only p-value that is valid for a branch that was selected as an
        argmax, and any published claim must set it.
    random_state
        Seeds both permutation nulls. Fixed by default, so the function stays
        reproducible; with `n_permutations = n_permutations_familywise = 0` it
        is fully deterministic and consumes no randomness.
    progress
        Optional `(done, total)` callback, invoked during the familywise null
        only - the phase whose runtime is worth reporting to a user.
    objective
        What the exhaustive search maximises. `"auto"` (default) resolves to
        `"coverage"` when a positive target value is available and to
        `"mi_adj"` otherwise. `"coverage"` ranks by
        `vsf.centers.coverage_score` -- the statistic the display reports --
        and requires a resolvable positive class; `"mi_adj"` keeps v2.1's
        ranking. This changes WHICH subset wins, not just how it is
        described: a branch can be the MI_adj argmax and contain no centres,
        and vice versa.
    positive_class
        The raw target value that centres are certified against. Optional for
        a two-valued target (the higher `np.unique` code is used, matching
        the renderer and the One-vs-Rest `criterion` path); required for a
        K-class target, without which `BranchResult.centers` is None.
    center_spec
        `vsf.centers.CenterSpec` - the purity floor `tau`, the simultaneous
        error rate `alpha`, and the multiplicity policy. Defaults to
        tau = 0.90, alpha = 0.05, Bonferroni over the branch's occupied
        cells.
    n_permutations_centers
        Replicates for each branch's uncorrected coverage p-value. Cheap: the
        null is drawn directly from the multivariate hypergeometric law of
        the cell counts, O(C) per replicate rather than O(N).
    n_permutations_familywise_coverage
        Replicates for `coverage_p_value_familywise`. Costs B x the whole
        candidate family, like its MI_adj counterpart, and is 0 by default
        for the same reason; a published coverage claim must set it.
    cv_splits / cv_repeats
        Stratified repeated K-fold parameters behind
        `CenterReport.coverage_cv`. Set `cv_repeats = 0` to skip the
        out-of-sample estimate entirely (and with it
        `select_branch_dimensionality`).

    Cost: see Project_Master_Document.md Section 4.6. This function makes no
    attempt to bound the number of combinations evaluated - by explicit
    product decision, "always honest full enumeration", regardless of M.
    Callers with very wide datasets (M >~ 50-100) should expect this to take
    minutes; that is the accepted cost of never approximating.
    """
    if max_d < 1 or max_d > MAX_BRANCH_D:
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
    if n_permutations < 0 or n_permutations_familywise < 0:
        raise ValueError("permutation counts must be non-negative")
    if n_permutations_centers < 0 or n_permutations_familywise_coverage < 0:
        raise ValueError("permutation counts must be non-negative")
    if objective not in ("auto", "mi_adj", "coverage"):
        raise ValueError(f"unknown objective: {objective!r}")
    spec = center_spec if center_spec is not None else CenterSpec()

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
    z_codes, n_rows = cell_codes(Z_discrete)
    h_target = entropy_bits_from_counts(np.bincount(z_codes, minlength=max(1, n_rows)))

    effective_max_d = min(max_d, n_features)
    z_binary = _resolve_positive_indicator(Z_arr, z_codes, n_rows, positive_class)
    if objective == "coverage" and z_binary is None:
        raise ValueError(
            'objective="coverage" needs a positive class: the target has '
            f"{n_rows} distinct values and no positive_class was given"
        )
    if objective == "auto":
        objective = "coverage" if z_binary is not None else "mi_adj"

    # ---- exhaustive search ----------------------------------------------
    # The ranking key is a float for "mi_adj" and a lexicographic tuple for
    # "coverage"; both are compared with `>` against the incumbent, and the
    # objective is fixed for the whole call, so the two key types never meet.
    best_by_d: Dict[int, Tuple[object, Tuple[int, ...]]] = {}
    for combo, codes, n_cells in _iter_candidates(
        X_discrete, bin_counts, n_samples, effective_max_d
    ):
        if objective == "mi_adj":
            _, _, mi_adj, _ = _score_candidate(
                z_codes, n_rows, codes, n_cells, h_target, null
            )
            key: object = mi_adj
        else:
            assert z_binary is not None  # guaranteed by the check above
            key = coverage_score(z_binary, codes, n_cells, spec)
        d = len(combo)
        current = best_by_d.get(d)
        if current is None or key > current[0]:  # type: ignore[operator]
            best_by_d[d] = (key, combo)

    if not best_by_d:
        return branches

    # ---- familywise null, computed once for the whole scan ---------------
    fw_p: Optional[Callable[[float], float]] = None
    if n_permutations_familywise > 0:
        fw_null = familywise_max_null(
            z_codes,
            lambda: _iter_candidates(X_discrete, bin_counts, n_samples, effective_max_d),
            n_permutations=n_permutations_familywise,
            random_state=random_state,
            null=null,
            progress=progress,
        )
        fw_p = fw_null.p_value

    fw_cov_p: Optional[Callable[[float], float]] = None
    if n_permutations_familywise_coverage > 0 and z_binary is not None:
        fw_cov_null = familywise_max_coverage_null(
            z_binary,
            lambda: (
                (combo, codes, n_cells)
                for combo, codes, n_cells in _iter_candidates(
                    X_discrete, bin_counts, n_samples, effective_max_d
                )
            ),
            spec=spec,
            n_permutations=n_permutations_familywise_coverage,
            random_state=random_state,
            progress=progress,
        )
        fw_cov_p = fw_cov_null.p_value

    # ---- per-branch reporting -------------------------------------------
    for d, (_, best_combo) in sorted(best_by_d.items()):
        ordered_features = list(best_combo)
        mi_by_k, mi_adj_by_k, u_adj_by_k = _prefix_metric_series(
            z_codes, n_rows, X_discrete, bin_counts, ordered_features, h_target, null
        )
        codes, n_cells = _subset_codes(X_discrete, bin_counts, n_samples, best_combo)
        mi, floor, mi_adj, u_adj = _score_candidate(
            z_codes, n_rows, codes, n_cells, h_target, null
        )

        p_value: Optional[float] = None
        if n_permutations > 0:
            p_value, _, _ = permutation_pvalue(
                z_codes, codes, n_rows, n_cells, n_permutations, random_state
            )

        report = information_report(z_codes, codes, null=null)

        centers: Optional[CenterReport] = None
        cov_by_k: List[float] = []
        k_by_k: List[int] = []
        pur_by_k: List[float] = []
        if z_binary is not None:
            centers = center_report(
                z_binary,
                codes,
                n_cells,
                spec,
                n_splits=cv_splits,
                n_repeats=cv_repeats,
                n_permutations=n_permutations_centers,
                random_state=random_state,
            )
            cov_by_k, k_by_k, pur_by_k = _center_prefix_series(
                z_binary, X_discrete, bin_counts, ordered_features, spec
            )

        branches[d] = BranchResult(
            d=d,
            selected_features=ordered_features,
            selected_feature_names=[feature_names[j] for j in ordered_features],
            mi=mi,
            mi_null=floor,
            mi_adj=mi_adj,
            u_adj=u_adj,
            h_target=h_target,
            mi_by_prefix_d=mi_by_k,
            mi_adj_by_prefix_d=mi_adj_by_k,
            u_adj_by_prefix_d=u_adj_by_k,
            p_value=p_value,
            p_value_familywise=fw_p(mi_adj) if fw_p is not None else None,
            per_class=report.per_class,
            centers=centers,
            coverage_by_prefix_d=cov_by_k,
            n_centers_by_prefix_d=k_by_k,
            purity_by_prefix_d=pur_by_k,
            coverage_p_value_familywise=(
                fw_cov_p(centers.coverage)
                if (fw_cov_p is not None and centers is not None)
                else None
            ),
        )

    return branches


class BranchEngine:
    """
    Thin, stateless wrapper around `discover_branches` matching v1.0's
    `AVREngine(...).fit(...)` call shape, so callers (server handlers,
    `vsf.dashboard`, tests) construct-then-fit as before.

    Unlike v2.0's wrapper this DOES carry statistical parameters. That is not a
    regression toward v1.0's removed significance infrastructure: v1.0 used
    permutation tests to GATE greedy feature selection, whereas these
    parameters only attach a significance statement to an already-selected,
    exhaustively-searched branch. Selection itself remains deterministic given
    `null`.
    """

    def __init__(
        self,
        max_d: int = MAX_BRANCH_D,
        null: NullMethod = "exact",
        n_permutations: int = DEFAULT_N_PERMUTATIONS,
        n_permutations_familywise: int = 0,
        random_state: Optional[int] = 0,
        objective: Objective = "auto",
        positive_class: Optional[object] = None,
        center_spec: Optional[CenterSpec] = None,
        n_permutations_centers: int = DEFAULT_N_PERMUTATIONS,
        n_permutations_familywise_coverage: int = 0,
        cv_splits: int = 5,
        cv_repeats: int = 5,
    ):
        if max_d < 1 or max_d > MAX_BRANCH_D:
            raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
        self.max_d = max_d
        self.null: NullMethod = null
        self.n_permutations = n_permutations
        self.n_permutations_familywise = n_permutations_familywise
        self.random_state = random_state
        self.objective: Objective = objective
        self.positive_class = positive_class
        self.center_spec = center_spec
        self.n_permutations_centers = n_permutations_centers
        self.n_permutations_familywise_coverage = n_permutations_familywise_coverage
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        feature_names: Optional[List[str]] = None,
        feature_channels: Optional[List[str]] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[int, BranchResult]:
        return discover_branches(
            X,
            Z,
            feature_names=feature_names,
            feature_channels=feature_channels,
            max_d=self.max_d,
            null=self.null,
            n_permutations=self.n_permutations,
            n_permutations_familywise=self.n_permutations_familywise,
            random_state=self.random_state,
            progress=progress,
            objective=self.objective,
            positive_class=self.positive_class,
            center_spec=self.center_spec,
            n_permutations_centers=self.n_permutations_centers,
            n_permutations_familywise_coverage=self.n_permutations_familywise_coverage,
            cv_splits=self.cv_splits,
            cv_repeats=self.cv_repeats,
        )
