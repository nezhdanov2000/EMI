"""
Post-selection certificates, nested cross-validation and schema stability.

Why this module exists
-----------------------------------------------------------------------
`CenterSpec(rule="certified")` corrects for the C occupied cells of ONE
schema. `vsf.avr.discover_branches` reports, for each d, the best of
C(M, d) schemas. Proposition 2 of the specification is true for a schema
fixed in advance and false for the reported one: on pure noise (N = 3000,
12 four-level columns, prevalence 0.45, tau = 0.5, alpha = 0.05) the
reported 3-D winner carries a "certified" centre in 36 % of runs and the
4-D winner in 60 % (`experiments/null_certificate.py`). The same selection
makes `CenterReport.coverage_cv` optimistic, because the schema - not only
its centres - was chosen on the rows the folds later hold out.

This module provides the corrected versions. Nothing in `vsf.avr` or
`vsf.centers` changes meaning; the functions here are separate entry points
whose results carry their own guarantee statement.

The claim and the model
-----------------------------------------------------------------------
Assumption: conditional on the feature values X, the target values of
distinct rows are independent (true for an i.i.d. sample; not for rows
that share an outcome, e.g. members of one family). Partitions, and the
family of partitions, are functions of X only. Conditional on X, a cell's
count is then a sum of independent Bernoulli(p_i), p_i = P(Z = 1 | x_i),
and the certificate is the claim

    pbar_c = (1 / n_c) * sum over the cell's rows of P(Z = 1 | x_i) > tau,

the mean purity of the rows in hand. It is not a claim about the
population share P(Z = 1 | X in c), which differs from pbar_c by the
sampling of the cell's composition. A cell is certified iff
k >= `vsf.centers.min_successes_to_certify_heterogeneous`(n, tau, level),
whose docstring proves P(certified) <= level whenever pbar_c <= tau
(Hoeffding, 1956, Theorem 4, plus monotonicity of the binomial tail), for
any level and every tau at once. `p_value` below is the exact binomial tail
at tau, reported for reading; the decision also requires k >= ceil(n tau) + 1.

Certification methods
-----------------------------------------------------------------------
`"family_bonferroni"`
    Every occupied cell of every partition the search can report or display
    - each scored schema's search partition and, where the capacity rule
    coarsened it, its uncoarsened partition too - over all dimensionalities
    is one family of T tests (`vsf.avr.family_cell_count`). A cell is
    certified iff p_c <= alpha / T. The union bound over the WHOLE family
    makes the choice of schema irrelevant: P(any cell with pi_c <= tau
    certified, anywhere) <= alpha, whatever the search does, with no
    assumption on the dependence between cells. The search ranks schemas by
    the coverage of cells certified this way, so the reported branch
    maximises the honest quantity. This is exactly
    `CenterSpec(rule="certified", multiplicity="family")`, the certificate
    the interface colours by. Price: T is in the thousands on small data and
    grows as sum_d C(M, d) x cells. T does not depend on `min_samples`,
    which a user may move after seeing the data.

`"split"`
    The rows are divided at random, independently of the target, into a
    search half A and an evaluation half B. The search runs on A and ranks
    by observed purity (`rule="purity"` at the caller's tau and
    min_samples) whatever rule the caller's spec names - A only proposes,
    B certifies. On A the cells of every reported schema that pass
    a screen (default: A-purity >= tau and A-size >= min_samples) become the
    candidate cells. Only those are tested, on B's counts, at alpha / T_B
    where T_B is the number of candidates (all reported branches together by
    default). Conditional on the feature values and on A, B's target values
    are independent of everything that chose the candidates, so the
    Bonferroni bound over T_B is exact: strong family-wise control of the
    reported certificates. The split is deliberately NOT stratified on the
    target - a stratified split fixes B's positive count from the target
    values and breaks the conditional-independence argument.

What was considered and rejected: a Westfall-Young min-p permutation null.
Permuting the target tests "the target is independent of the features",
not "pi_c <= tau". With tau above the base rate, a cell whose true purity is
0.85 is extreme against independence and would be "certified" as > 0.90
with high probability. It controls false certificates only under the global
null (weak control) and cannot back the claim the display makes.

Neither method makes the reported coverage an out-of-sample estimate: the
evaluation counts both certify the cells and measure their coverage. The
out-of-sample number is `nested_crossvalidation`.

Nested cross-validation and stability
-----------------------------------------------------------------------
`vsf.centers.crossvalidated_coverage` re-selects centres on each training
fold but keeps the schema chosen on all rows. `nested_crossvalidation`
repeats the whole search on each training fold, applies the fold's own
winner and centres to the held-out fold, and reports the Nadeau-Bengio
corrected estimate on the same folds as the fixed-schema version (the folds
are a function of the target and the seed only), so the two are paired. It
also reports how often each fold's winner coincides with the full-data
winner: a schema that changes from fold to fold is not a finding about the
population, whatever its coverage.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, Final, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .avr import (
    MAX_BRANCH_D,
    Direction,
    _CandidateFactory,
    _exhaustive_search,
    _prepare_search,
    apply_fitted_partition,
    family_cell_count,
    resolve_center_spec,
)
from .centers import (
    MIN_POSITIVES_FOR_CV,
    CenterSpec,
    CVCoverage,
    _cell_counts,
    _lgamma_table,
    clopper_pearson_lower,
    crossvalidated_coverage,
    select_centers,
    select_dimensionality,
    min_successes_to_certify_heterogeneous,
    stratified_repeated_kfold,
    summarize_cv,
)
from .screen import exact_dependencies

__all__ = [
    "BranchMultiplicity",
    "CertificationMethod",
    "CertifiedCell",
    "FoldEncoding",
    "NestedCVBranch",
    "NestedCVResult",
    "SchemaStability",
    "SelectiveCertificate",
    "SelectiveDiscovery",
    "certification_passes",
    "certify_discovery",
    "exact_upper_tail",
    "family_test_count",
    "nested_passes",
    "nested_crossvalidation",
    "random_halves",
    "schema_stability",
]

CertificationMethod = Literal["family_bonferroni", "split"]
BranchMultiplicity = Literal["all_branches", "per_branch"]
ScreenRule = Literal["purity", "all"]
#: Which rows a nested cross-validation fold fits its partitions on.
FoldEncoding = Literal["train", "all_rows"]
#: `(done, total)` callback, counted in exhaustive passes over the family.
Progress = Callable[[int, int], None]

_TAIL_MEMO_MAX_ENTRIES: Final[int] = 20_000
_TAIL_MEMO: Dict[Tuple[float, int], np.ndarray] = {}


# --------------------------------------------------------------------------
# Exact binomial upper tail
# --------------------------------------------------------------------------
def _log_upper_tail_table(n: int, tau: float) -> np.ndarray:
    """log P(Bin(n, tau) >= k) for k = 0 .. n, by a reverse log-sum-exp."""
    key = (float(tau), int(n))
    hit = _TAIL_MEMO.get(key)
    if hit is not None:
        return hit
    lg = _lgamma_table(n + 1)
    k = np.arange(n + 1, dtype=np.int64)
    log_pmf = (
        lg[n + 1] - lg[k + 1] - lg[n - k + 1]
        + k.astype(np.float64) * math.log(tau)
        + (n - k).astype(np.float64) * math.log1p(-tau)
    )
    table = np.minimum(np.logaddexp.accumulate(log_pmf[::-1])[::-1], 0.0)
    if len(_TAIL_MEMO) >= _TAIL_MEMO_MAX_ENTRIES:
        _TAIL_MEMO.clear()
    _TAIL_MEMO[key] = table
    return table


def exact_upper_tail(
    k: np.ndarray | Sequence[int] | int, n: np.ndarray | Sequence[int] | int, tau: float
) -> np.ndarray:
    """
    One-sided exact binomial p-value P(Bin(n, tau) >= k) of H_0: pi <= tau,
    elementwise. k <= 0 gives 1; k > n gives 0; n = 0 gives 1 (no evidence).

    Agrees with `vsf.centers.min_successes_to_certify`: p <= a iff
    k >= min_successes_to_certify(n, tau, a), up to floating-point ties at
    the boundary (`tests/test_selective.py`).
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    k_arr = np.atleast_1d(np.asarray(k, dtype=np.int64))
    n_arr = np.atleast_1d(np.asarray(n, dtype=np.int64))
    k_arr, n_arr = np.broadcast_arrays(k_arr, n_arr)
    if np.any(n_arr < 0):
        raise ValueError("cell sizes must be non-negative")
    out = np.ones(k_arr.shape, dtype=np.float64)
    inside = (k_arr > 0) & (k_arr <= n_arr) & (n_arr > 0)
    out[(k_arr > n_arr) & (n_arr > 0)] = 0.0
    if np.any(inside):
        ks, ns = k_arr[inside], n_arr[inside]
        vals = np.empty(ks.shape, dtype=np.float64)
        for size in np.unique(ns).tolist():
            sel = ns == size
            vals[sel] = np.exp(_log_upper_tail_table(int(size), tau)[ks[sel]])
        out[inside] = vals
    return out


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CertifiedCell:
    """One tested cell of a reported schema, on the evaluation sample."""

    cell: int
    n: int
    k: int
    p_value: float
    p_adjusted: float
    purity_lower: float
    certified: bool


@dataclass(frozen=True)
class SelectiveCertificate:
    """
    The certificate of one reported branch.

    `cells` lists every TESTED cell (certified or not), most significant
    first; `certified_cells` the ones that passed. `coverage_eval` is the
    share of the evaluation sample's positives inside certified cells and
    `coverage_all_rows` the same share over every row - both descriptive:
    the evaluation counts decided certification, so neither is an
    out-of-sample estimate (`nested_crossvalidation` is).
    """

    d: int
    features: Tuple[int, ...]
    feature_names: Tuple[str, ...]
    cells: Tuple[CertifiedCell, ...]
    n_eval: int
    n_eval_positive: int
    coverage_eval: float
    coverage_all_rows: float
    mass_eval: float

    @property
    def certified_cells(self) -> Tuple[CertifiedCell, ...]:
        return tuple(c for c in self.cells if c.certified)

    @property
    def n_certified(self) -> int:
        return sum(1 for c in self.cells if c.certified)


@dataclass(frozen=True)
class SelectiveDiscovery:
    """Result of `certify_discovery`: the reported branches and their certificates."""

    method: CertificationMethod
    spec: CenterSpec
    branch_multiplicity: BranchMultiplicity
    screen: ScreenRule
    n_tests: Dict[int, int]
    per_cell_level: Dict[int, Optional[float]]
    guarantee: str
    branches: Dict[int, SelectiveCertificate]
    search_rows: int
    eval_rows: int
    random_state: Optional[int]


@dataclass(frozen=True)
class SchemaStability:
    """How consistently the search picks the same schema across resamples."""

    n_resamples: int
    reference: Tuple[int, ...]
    share_equal_to_reference: float
    modal_schema: Tuple[int, ...]
    modal_share: float
    n_distinct: int
    mean_pairwise_jaccard: float


@dataclass(frozen=True)
class NestedCVBranch:
    d: int
    full_data_winner: Tuple[int, ...]
    full_data_winner_names: Tuple[str, ...]
    nested: CVCoverage
    fixed_schema: CVCoverage
    fold_winners: Tuple[Tuple[int, ...], ...]
    stability: SchemaStability


@dataclass(frozen=True)
class NestedCVResult:
    branches: Dict[int, NestedCVBranch]
    spec: CenterSpec
    n_splits: int
    n_repeats: int
    random_state: Optional[int]
    n_positive: int
    undetermined_reason: Optional[str]
    encoding: "FoldEncoding" = "train"

    def select_dimensionality(self, t_threshold: float = 2.0) -> Optional[int]:
        """`vsf.centers.select_dimensionality` on the NESTED estimates (paired folds)."""
        if self.undetermined_reason is not None:
            return None
        return select_dimensionality(
            {d: b.nested for d, b in self.branches.items()}, t_threshold=t_threshold
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def random_halves(n_rows: int, random_state: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    A uniformly random split of 0 .. n_rows-1 into a search half A
    (ceil(n/2) rows) and an evaluation half B, sorted. A function of
    (n_rows, random_state) only - never of the target.
    """
    if n_rows < 2:
        raise ValueError(f"need at least 2 rows to split, got {n_rows}")
    perm = np.random.default_rng(random_state).permutation(n_rows)
    cut = (n_rows + 1) // 2
    return np.sort(perm[:cut]), np.sort(perm[cut:])


def family_test_count(factory: _CandidateFactory, max_d: int) -> int:
    """
    T of `"family_bonferroni"`: `vsf.avr.family_cell_count` - every occupied
    cell of every partition a search over d <= max_d can report or display
    (pruned renamings carry the row sets of a counted partition and add no
    new hypothesis).
    """
    return family_cell_count(factory, max_d)


def certification_passes(method: CertificationMethod) -> int:
    """Exhaustive passes over the candidate family `certify_discovery` makes."""
    return 2 if method == "family_bonferroni" else 1


def nested_passes(n_splits: int, n_repeats: int) -> int:
    """Exhaustive searches `nested_crossvalidation` runs: all rows, then every split."""
    return 1 + n_splits * n_repeats


def schema_stability(
    winners: Sequence[Tuple[int, ...]], reference: Tuple[int, ...]
) -> SchemaStability:
    """Agreement of per-resample winners with each other and with `reference`."""
    ws = [tuple(sorted(w)) for w in winners]
    ref = tuple(sorted(reference))
    if not ws:
        return SchemaStability(0, ref, float("nan"), ref, float("nan"), 0, float("nan"))
    counts = Counter(ws)
    modal, modal_n = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ws) > 1:
        jac = [
            len(set(a) & set(b)) / len(set(a) | set(b))
            for a, b in combinations(ws, 2)
        ]
        mean_jac = float(np.mean(jac))
    else:
        mean_jac = 1.0
    return SchemaStability(
        n_resamples=len(ws),
        reference=ref,
        share_equal_to_reference=sum(1 for w in ws if w == ref) / len(ws),
        modal_schema=modal,
        modal_share=modal_n / len(ws),
        n_distinct=len(counts),
        mean_pairwise_jaccard=mean_jac,
    )


def _require_certifiable(spec: CenterSpec) -> None:
    if not (0.0 < spec.tau < 1.0):
        raise ValueError(
            f"a certificate needs 0 < tau < 1, got tau = {spec.tau}: "
            "H_0: pi <= 1 can never be rejected"
        )


def _tested_cells(
    k: np.ndarray, n: np.ndarray, candidates: np.ndarray, tau: float, alpha: float, n_tests: int,
) -> Tuple[CertifiedCell, ...]:
    """Test `candidates` (cell indices) at alpha / n_tests; most significant first."""
    if candidates.size == 0:
        return ()
    level = alpha / n_tests
    kc, nc = k[candidates], n[candidates]
    p = exact_upper_tail(kc, nc, tau)
    k_min = min_successes_to_certify_heterogeneous(nc, tau, level)
    lower = clopper_pearson_lower(kc, nc, level)
    order = np.lexsort((candidates, -kc, p))
    return tuple(
        CertifiedCell(
            cell=int(candidates[i]),
            n=int(nc[i]),
            k=int(kc[i]),
            p_value=float(p[i]),
            p_adjusted=float(min(1.0, p[i] * n_tests)),
            purity_lower=float(lower[i]),
            certified=bool(kc[i] >= k_min[i]),
        )
        for i in order.tolist()
    )


def _certificate(
    d: int,
    combo: Tuple[int, ...],
    names: Sequence[str],
    cells: Tuple[CertifiedCell, ...],
    z_eval: np.ndarray,
    z_all: np.ndarray,
    codes_all: np.ndarray,
) -> SelectiveCertificate:
    certified = np.array([c.cell for c in cells if c.certified], dtype=np.int64)
    n_eval = int(z_eval.shape[0])
    pos_eval = int(z_eval.sum())
    k_eval = sum(c.k for c in cells if c.certified)
    n_sel = sum(c.n for c in cells if c.certified)
    in_cert = np.isin(codes_all, certified)
    pos_all = int(z_all.sum())
    return SelectiveCertificate(
        d=d,
        features=tuple(combo),
        feature_names=tuple(names[j] for j in combo),
        cells=cells,
        n_eval=n_eval,
        n_eval_positive=pos_eval,
        coverage_eval=(k_eval / pos_eval) if pos_eval else 0.0,
        coverage_all_rows=(int(z_all[in_cert].sum()) / pos_all) if pos_all else 0.0,
        mass_eval=(n_sel / n_eval) if n_eval else 0.0,
    )


# --------------------------------------------------------------------------
# Certified discovery
# --------------------------------------------------------------------------
def certify_discovery(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[List[str]] = None,
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    method: CertificationMethod = "split",
    max_d: int = MAX_BRANCH_D,
    screen: ScreenRule = "purity",
    branch_multiplicity: BranchMultiplicity = "all_branches",
    random_state: Optional[int] = 0,
    direction: Direction = "presence",
    prune_dependent: bool = False,
    progress: Optional[Progress] = None,
) -> SelectiveDiscovery:
    """
    Branch discovery whose certificates hold for the REPORTED schemas.

    `center_spec` supplies tau, alpha and min_samples (and, for `"split"`,
    the rule the search on half A ranks by). See the module docstring for
    what each `method` guarantees. `branch_multiplicity="all_branches"`
    (default) makes one family of the tests of all reported dimensionalities
    under `"split"`, since the display shows them together;
    `"family_bonferroni"` always covers every dimensionality.

    `progress` is called after each exhaustive pass over the candidate
    family: two passes for `"family_bonferroni"` (count, search), one for
    `"split"` (`certification_passes`).
    """
    spec = center_spec if center_spec is not None else CenterSpec()
    _require_certifiable(spec)
    if not (1 <= max_d <= MAX_BRANCH_D):
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
    if method not in ("family_bonferroni", "split"):
        raise ValueError(f"unknown certification method: {method!r}")
    if screen not in ("purity", "all"):
        raise ValueError(f"unknown screen: {screen!r}")
    if branch_multiplicity not in ("all_branches", "per_branch"):
        raise ValueError(f"unknown branch multiplicity: {branch_multiplicity!r}")

    prepared = _prepare_search(X, Z, feature_names, positive_class, spec, direction, prune_dependent)
    if prepared is None:
        raise ValueError("no feature columns")
    factory, z, names = prepared
    eff_d = min(max_d, factory.n_features)
    z64 = z.astype(np.int64)
    m = max(1, spec.min_samples)

    total_passes = certification_passes(method)

    def tick(done: int) -> None:
        if progress is not None:
            progress(done, total_passes)

    if method == "family_bonferroni":
        search_spec = resolve_center_spec(
            factory,
            CenterSpec(tau=spec.tau, alpha=spec.alpha, rule="certified", min_samples=m,
                       method="clopper-pearson", multiplicity="family"),
            eff_d,
        )
        assert search_spec.family_tests is not None
        t_family = int(search_spec.family_tests)
        tick(1)
        best = _exhaustive_search(factory, z64, 2, [1], search_spec, eff_d)[0]
        tick(2)
        branches: Dict[int, SelectiveCertificate] = {}
        for d, (_, combo) in sorted(best.items()):
            codes, n_cells = factory.codes(tuple(combo))
            k, n = _cell_counts(z, codes, n_cells)
            candidates = np.nonzero(n >= m)[0]
            cells = _tested_cells(k, n, candidates, spec.tau, spec.alpha, max(1, t_family))
            branches[d] = _certificate(d, tuple(combo), names, cells, z, z, codes)
        return SelectiveDiscovery(
            method=method, spec=spec, branch_multiplicity="all_branches", screen="all",
            n_tests={d: t_family for d in branches},
            per_cell_level={d: (spec.alpha / t_family if t_family else None) for d in branches},
            guarantee=(
                f"P(any cell with purity <= {spec.tau:g} is certified, among the "
                f"{t_family} cells of every partition of d <= {eff_d} the search can show) "
                f"<= {spec.alpha:g}; valid for the reported schemas whatever the search chose "
                "(Bonferroni over the whole family; rows i.i.d.)"
            ),
            branches=branches,
            search_rows=factory.n_samples,
            eval_rows=factory.n_samples,
            random_state=random_state,
        )

    # ---- split -------------------------------------------------------------
    rows_a, rows_b = random_halves(factory.n_samples, random_state)
    proposal_spec = CenterSpec(tau=spec.tau, alpha=spec.alpha, rule="purity", min_samples=m)
    best = _exhaustive_search(factory, z64, 2, [1], proposal_spec, eff_d, rows=rows_a)[0]
    tick(1)
    z_a, z_b = z[rows_a], z[rows_b]
    per_branch: Dict[int, Tuple[Tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for d, (_, combo) in sorted(best.items()):
        codes, n_cells = factory.codes(tuple(combo))
        k_a, n_a = _cell_counts(z_a, codes[rows_a], n_cells)
        k_b, n_b = _cell_counts(z_b, codes[rows_b], n_cells)
        if screen == "purity":
            screened = (n_a >= m) & (k_a >= np.ceil(spec.tau * n_a - 1e-9)) & (n_a > 0)
        else:
            screened = n_a > 0
        candidates = np.nonzero(screened & (n_b >= m))[0]
        per_branch[d] = (tuple(combo), codes, k_b, n_b, candidates)

    n_tests: Dict[int, int] = {}
    if branch_multiplicity == "all_branches":
        total = sum(int(v[4].size) for v in per_branch.values())
        n_tests = {d: total for d in per_branch}
    else:
        n_tests = {d: int(v[4].size) for d, v in per_branch.items()}

    branches = {}
    for d, (combo, codes, k_b, n_b, candidates) in per_branch.items():
        cells = _tested_cells(k_b, n_b, candidates, spec.tau, spec.alpha, max(1, n_tests[d]))
        branches[d] = _certificate(d, combo, names, cells, z_b, z, codes)
    scope = (
        "all reported branches together" if branch_multiplicity == "all_branches"
        else "each reported branch separately"
    )
    return SelectiveDiscovery(
        method=method, spec=spec, branch_multiplicity=branch_multiplicity, screen=screen,
        n_tests=n_tests,
        per_cell_level={d: (spec.alpha / t if t else None) for d, t in n_tests.items()},
        guarantee=(
            f"P(any certified cell has purity <= {spec.tau:g}) <= {spec.alpha:g} over "
            f"{scope}: schemas and candidate cells chosen on a random half "
            f"({rows_a.size} rows), tested once on the other half ({rows_b.size} rows) "
            "with Bonferroni over the candidates (rows i.i.d.)"
        ),
        branches=branches,
        search_rows=int(rows_a.size),
        eval_rows=int(rows_b.size),
        random_state=random_state,
    )


# --------------------------------------------------------------------------
# Nested cross-validation
# --------------------------------------------------------------------------
def nested_crossvalidation(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[List[str]] = None,
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    max_d: int = MAX_BRANCH_D,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: Optional[int] = 0,
    direction: Direction = "presence",
    prune_dependent: bool = False,
    progress: Optional[Progress] = None,
    encoding: FoldEncoding = "train",
) -> NestedCVResult:
    """
    Out-of-sample coverage of the whole procedure "search, then select
    centres", per dimensionality, with schema stability.

    For every split of `vsf.centers.stratified_repeated_kfold` (identical
    folds to `crossvalidated_coverage` for the same target and seed), the
    exhaustive search runs on the training rows only; the fold's winner of
    each d and its centres (selected on the training counts under
    `center_spec`) are applied to the test rows. `fixed_schema` is the
    existing estimate for the full-data winner on the same folds, so
    `nested` and `fixed_schema` are paired.

    `encoding` decides which rows the partitions are fitted on:
    `"train"` (default) - the grid capacity, the level frequencies that
    decide merges, the merge state and, for a family certificate, the
    family size T are all computed from the training rows alone, and the
    fitted partition is then applied to the test rows
    (`vsf.avr.apply_fitted_partition`); nothing about a test row reaches
    the procedure before it is scored. `"all_rows"` - partitions are built
    from every row's feature values, as the interactive product does (no
    target is read, so this is valid conditional on the features, but the
    test rows' feature values shape the grid). Both are reported by
    `experiments/nested_cv.py` so the difference is measured, not assumed.
    The level codes themselves (value -> integer, sorted) always come from
    all rows; they are a relabelling and carry no frequencies.

    Cost: 1 + n_splits x n_repeats full searches (`nested_passes`);
    `progress` is called after each.
    """
    requested_spec = center_spec if center_spec is not None else CenterSpec()
    if not (1 <= max_d <= MAX_BRANCH_D):
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
    if encoding not in ("train", "all_rows"):
        raise ValueError(f"encoding must be 'train' or 'all_rows', got {encoding!r}")
    prepared = _prepare_search(
        X, Z, feature_names, positive_class, requested_spec, direction, prune_dependent
    )
    if prepared is None:
        raise ValueError("no feature columns")
    factory, z, names = prepared
    X_arr = np.asarray(X)
    eff_d = min(max_d, factory.n_features)
    spec = resolve_center_spec(factory, requested_spec, eff_d)
    z64 = z.astype(np.int64)
    n_positive = int(z.sum())
    if n_positive < MIN_POSITIVES_FOR_CV:
        return NestedCVResult(
            branches={}, spec=spec, n_splits=n_splits, n_repeats=n_repeats,
            random_state=random_state, n_positive=n_positive, encoding=encoding,
            undetermined_reason=(
                f"only {n_positive} samples carry the searched value; "
                f"{MIN_POSITIVES_FOR_CV} are required for an out-of-sample estimate"
            ),
        )

    full = _exhaustive_search(factory, z64, 2, [1], spec, eff_d)[0]
    splits = stratified_repeated_kfold(z, n_splits, n_repeats, random_state)
    total_passes = nested_passes(n_splits, n_repeats)
    if progress is not None:
        progress(1, total_passes)
    n_obs = len(splits)
    cov: Dict[int, np.ndarray] = {d: np.zeros(n_obs) for d in full}
    pur: Dict[int, np.ndarray] = {d: np.full(n_obs, np.nan) for d in full}
    winners: Dict[int, List[Tuple[int, ...]]] = {d: [] for d in full}
    code_cache: Dict[Tuple[int, ...], Tuple[np.ndarray, int]] = {}

    def codes_of(combo: Tuple[int, ...]) -> Tuple[np.ndarray, int]:
        hit = code_cache.get(combo)
        if hit is None:
            hit = factory.codes(combo)
            code_cache[combo] = hit
        return hit

    for i, (train, test) in enumerate(splits):
        if encoding == "train":
            fold_factory = _CandidateFactory(
                factory._raw[train], factory.bin_counts, int(train.shape[0]),
                ordered=factory.ordered,
                determined_by=(
                    exact_dependencies(X_arr[train], names) if prune_dependent else None
                ),
            )
            fold_spec = resolve_center_spec(fold_factory, requested_spec, eff_d)
            best = _exhaustive_search(
                fold_factory, z64[train], 2, [1], fold_spec, eff_d
            )[0]
        else:
            fold_spec = spec
            best = _exhaustive_search(factory, z64, 2, [1], spec, eff_d, rows=train)[0]
        for d in full:
            combo = tuple(best[d][1])
            winners[d].append(combo)
            if encoding == "train":
                codes, n_cells = apply_fitted_partition(fold_factory, factory._raw, combo)
            else:
                codes, n_cells = codes_of(combo)
            k_tr, n_tr = _cell_counts(z[train], codes[train], n_cells)
            mask, _ = select_centers(k_tr, n_tr, fold_spec)
            k_te, n_te = _cell_counts(z[test], codes[test], n_cells)
            pos_te = int(k_te.sum())
            k_sel = int(k_te[mask].sum())
            n_sel = int(n_te[mask].sum())
            cov[d][i] = (k_sel / pos_te) if pos_te > 0 else 0.0
            if n_sel > 0:
                pur[d][i] = k_sel / n_sel
        if progress is not None:
            progress(i + 2, total_passes)

    branches: Dict[int, NestedCVBranch] = {}
    for d, (_, combo) in sorted(full.items()):
        combo = tuple(combo)
        codes, n_cells = codes_of(combo)
        branches[d] = NestedCVBranch(
            d=d,
            full_data_winner=combo,
            full_data_winner_names=tuple(names[j] for j in combo),
            nested=summarize_cv(cov[d], pur[d], n_splits, n_repeats),
            fixed_schema=crossvalidated_coverage(
                z, codes, n_cells, spec, n_splits=n_splits,
                n_repeats=n_repeats, random_state=random_state,
            ),
            fold_winners=tuple(winners[d]),
            stability=schema_stability(winners[d], combo),
        )
    return NestedCVResult(
        branches=branches, spec=spec, n_splits=n_splits, n_repeats=n_repeats,
        random_state=random_state, n_positive=n_positive, undetermined_reason=None,
        encoding=encoding,
    )
