"""
Unit tests for `vsf.centers` — the certified-discrete-centre layer added in
v2.2.

These are not incidental coverage. Each block pins one of the claims the v2.2
redesign rests on, so that a future "simplification" back toward a point
purity with hand-set colour bands fails loudly rather than silently:

  1. The special-function layer is exact. `_betainc` reproduces the binomial
     tail it is standing in for, so the Clopper-Pearson bound is the textbook
     object and not an approximation of one.
  2. Clopper-Pearson is used rather than Wilson for a stated reason, and that
     reason is measured here, not asserted: at n = 30 the one-sided 95%
     Wilson lower bound has a minimum coverage of 88.1%, the Clopper-Pearson
     bound 95.0%. A certificate whose actual error rate is 12% is not a
     certificate.
  3. The two routes into a certification decision — invert the beta quantile
     and compare to tau, or evaluate the exact binomial tail and compare
     counts — agree cell by cell. The renderer draws the first and the
     permutation nulls count the second; they must not be able to disagree.
  4. Bonferroni over the occupied cells is load-bearing and its effect is
     quantified exactly: on a 200-cell grid whose every cell sits exactly at
     the null boundary pi = tau, the uncorrected rule certifies 8.5 cells in
     expectation and the corrected rule 0.005.
  5. A cell holding one sample cannot be certified at any sensible (tau,
     alpha). This is the regression witness for deleting the "Noise Reduction
     (minimum samples)" slider: the bound already does that job, with a
     stated error rate instead of a hand-set integer.
  6. Coverage is 1 under a deterministic relation, 0 under pure noise at high
     cardinality, and — the property no in-sample statistic has — its
     cross-validated version is strictly SMALLER than its in-sample version
     when cells are selected by their own contents.
  7. `select_dimensionality` returns the smallest SUFFICIENT dimensionality,
     does not stop scanning at the first flat step, and returns None rather
     than 1 when nothing certifies.

Block 8 carries the headline REGRESSION WITNESS, on `data/adult_census.csv`:
target `occupation = Armed-Forces`, U_adj = 41.3% at d = 3 with zero
certified centres and a highest cell purity of 2.42%. Those numbers are
asserted, because "U_adj is high, therefore the display is informative" is
the exact inference v2.2 exists to break, and it is superficially convincing.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vsf.centers import (
    MIN_POSITIVES_FOR_CV,
    CenterSpec,
    binarize_target,
    center_report,
    certified_centers,
    clopper_pearson_lower,
    select_centers,
    clopper_pearson_upper,
    coverage_null,
    coverage_score,
    crossvalidated_coverage,
    familywise_max_coverage_null,
    min_successes_to_certify,
    paired_gain,
    select_dimensionality,
    stratified_repeated_kfold,
    wilson_lower,
)
from vsf.centers import CVCoverage, _betainc


def _binomial_tail(n: int, k: int, p: float) -> float:
    """P(Bin(n, p) >= k), by exact summation."""
    return float(
        sum(math.comb(n, j) * p ** j * (1.0 - p) ** (n - j) for j in range(k, n + 1))
    )


def _lower_bound_coverage(n: int, alpha: float, lower_fn, grid: np.ndarray) -> np.ndarray:
    """
    Exact (not simulated) coverage of a one-sided lower bound: for each true
    p, the probability that the interval [L(K, n), 1] contains p, summed over
    the binomial law of K. No Monte Carlo, so the assertion is a fact about
    the estimator rather than about a seed.
    """
    ks = np.arange(n + 1)
    bounds = lower_fn(ks, np.full(n + 1, n), alpha)
    out = np.empty(grid.shape[0], dtype=float)
    for i, p in enumerate(grid.tolist()):
        pmf = np.array([math.comb(n, int(k)) * p ** int(k) * (1 - p) ** (n - int(k)) for k in ks])
        out[i] = float(pmf[bounds <= p].sum())
    return out


# ---------------------------------------------------------------------------
# 1. Special functions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n,k,p",
    [(10, 3, 0.4), (50, 25, 0.5), (330, 8, 0.02), (7, 7, 0.9), (1, 1, 0.05), (200, 194, 0.9)],
)
def test_betainc_equals_the_exact_binomial_tail(n, k, p):
    # I_p(k, n - k + 1) = P(Bin(n, p) >= k) is the identity the whole
    # Clopper-Pearson construction rests on. If this drifts, every bound in
    # the package is silently wrong by an unknown amount.
    got = float(_betainc(np.array([float(k)]), np.array([float(n - k + 1)]), np.array([p]))[0])
    assert got == pytest.approx(_binomial_tail(n, k, p), abs=1e-12)


def test_clopper_pearson_matches_its_closed_forms():
    # Three cases where the bound is available in closed form, so the
    # bisection is checked against algebra rather than against itself.
    assert float(clopper_pearson_lower(0, 50, 0.05)) == 0.0
    assert float(clopper_pearson_lower(1, 1, 0.05)) == pytest.approx(0.05, abs=1e-9)
    assert float(clopper_pearson_lower(30, 30, 0.05)) == pytest.approx(0.05 ** (1 / 30), abs=1e-9)
    assert float(clopper_pearson_upper(50, 50, 0.05)) == 1.0
    assert float(clopper_pearson_upper(0, 30, 0.05)) == pytest.approx(1 - 0.05 ** (1 / 30), abs=1e-9)


def test_bounds_are_monotone_in_k_and_ordered():
    k = np.arange(0, 51)
    n = np.full(51, 50)
    lower = clopper_pearson_lower(k, n, 0.05)
    upper = clopper_pearson_upper(k, n, 0.05)
    assert np.all(np.diff(lower) >= -1e-12)
    assert np.all(np.diff(upper) >= -1e-12)
    assert np.all(lower <= k / n + 1e-12)
    assert np.all(upper >= k / n - 1e-12)


# ---------------------------------------------------------------------------
# 2. Why Clopper-Pearson and not Wilson
# ---------------------------------------------------------------------------
def test_wilson_undercovers_where_clopper_pearson_does_not():
    # REGRESSION WITNESS for the choice of bound. "Use Wilson, it is the
    # modern recommendation" is a real and superficially convincing
    # counter-proposal; it is correct for a bound whose AVERAGE coverage
    # matters and wrong for one that certifies a published claim.
    grid = np.linspace(0.01, 0.99, 199)
    cp = _lower_bound_coverage(30, 0.05, clopper_pearson_lower, grid)
    wilson = _lower_bound_coverage(30, 0.05, wilson_lower, grid)
    assert cp.min() >= 0.95
    assert wilson.min() < 0.90
    assert wilson.min() == pytest.approx(0.8805, abs=5e-4)


# ---------------------------------------------------------------------------
# 3. The two certification routes agree
# ---------------------------------------------------------------------------
def test_bound_route_and_exact_tail_route_certify_the_same_cells():
    rng = np.random.default_rng(0)
    n = rng.integers(1, 400, size=800)
    k = (rng.random(800) * n).astype(np.int64)
    spec = CenterSpec(tau=0.9, alpha=0.05, rule="certified")
    mask, alpha_eff = certified_centers(k, n, spec)
    by_bound = clopper_pearson_lower(k, n, alpha_eff) >= spec.tau
    by_tail = k >= min_successes_to_certify(n, spec.tau, alpha_eff)
    assert np.array_equal(mask, by_bound)
    assert np.array_equal(mask, by_tail)


def test_min_successes_never_exceeds_n_plus_one_and_is_monotone_in_alpha():
    n = np.array([1, 5, 10, 30, 100, 300])
    loose = min_successes_to_certify(n, 0.9, 0.05)
    tight = min_successes_to_certify(n, 0.9, 1e-4)
    assert np.all(loose <= n + 1)
    assert np.all(tight >= loose)


# ---------------------------------------------------------------------------
# 4. Multiplicity control is load-bearing, and by how much
# ---------------------------------------------------------------------------
def test_bonferroni_suppresses_the_false_certifications_the_naive_bound_makes():
    # Every cell sits EXACTLY at the null boundary pi = tau, i.e. H_0 is true
    # everywhere and no cell deserves certification. Expected counts computed
    # exactly from the binomial law, not simulated.
    n_cell, tau, n_cells = 30, 0.9, 200
    expected = {}
    for policy, alpha_eff in (("none", 0.05), ("bonferroni", 0.05 / n_cells)):
        k_min = int(min_successes_to_certify(np.array([n_cell]), tau, alpha_eff)[0])
        expected[policy] = _binomial_tail(n_cell, k_min, tau) * n_cells
    assert expected["none"] == pytest.approx(8.48, abs=0.05)
    assert expected["bonferroni"] < 0.05


def test_effective_alpha_divides_by_the_occupied_cell_count():
    spec = CenterSpec(alpha=0.05)
    assert spec.effective_alpha(200) == pytest.approx(0.05 / 200)
    assert CenterSpec(alpha=0.05, multiplicity="none").effective_alpha(200) == 0.05


# ---------------------------------------------------------------------------
# 5. A singleton cell is not a centre
# ---------------------------------------------------------------------------
def test_the_purity_rule_admits_a_single_sample_cell_and_reports_its_interval():
    # Deliberate, and the caller's decision: with min_samples = 1 a cell
    # holding one target-value object IS a centre under the default rule.
    # What must never happen is that the weakness is invisible -- the cell's
    # reported interval is [alpha, 1], i.e. the data are equally consistent
    # with a purity of 5 %.
    report = center_report(
        np.array([1] + [0] * 99, dtype=np.int8),
        np.concatenate([[0], np.repeat(np.arange(1, 11), 10)[:99]]),
        11,
        CenterSpec(tau=0.9),
    )
    assert report.n_centers == 1
    only = report.centers[0]
    assert only.n == 1 and only.k == 1 and only.purity == 1.0
    assert only.purity_lower == pytest.approx(report.alpha_effective, abs=1e-9)
    assert only.purity_upper == 1.0
    # And min_samples removes it without touching anything else.
    tightened = center_report(
        np.array([1] + [0] * 99, dtype=np.int8),
        np.concatenate([[0], np.repeat(np.arange(1, 11), 10)[:99]]),
        11,
        CenterSpec(tau=0.9, min_samples=2),
    )
    assert tightened.n_centers == 0


def test_a_single_sample_cell_can_never_be_certified():
    # REGRESSION WITNESS for removing the "Noise Reduction (minimum samples)"
    # control. A cell with k = n = 1 has a point purity of 1.0 and drew as a
    # saturated green centre in v2.1; its Clopper-Pearson lower bound is
    # alpha itself, so no (tau >= alpha, alpha) can certify it.
    for alpha in (0.05, 0.1, 0.2):
        assert float(clopper_pearson_lower(1, 1, alpha)) == pytest.approx(alpha, abs=1e-9)
        mask, _ = select_centers(
            np.array([1]), np.array([1]),
            CenterSpec(tau=0.9, alpha=alpha, rule="certified"),
        )
        assert not mask.any()
    # And the smallest fully-pure cell that CAN be certified at tau = 0.9,
    # alpha = 0.05 with a single occupied cell is n = 29 (0.05^(1/n) >= 0.9).
    sizes = np.arange(1, 40)
    k_min = min_successes_to_certify(sizes, 0.9, 0.05)
    certifiable_pure = sizes[k_min <= sizes]
    assert int(certifiable_pure.min()) == 29


# ---------------------------------------------------------------------------
# 6. Coverage: determinism, noise, and out-of-sample optimism
# ---------------------------------------------------------------------------
def test_coverage_is_one_under_a_deterministic_relation():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 50, size=20000)
    z = (codes < 10).astype(np.int8)
    report = center_report(z, codes, 50, CenterSpec(tau=0.9, alpha=0.05))
    assert report.n_centers == 10
    assert report.coverage == pytest.approx(1.0)
    assert report.purity_pooled == pytest.approx(1.0)
    assert report.coverage_cv is not None
    assert report.coverage_cv.mean == pytest.approx(1.0)


def test_pure_noise_at_high_cardinality_measures_the_cost_of_min_samples_1():
    """
    The case that breaks a raw-MI ranking (`vsf.avr`'s module docstring):
    3000 cells over 32 561 rows, target independent of every one of them.

    This test does NOT assert that the default rule returns zero centres. It
    measures what it actually returns, because that is the price of
    `rule="purity", min_samples=1` and the caller is entitled to see it
    rather than be told it does not exist: with one-object cells eligible, a
    handful of them come up entirely target-valued by chance.

    What must hold is that the price is small, that the permutation test does
    not mistake it for a finding, and that both of the available remedies
    drive it to exactly zero.
    """
    rng = np.random.default_rng(7)
    n = 32561
    z = (rng.random(n) < 0.30).astype(np.int8)
    codes = rng.integers(0, 3000, size=n)

    loose = center_report(z, codes, 3000, CenterSpec(tau=0.9), n_permutations=299)
    assert loose.n_centers == 2
    assert loose.coverage == pytest.approx(0.0009, abs=0.0002)
    # The whole point: this is NOT significant, and the panel's p-value says so.
    assert loose.coverage_p_value > 0.05

    for remedy in (CenterSpec(tau=0.9, min_samples=20), CenterSpec(tau=0.9, rule="certified")):
        strict = center_report(z, codes, 3000, remedy, n_permutations=299)
        assert strict.n_centers == 0, remedy
        assert strict.coverage == 0.0
        assert strict.coverage_cv is not None and strict.coverage_cv.mean == 0.0
        assert strict.coverage_null_mean == 0.0
        assert strict.coverage_p_value == 1.0


def test_crossvalidated_coverage_is_below_the_in_sample_value_when_cells_are_cherry_picked():
    # The property that makes `coverage_cv` the statistic behind
    # `select_dimensionality`: cells are certified by looking at their own
    # contents, so the in-sample coverage is optimistic even though every
    # individual certificate is honest. True purities straddle tau, so the
    # certified set is exactly the set that fluctuated upward.
    rng = np.random.default_rng(11)
    n_cells, per_cell = 40, 300
    pis = rng.uniform(0.85, 0.95, n_cells)
    codes = np.repeat(np.arange(n_cells), per_cell)
    z = (rng.random(n_cells * per_cell) < np.repeat(pis, per_cell)).astype(np.int8)
    report = center_report(z, codes, n_cells, CenterSpec(tau=0.9))
    assert report.n_centers > 0
    assert report.coverage_cv is not None
    assert report.coverage_cv.mean < report.coverage
    # Pinned so a change in the CV protocol has to be deliberate.
    assert report.coverage == pytest.approx(0.5894, abs=0.005)
    assert report.coverage_cv.mean == pytest.approx(0.5472, abs=0.005)

    # Under the conservative rule the same data give a smaller in-sample
    # coverage and a smaller gap, which is what "conservative" means here.
    certified = center_report(z, codes, n_cells, CenterSpec(tau=0.9, rule="certified"))
    assert certified.coverage < report.coverage


def test_micro_class_is_undetermined_rather_than_zero():
    # A coverage of 0% measured on 9 positives and one measured on 900 are
    # different claims; the report must not present them identically.
    rng = np.random.default_rng(5)
    n = 32561
    z = np.zeros(n, dtype=np.int8)
    z[rng.choice(n, MIN_POSITIVES_FOR_CV - 1, replace=False)] = 1
    report = center_report(z, rng.integers(0, 300, size=n), 300, CenterSpec())
    assert report.is_undetermined
    assert report.coverage_cv is None
    assert str(MIN_POSITIVES_FOR_CV) in report.undetermined_reason


def test_pooled_purity_bound_is_reported_and_conservative():
    rng = np.random.default_rng(17)
    codes = np.repeat(np.arange(20), 500)
    z = np.zeros(10000, dtype=np.int8)
    z[codes < 5] = (rng.random(int((codes < 5).sum())) < 0.97).astype(np.int8)
    report = center_report(z, codes, 20, CenterSpec(tau=0.9, alpha=0.05))
    assert report.n_centers == 5
    assert report.purity_pooled_lower <= report.purity_pooled
    assert report.coverage_lower <= report.coverage
    assert report.lift > 1.0


# ---------------------------------------------------------------------------
# 7. Nulls
# ---------------------------------------------------------------------------
def test_coverage_null_holds_both_margins_fixed():
    rng = np.random.default_rng(2)
    n_per_cell = rng.integers(20, 200, size=60)
    scores = coverage_null(n_per_cell, 400, CenterSpec(tau=0.9, alpha=0.05), 199, 0)
    assert scores.shape == (199,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_familywise_coverage_null_is_never_anti_conservative():
    # The look-elsewhere correction can only make a p-value larger: the
    # maximum over a family dominates any single member of it.
    rng = np.random.default_rng(23)
    n = 4000
    z = (rng.random(n) < 0.4).astype(np.int8)
    candidates_codes = [rng.integers(0, 12, size=n) for _ in range(6)]
    spec = CenterSpec(tau=0.9, alpha=0.05)
    observed = max(coverage_score(z, c, 12, spec)[0] for c in candidates_codes)
    fw = familywise_max_coverage_null(
        z, lambda: ((i, c, 12) for i, c in enumerate(candidates_codes)),
        spec=spec, n_permutations=199, random_state=0,
    )
    assert fw.n_candidates == 6
    single = coverage_null(
        np.bincount(candidates_codes[0], minlength=12), int(z.sum()), spec, 199, 0
    )
    single_p = float((1 + int(np.sum(single >= observed - 1e-12))) / 200)
    assert fw.p_value(observed) >= single_p - 1e-12


# ---------------------------------------------------------------------------
# 8. Cross-validation machinery
# ---------------------------------------------------------------------------
def test_stratified_folds_partition_the_rows_and_balance_the_classes():
    rng = np.random.default_rng(1)
    z = (rng.random(1000) < 0.07).astype(np.int8)
    splits = stratified_repeated_kfold(z, n_splits=5, n_repeats=3, random_state=0)
    assert len(splits) == 15
    for train, test in splits:
        assert set(train.tolist()) | set(test.tolist()) == set(range(1000))
        assert not (set(train.tolist()) & set(test.tolist()))
    per_fold_positives = [int(z[test].sum()) for _, test in splits[:5]]
    assert max(per_fold_positives) - min(per_fold_positives) <= 1


def test_folds_are_a_function_of_the_target_alone_so_branches_are_paired():
    z = np.array([0, 1] * 500, dtype=np.int8)
    a = stratified_repeated_kfold(z, 5, 2, random_state=0)
    b = stratified_repeated_kfold(z, 5, 2, random_state=0)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
        assert np.array_equal(tr_a, tr_b) and np.array_equal(te_a, te_b)


def test_nadeau_bengio_standard_error_exceeds_the_naive_one():
    rng = np.random.default_rng(31)
    codes = rng.integers(0, 30, size=6000)
    z = (rng.random(6000) < 0.5).astype(np.int8)
    cv = crossvalidated_coverage(z, codes, 30, CenterSpec(), n_splits=5, n_repeats=5)
    per_split = np.asarray(cv.per_split, dtype=float)
    naive = float(per_split.std(ddof=1) / np.sqrt(per_split.size)) if per_split.size > 1 else 0.0
    assert cv.se >= naive
    assert cv.test_train_ratio == pytest.approx(0.25)


def test_paired_gain_requires_matched_folds():
    a = CVCoverage(per_split=(0.1, 0.2), mean=0.15, se=0.01, purity_mean=0.9,
                   n_splits=2, n_repeats=1, test_train_ratio=1.0)
    b = CVCoverage(per_split=(0.1, 0.2, 0.3), mean=0.2, se=0.01, purity_mean=0.9,
                   n_splits=3, n_repeats=1, test_train_ratio=0.5)
    with pytest.raises(ValueError):
        paired_gain(a, b)


# ---------------------------------------------------------------------------
# 9. Dimensionality selection
# ---------------------------------------------------------------------------
def _cv(values, ratio: float = 0.25) -> CVCoverage:
    arr = np.asarray(values, dtype=float)
    var = float(arr.var(ddof=1)) if arr.size > 1 else 0.0
    se = float(np.sqrt(max(0.0, (1.0 / arr.size + ratio) * var)))
    return CVCoverage(
        per_split=tuple(arr.tolist()), mean=float(arr.mean()), se=se,
        purity_mean=1.0, n_splits=5, n_repeats=arr.size // 5,
        test_train_ratio=ratio,
    )


def test_select_dimensionality_returns_none_when_nothing_certifies():
    zeros = _cv([0.0] * 25)
    assert select_dimensionality({1: zeros, 2: zeros, 3: zeros}) is None


def test_select_dimensionality_returns_the_smallest_sufficient_not_the_first_useful():
    rng = np.random.default_rng(4)
    d1 = _cv(rng.normal(0.30, 0.01, 25))
    d2 = _cv(rng.normal(0.60, 0.01, 25))
    d3 = _cv(rng.normal(0.61, 0.01, 25))
    # d2 genuinely beats d1, so one characteristic does not describe the
    # target; d3 does not beat d2, so the third is not needed.
    assert select_dimensionality({1: d1, 2: d2, 3: d3}) == 2


def test_select_dimensionality_does_not_stop_at_a_flat_step():
    # Coverage is not submodular: a pair of axes can isolate a cell neither
    # axis produces alone, so a flat step from d to d+1 must not terminate
    # the scan at d.
    rng = np.random.default_rng(9)
    d1 = _cv(rng.normal(0.50, 0.01, 25))
    d2 = _cv(rng.normal(0.50, 0.01, 25))
    d3 = _cv(rng.normal(0.90, 0.01, 25))
    assert select_dimensionality({1: d1, 2: d2, 3: d3}) == 3


def test_select_dimensionality_rejects_a_first_candidate_that_is_only_fold_noise():
    # One fold out of 25 certified a centre and the other 24 found nothing.
    # The mean is positive (4%), and a naive "coverage > 0" rule would report
    # d = 1 as the answer. Against the Nadeau-Bengio corrected spread the
    # statistic is t = 0.38, so it is not distinguishable from zero.
    spiky = _cv([1.0] + [0.0] * 24)
    assert spiky.mean == pytest.approx(0.04)
    assert spiky.mean / spiky.se < 2.0
    assert select_dimensionality({1: spiky}) is None


# ---------------------------------------------------------------------------
# 10. Spec validation
# ---------------------------------------------------------------------------
def test_tau_of_one_is_allowed_by_the_purity_rule_and_rejected_by_the_certified_rule():
    # "Cells that are entirely the target value" is a well-posed request
    # about the observed table, so rule="purity" accepts tau = 1.0 and
    # selects exactly the fully pure cells.
    spec = CenterSpec(tau=1.0)
    mask, _ = select_centers(np.array([10, 9, 1]), np.array([10, 10, 1]), spec)
    assert mask.tolist() == [True, False, True]
    # H_0: pi <= 1 is never rejectable, so a request to CERTIFY exact purity
    # has no valid answer. Clamping it to 0.999 would answer a different
    # question than the one asked.
    with pytest.raises(ValueError, match="not certifiable"):
        CenterSpec(tau=1.0, rule="certified")
    with pytest.raises(ValueError):
        CenterSpec(tau=0.0)
    with pytest.raises(ValueError):
        CenterSpec(min_samples=0)
    with pytest.raises(ValueError):
        CenterSpec(rule="strict")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CenterSpec(alpha=0.0)
    with pytest.raises(ValueError):
        CenterSpec(method="jeffreys")  # type: ignore[arg-type]


def test_binarize_target_requires_an_explicit_positive_value():
    z = binarize_target(np.array(["a", "b", "c", "b"]), "b")
    assert z.tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError):
        center_report(np.array([0, 1, 2]), np.array([0, 0, 1]), 2)


# ---------------------------------------------------------------------------
# 11. REGRESSION WITNESS on the reference dataset
# ---------------------------------------------------------------------------
_ADULT = Path(__file__).resolve().parent.parent / "data" / "adult_census.csv"


@pytest.mark.skipif(not _ADULT.exists(), reason="reference dataset not present")
def test_armed_forces_branch_has_no_certifiable_centre_despite_a_high_historical_association():
    # THE claim this module exists for (see `vsf.centers`'s module
    # docstring). `occupation = Armed-Forces` is 9 of 32 561 rows. Under the
    # v2.1/v2.2 bias-corrected association statistic -- since deleted along
    # with the rest of the MI-ranking machinery (v2.3, see `vsf.avr`'s
    # module docstring) -- this same 3-D branch read U_adj = 41.3%,
    # arithmetically correct and operationally empty: the highest purity of
    # any of its 32 cells is 2.42% (8 of 330), so no colour threshold could
    # make the display green and no decision rule could use it. This test
    # pins the half of that claim vsf.centers still can: the coverage layer
    # correctly reports nothing certifiable, regardless of what an
    # association statistic would have said.
    import pandas as pd

    from vsf.metrics import cell_codes

    df = pd.read_csv(_ADULT)
    z = binarize_target(df["occupation"].values, "Armed-Forces")
    features = df[["workclass", "sex", "income"]].astype(str).values
    assert int(z.sum()) == 9

    codes, n_cells = cell_codes(features)
    report = center_report(z, codes, n_cells, CenterSpec(tau=0.9, alpha=0.05), n_permutations=99)
    assert report.n_cells_occupied == 32
    assert report.max_purity_point == pytest.approx(8 / 330, abs=1e-9)
    assert report.n_centers == 0
    assert report.coverage == 0.0
    assert report.is_undetermined  # 9 positives: no out-of-sample statement


@pytest.mark.skipif(not _ADULT.exists(), reason="reference dataset not present")
def test_a_target_with_real_centres_reports_them():
    # The complement of the witness above: the same machinery must not be
    # merely conservative. `relationship = Husband` is captured almost
    # entirely by one cell of `marital_status x sex`.
    import pandas as pd

    from vsf.metrics import cell_codes

    df = pd.read_csv(_ADULT)
    z = binarize_target(df["relationship"].values, "Husband")
    codes, n_cells = cell_codes(df[["marital_status", "sex"]].astype(str).values)
    report = center_report(z, codes, n_cells, CenterSpec(tau=0.9))
    assert report.n_centers == 2
    assert report.coverage > 0.99
    assert report.purity_pooled > 0.98
    assert report.coverage_cv is not None
    assert report.coverage_cv.mean > 0.99
    # The conservative rule keeps only the large cell and loses nothing that
    # matters: one centre, the same coverage to three decimals.
    strict = center_report(z, codes, n_cells, CenterSpec(tau=0.9, rule="certified"))
    assert strict.n_centers == 1
    assert strict.coverage == pytest.approx(report.coverage, abs=1e-3)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
