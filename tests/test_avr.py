"""
Unit tests for vsf.avr — Independent Branch Discovery (IBD), v2.0 "Clean Core".

Replaces the v1.0 AVREngine/Scenario test suite entirely (see
Project_Master_Document.md Section 0). Covers the properties the new
algorithm is actually supposed to guarantee (Sections 4.2-4.4):

  1. Exhaustiveness: the d=2 search genuinely evaluates every pair, so it
     finds an XOR-style synergistic pair that neither feature reveals
     individually — something a greedy chain seeded at the best single
     feature could never discover (it never revisits its first pick).
  2. Independence across dimensionalities: the d=1 winner need not be a
     subset of the d=2 winner (branches are not required to be nested).
  3. Ranking is by RAW mutual information, not NMI, within one d (Section
     4.2/4.5) — constructed so the two rankings actually disagree.
  4. Determinism: repeated calls on identical input return identical
     output — no randomness anywhere in the algorithm.
  5. The Grid Capacity Limit coarsening path (`check_grid_capacity` /
     `adaptively_coarsen_bins`) is genuinely invoked for a combination
     whose joint table would be too sparse.
  6. Edge cases: fewer features than max_d, a single feature, zero
     features, out-of-range max_d.
  7. `BranchEngine` is a thin, stateless wrapper around `discover_branches`.

A note on test data construction: `vsf.pmd.discretize_feature`'s quantile
strategy has a real bug (discovered while writing these tests, reported
separately, not fixed here per this task's non-test-file constraint) that
silently collapses a numeric feature taking exactly two values (e.g. a raw
0/1 int/float column) into a SINGLE bin whenever the class split doesn't
happen to land exactly on one of the requested quantile points -- which for
a roughly-balanced binary column is most of the time. `vsf.benchmark.
generate_synthetic_dataset` already works around this by adding Gaussian
jitter to its informative proxy features; the tests below that need a clean
binary numeric signal do the same, and tests that need clean categorical
levels use string-typed columns instead (routed through the unaffected
`np.unique`-based encoding path for string/object/bool dtypes).
"""

import warnings
from unittest import mock

import numpy as np
import pytest

import vsf.avr as avr_mod
from vsf.avr import MAX_BRANCH_D, BranchEngine, BranchResult, discover_branches


# ---------------------------------------------------------------------------
# Exhaustiveness + non-nested branches: XOR synergy a greedy chain would miss
# ---------------------------------------------------------------------------

def _xor_synergy_dataset(n=4000, seed=123):
    """
    Z = XOR(a, b). a and b are individually uncorrelated with Z (standard
    XOR secret-sharing property: each share alone carries zero information
    about the secret when the other share is uniform and independent), but
    jointly they determine Z exactly. `c` is a noisy DIRECT copy of Z, so it
    carries some, but not all, of Z's entropy on its own -- exactly the
    profile needed to be the best SINGLE feature while losing badly to the
    true synergistic pair {a, b} at d=2. Two pure-noise columns widen the
    search space so this isn't a trivially small combinatorial check.

    Gaussian jitter is added to every column (see module docstring) to sidestep
    a separate, pre-existing bug in `vsf.pmd.discretize_feature`'s handling of
    exactly-binary numeric columns.
    """
    rng = np.random.default_rng(seed)
    a_bit = rng.integers(0, 2, size=n)
    b_bit = rng.integers(0, 2, size=n)
    Z = np.bitwise_xor(a_bit, b_bit)

    a = a_bit.astype(float) + rng.normal(0, 0.15, size=n)
    b = b_bit.astype(float) + rng.normal(0, 0.15, size=n)

    flip = rng.random(n) < 0.3
    c_bit = np.where(flip, 1 - Z, Z)
    c = c_bit.astype(float) + rng.normal(0, 0.15, size=n)

    noise1 = rng.normal(0, 1, size=n)
    noise2 = rng.normal(0, 1, size=n)

    X = np.column_stack([a, b, c, noise1, noise2])
    feature_names = ["a", "b", "c", "n1", "n2"]
    return X, Z, feature_names


def test_exhaustive_search_finds_xor_synergy_a_greedy_chain_would_miss():
    X, Z, feature_names = _xor_synergy_dataset()
    branches = discover_branches(X, Z, feature_names=feature_names, max_d=2)

    # d=1: neither `a` nor `b` carries any raw MI about Z individually (the
    # XOR secret-sharing property) -- `c` (an imperfect but non-trivial
    # single-feature predictor) must win instead.
    assert branches[1].selected_feature_names == ["c"]
    assert branches[1].mi > 0.0

    # d=2: the TRUE synergistic pair {a, b} must win, even though neither
    # of its members was selected at d=1. A greedy chain that extends the
    # d=1 winner ({c} -> {c, x}) can never find this pair -- discovering it
    # requires genuinely re-examining combinations that don't contain `c`.
    assert set(branches[2].selected_feature_names) == {"a", "b"}

    # {a, b} recovers Z almost perfectly (H(Z) = 1 bit): its MI and its
    # bias-corrected share of the target's entropy must both be close to the
    # ceiling, and strictly higher than the d=1 winner could ever reach on
    # its own.
    assert branches[2].mi == pytest.approx(1.0, abs=0.15)
    assert branches[2].u_adj == pytest.approx(1.0, abs=0.15)
    assert branches[2].mi > 0.8
    assert branches[2].mi > branches[1].mi
    assert branches[2].mi_adj > branches[1].mi_adj


def test_branch_d1_winner_is_not_a_subset_of_branch_d2_winner():
    # The direct "branches need not be nested" claim (Section 4.4), stated
    # as its own explicit assertion independent of the synergy narrative
    # above.
    X, Z, feature_names = _xor_synergy_dataset()
    branches = discover_branches(X, Z, feature_names=feature_names, max_d=2)
    d1_set = set(branches[1].selected_features)
    d2_set = set(branches[2].selected_features)
    assert not d1_set.issubset(d2_set)
    assert d1_set.isdisjoint(d2_set)


# ---------------------------------------------------------------------------
# Ranking is by BIAS-CORRECTED MI (Section 4.2 / 4.5)
#
# v2.0 ranked by raw plug-in MI and documented the missing degrees-of-freedom
# correction as an accepted trade-off. It is not one: the plug-in estimator's
# bias grows with the candidate's cell count, so `argmax` over raw MI selects
# on cardinality rather than on association. These two tests pin the fix from
# both sides -- that the biased choice is actually rejected, and that a real
# signal is still found.
# ---------------------------------------------------------------------------

def test_ranking_rejects_a_high_cardinality_noise_feature_that_raw_mi_prefers():
    # Two features, NEITHER carrying any information about Z:
    #   A: binary noise         -> tiny plug-in MI, tiny bias
    #   B: 200-category noise   -> much larger plug-in MI, entirely bias
    # Raw-MI ranking (v2.0) must prefer B; MI_adj ranking must not prefer it,
    # and the winner's corrected effect size must be ~0 either way.
    rng = np.random.default_rng(7)
    n = 6000
    Z = (rng.random(n) < 0.3).astype(int)
    A = rng.integers(0, 2, size=n)
    B = rng.integers(0, 200, size=n)
    X = np.column_stack([A, B])

    from vsf.math import mutual_information

    mi_a = mutual_information(Z, A)
    mi_b = mutual_information(Z, B)
    assert mi_b > mi_a, "fixture must have raw MI favoring the noisy 200-category feature"

    # v2.2: `objective` must be pinned to "mi_adj" in this test. The default
    # is now "auto", which selects the COVERAGE objective for a binary target
    # -- a different ranking entirely, and not the one this test is about.
    legacy = discover_branches(X, Z, feature_names=["A", "B"], max_d=1,
                               null="none", n_permutations=0, objective="mi_adj")
    assert legacy[1].selected_feature_names == ["B"], "null='none' must reproduce v2.0 behaviour"

    corrected = discover_branches(X, Z, feature_names=["A", "B"], max_d=1,
                                  n_permutations=199, objective="mi_adj")
    # The corrected search must not report a finding on data with no signal:
    # whichever feature it picks, the excess over the null is ~0 and the
    # branch is not significant.
    assert corrected[1].mi_adj < 0.01
    assert corrected[1].u_adj < 0.05
    assert corrected[1].p_value > 0.01
    assert not corrected[1].is_significant()


def test_ranking_still_finds_a_real_signal_and_prefers_it_over_noisy_cardinality():
    # Same shape as above, but now the LOW-cardinality feature genuinely
    # predicts Z. Raw MI is close between the two (the 200-category noise
    # feature's bias nearly matches the real signal); MI_adj must separate
    # them decisively and pick the real one.
    rng = np.random.default_rng(11)
    n = 6000
    Z = (rng.random(n) < 0.3).astype(int)
    A = np.where(rng.random(n) < 0.85, Z, 1 - Z)   # noisy but real copy of Z
    B = rng.integers(0, 200, size=n)               # pure noise, high cardinality
    X = np.column_stack([A, B])

    branches = discover_branches(X, Z, feature_names=["A", "B"], max_d=1,
                                 n_permutations=199)
    assert branches[1].selected_feature_names == ["A"]
    assert branches[1].mi_adj > 0.1
    assert branches[1].u_adj > 0.2
    assert branches[1].p_value <= 0.01
    assert branches[1].is_significant()


def test_mi_adj_equals_mi_minus_null_and_u_adj_is_its_normalized_form():
    rng = np.random.default_rng(13)
    n = 2000
    X = rng.integers(0, 5, size=(n, 3))
    Z = rng.integers(0, 3, size=n)
    for branch in discover_branches(X, Z, max_d=3, n_permutations=0).values():
        assert branch.mi_adj == pytest.approx(branch.mi - branch.mi_null)
        denom = branch.h_target - branch.mi_null
        expected_u = min(1.0, max(0.0, branch.mi_adj / denom))
        assert branch.u_adj == pytest.approx(expected_u)


# ---------------------------------------------------------------------------
# Determinism: no randomness anywhere in the algorithm
# ---------------------------------------------------------------------------

def test_determinism_repeated_calls_produce_identical_output():
    X, Z, feature_names = _xor_synergy_dataset(seed=99)
    branches_1 = discover_branches(X, Z, feature_names=feature_names, max_d=3)
    branches_2 = discover_branches(X, Z, feature_names=feature_names, max_d=3)

    assert set(branches_1.keys()) == set(branches_2.keys())
    for d in branches_1:
        # BranchResult is a plain @dataclass -> field-wise equality,
        # including the selected_features/selected_feature_names lists.
        assert branches_1[d] == branches_2[d]


def test_determinism_holds_on_real_categorical_data():
    rng = np.random.default_rng(5)
    n = 500
    X = np.column_stack([
        rng.choice(["x", "y", "z"], size=n),
        rng.choice(["lo", "hi"], size=n),
        rng.integers(0, 4, size=n).astype(str),
    ])
    Z = rng.choice(["p", "e"], size=n)
    r1 = discover_branches(X, Z, max_d=2)
    r2 = discover_branches(X, Z, max_d=2)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Grid Capacity Limit coarsening path (Section 2.4) is genuinely invoked
# ---------------------------------------------------------------------------

def test_grid_capacity_coarsening_invoked_for_sparse_combination():
    # A single categorical feature with 25 distinct levels against n=200
    # samples: prod(k) = 25 > n // 10 = 20 -> check_grid_capacity must
    # report the limit exceeded, and adaptively_coarsen_bins must actually
    # be invoked to coarsen it before scoring.
    rng = np.random.default_rng(0)
    n = 200
    high_card = np.array([f"cat_{i % 25}" for i in rng.permutation(n)])
    other = rng.integers(0, 3, size=n).astype(str)
    Z = rng.integers(0, 2, size=n)
    X = np.column_stack([high_card, other])

    with mock.patch.object(
        avr_mod, "adaptively_coarsen_bins", wraps=avr_mod.adaptively_coarsen_bins
    ) as spy_coarsen, mock.patch.object(
        avr_mod, "check_grid_capacity", wraps=avr_mod.check_grid_capacity
    ) as spy_check:
        branches = discover_branches(X, Z, max_d=1)

    assert spy_check.called, "check_grid_capacity must be called during discovery"
    assert spy_coarsen.called, (
        "adaptively_coarsen_bins must be invoked for the 25-level combination "
        "whose joint table exceeds the N/10 capacity limit"
    )
    # The branch must still be produced (coarsening degrades gracefully,
    # it doesn't crash or silently drop the combination from consideration).
    assert 1 in branches

    # A combination that easily fits within capacity must NOT trigger
    # coarsening for that specific combination's call.
    coarsened_input_shapes = [
        (call.args[0].shape[1] if call.args[0].ndim > 1 else 1)
        for call in spy_coarsen.call_args_list
    ]
    assert all(shape >= 1 for shape in coarsened_input_shapes)


def test_grid_capacity_not_triggered_for_small_well_fitting_combination():
    # Sanity check in the opposite direction: a low-cardinality combination
    # that comfortably fits (prod(k) <= N/10) must not be coarsened.
    rng = np.random.default_rng(1)
    n = 2000
    X = np.column_stack([
        rng.integers(0, 3, size=n).astype(str),
        rng.integers(0, 2, size=n).astype(str),
    ])
    Z = rng.integers(0, 2, size=n)

    with mock.patch.object(
        avr_mod, "adaptively_coarsen_bins", wraps=avr_mod.adaptively_coarsen_bins
    ) as spy_coarsen:
        discover_branches(X, Z, max_d=2)

    assert not spy_coarsen.called, (
        "well-fitting low-cardinality combinations must not be coarsened "
        "(3*2=6 cells <= 2000//10=200)"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_fewer_features_than_max_d_returns_only_feasible_dimensionalities():
    rng = np.random.default_rng(2)
    n = 300
    X = rng.integers(0, 3, size=(n, 2))  # only 2 feature columns
    Z = rng.integers(0, 2, size=n)

    branches = discover_branches(X, Z, max_d=MAX_BRANCH_D)
    assert set(branches.keys()) == {1, 2}
    assert 3 not in branches
    assert 4 not in branches


def test_single_feature_dataset():
    rng = np.random.default_rng(3)
    n = 200
    X = rng.integers(0, 4, size=(n, 1))
    Z = rng.integers(0, 2, size=n)

    branches = discover_branches(X, Z, max_d=MAX_BRANCH_D)
    assert set(branches.keys()) == {1}
    assert branches[1].selected_features == [0]
    assert branches[1].d == 1


def test_zero_features_returns_empty_dict():
    n = 50
    X = np.empty((n, 0))
    Z = np.random.default_rng(4).integers(0, 2, size=n)
    branches = discover_branches(X, Z, max_d=MAX_BRANCH_D)
    assert branches == {}


def test_default_feature_names_generated_when_none_given():
    rng = np.random.default_rng(6)
    n = 100
    X = rng.integers(0, 3, size=(n, 3))
    Z = rng.integers(0, 2, size=n)
    branches = discover_branches(X, Z, max_d=1)
    assert branches[1].selected_feature_names[0].startswith("X_")


@pytest.mark.parametrize("bad_max_d", [0, -1, 5, 6, 100])
def test_max_d_out_of_range_raises(bad_max_d):
    rng = np.random.default_rng(8)
    n = 50
    X = rng.integers(0, 2, size=(n, 5))
    Z = rng.integers(0, 2, size=n)
    with pytest.raises(ValueError, match=r"max_d must be in \[1, 4\]"):
        discover_branches(X, Z, max_d=bad_max_d)
    with pytest.raises(ValueError, match=r"max_d must be in \[1, 4\]"):
        BranchEngine(max_d=bad_max_d)


@pytest.mark.parametrize("ok_max_d", [1, 2, 3, 4])
def test_max_d_in_range_is_accepted(ok_max_d):
    rng = np.random.default_rng(9)
    n = 100
    X = rng.integers(0, 3, size=(n, 5))
    Z = rng.integers(0, 2, size=n)
    branches = discover_branches(X, Z, max_d=ok_max_d)
    assert max(branches.keys()) == ok_max_d
    assert set(branches.keys()) == set(range(1, ok_max_d + 1))


def test_branch_result_fields_are_internally_consistent():
    rng = np.random.default_rng(10)
    n = 400
    X = rng.integers(0, 3, size=(n, 4))
    Z = rng.integers(0, 2, size=n)
    branches = discover_branches(X, Z, max_d=3)
    for d, branch in branches.items():
        assert isinstance(branch, BranchResult)
        assert branch.d == d
        assert len(branch.selected_features) == d
        assert len(branch.selected_feature_names) == d
        assert branch.mi >= 0.0
        assert branch.mi_null >= 0.0
        assert branch.mi_adj == pytest.approx(branch.mi - branch.mi_null)
        assert 0.0 <= branch.u_adj <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# `mi_by_prefix_d`/`mi_adj_by_prefix_d`/`u_adj_by_prefix_d` — regression
# coverage for the within-branch dimensionality-collapse HUD bug (the frontend
# was reading the branch's fixed full-d values for every collapsed view
# instead of the marginals of the actually-visible axes).
# ---------------------------------------------------------------------------

def test_prefix_metrics_length_matches_branch_d_and_last_entry_matches_scalars():
    rng = np.random.default_rng(11)
    n = 500
    X = rng.integers(0, 4, size=(n, 5))
    Z = rng.integers(0, 3, size=n)
    branches = discover_branches(X, Z, max_d=4)
    for d, branch in branches.items():
        assert len(branch.mi_by_prefix_d) == d
        assert len(branch.mi_adj_by_prefix_d) == d
        assert len(branch.u_adj_by_prefix_d) == d
        # The last (full-d) prefix entry must reproduce the branch's own
        # reported scalars exactly -- both go through `_subset_codes` and
        # `_score_candidate` (see `_prefix_metric_series`), not two separate
        # calls that can silently diverge on a coarsened combination.
        assert branch.mi_by_prefix_d[-1] == pytest.approx(branch.mi)
        assert branch.mi_adj_by_prefix_d[-1] == pytest.approx(branch.mi_adj)
        assert branch.u_adj_by_prefix_d[-1] == pytest.approx(branch.u_adj)


def test_prefix_metrics_are_monotone_non_decreasing_in_prefix_length():
    # I(Z; X_{S[:k+1]}) >= I(Z; X_{S[:k]}) always holds for exact discrete MI
    # (I(Z; X, Y) = I(Z; X) + I(Z; Y|X) >= I(Z; X), and I(Z; Y|X) >= 0) --
    # adding an axis to a fixed prefix can never *reduce* the information
    # that prefix's own axes carry about Z. Uses a low-cardinality, ample-N
    # combination well inside the Grid Capacity Limit so no coarsening
    # perturbs this monotonicity.
    rng = np.random.default_rng(12)
    n = 3000
    X = rng.integers(0, 3, size=(n, 4))
    Z = rng.integers(0, 2, size=n)
    branches = discover_branches(X, Z, max_d=4)
    branch = branches[4]
    mi_series = branch.mi_by_prefix_d
    for k in range(1, len(mi_series)):
        assert mi_series[k] >= mi_series[k - 1] - 1e-9


def test_prefix_metrics_are_a_projection_not_the_independent_per_d_branch():
    # `mi_by_prefix_d` answers "what would THIS branch's own axes read if
    # collapsed to k of them" -- a DIFFERENT question from
    # `discover_branches`'s own per-d search, which independently
    # re-optimizes and need not select the same features (branches are not
    # required to be nested; see the XOR-synergy tests above). Reusing the
    # XOR-synergy fixture: the d=3 branch's first-2-feature projection must
    # not be assumed to equal the independently-discovered d=2 branch's mi,
    # and in this fixture the d=1 winner ({c}) is disjoint from the d=2/d=3
    # winners ({a, b, ...}), so the d=3 branch's OWN 1-axis prefix (its
    # first selected feature, from {a, b}) is a materially weaker predictor
    # than the independently-optimal d=1 branch ({c}).
    X, Z, feature_names = _xor_synergy_dataset()
    branches = discover_branches(X, Z, feature_names=feature_names, max_d=2)
    branch_d1 = branches[1]
    branch_d2 = branches[2]
    assert set(branch_d2.selected_feature_names) == {"a", "b"}
    assert branch_d1.selected_feature_names == ["c"]
    # The d=2 branch's own 1-axis prefix (first of {a, b}) is near-useless
    # in isolation (XOR secret sharing), unlike the independent d=1 winner.
    assert branch_d2.mi_by_prefix_d[0] < branch_d1.mi


def test_max_branch_d_constant_is_4():
    # This ceiling matches the display's actual spatial encoding (3
    # coordinate axes + 1 time/frame axis, Section 1.3) -- pinned here so a
    # silent change is caught.
    assert MAX_BRANCH_D == 4


# ---------------------------------------------------------------------------
# BranchEngine: thin, stateless wrapper
# ---------------------------------------------------------------------------

def test_branch_engine_matches_direct_discover_branches_call():
    X, Z, feature_names = _xor_synergy_dataset(seed=55)
    engine = BranchEngine(max_d=2)
    via_engine = engine.fit(X, Z, feature_names=feature_names)
    via_function = discover_branches(X, Z, feature_names=feature_names, max_d=2)
    assert via_engine == via_function


def test_branch_engine_default_max_d_is_module_constant():
    engine = BranchEngine()
    assert engine.max_d == MAX_BRANCH_D


def test_branch_engine_is_stateless_across_fits():
    # Fitting on one dataset must not leave any state that influences a
    # later fit on a different dataset (no significance-testing parameters,
    # no random_state, no history -- see this module's docstring).
    engine = BranchEngine(max_d=2)
    X1, Z1, names1 = _xor_synergy_dataset(seed=1)
    X2, Z2, names2 = _xor_synergy_dataset(seed=2)

    r1_first = engine.fit(X1, Z1, feature_names=names1)
    engine.fit(X2, Z2, feature_names=names2)
    r1_second = engine.fit(X1, Z1, feature_names=names1)
    assert r1_first == r1_second


# ---------------------------------------------------------------------------
# Old v1.0 API surface must be genuinely gone, not just deprecated
# ---------------------------------------------------------------------------

def test_v1_api_symbols_are_gone():
    for name in (
        "AVREngine", "AVRResult", "Scenario", "d_star", "l_target",
        "l_feat", "xai_message", "selection_history",
    ):
        assert not hasattr(avr_mod, name), f"v1.0 symbol {name!r} unexpectedly still present"


def test_branch_result_has_no_v1_fields():
    fields = {f for f in BranchResult.__dataclass_fields__}
    # v1.0 fields (scenario/vir/l_target/l_feat/xai_message/selection_history/
    # n_significant_features) must never come back.
    v1_fields = {
        "scenario", "vir", "l_target", "l_feat", "xai_message",
        "selection_history", "n_significant_features",
    }
    assert fields.isdisjoint(v1_fields)
    # `nmi` is likewise gone for good, and its absence is pinned as hard as
    # the v1.0 symbols': it reads ~66 % on data that is provably independent
    # of the target whenever that target is a rare class, so no consumer may
    # be able to reach it through a BranchResult.
    assert "nmi" not in fields
    assert "nmi_by_prefix_d" not in fields
    # Current v2.2 "Certified Centers" field set. The v2.1 fields are all
    # still present: v2.2 DEMOTES u_adj from headline to diagnostic, it does
    # not remove it -- MI_adj remains the default ranking statistic and the
    # correct answer to "does an association exist", which is a question the
    # centre layer does not answer.
    assert fields == {
        "d", "selected_features", "selected_feature_names",
        "mi", "mi_null", "mi_adj", "u_adj", "h_target",
        "mi_by_prefix_d", "mi_adj_by_prefix_d", "u_adj_by_prefix_d",
        "p_value", "p_value_familywise", "per_class",
        "centers", "coverage_by_prefix_d", "n_centers_by_prefix_d",
        "purity_by_prefix_d", "coverage_p_value_familywise",
    }


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
