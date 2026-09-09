"""
Unit tests for vsf.avr — Independent Branch Discovery (IBD).

Covers the properties this module is actually supposed to guarantee now:

  1. Exhaustiveness: the d=2 search genuinely evaluates every pair, so it
     finds an XOR-style synergistic pair that neither feature reveals
     individually — something a greedy chain seeded at the best single
     feature could never discover (it never revisits its first pick).
  2. Independence across dimensionalities: the d=1 winner need not be a
     subset of the d=2 winner (branches are not required to be nested).
  3. A positive class is always required, resolved exactly the way the
     module docstring says (explicit and validated, or auto for a
     two-valued target, or a ValueError otherwise) — and which class is
     named actually changes what the search reports.
  4. Determinism: repeated calls on identical input return identical
     output — no randomness anywhere in the algorithm's own logic.
  5. The Grid Capacity coarsening path (occupancy check in
     `_CandidateFactory`, `coarsen_column`) is genuinely invoked for a combination
     whose joint table would be too sparse.
  6. Edge cases: fewer features than max_d, a single feature, zero
     features, out-of-range max_d, negative permutation counts.
  7. `BranchEngine` is a thin, stateless wrapper around `discover_branches`.
  8. `coverage_by_prefix_d` / `n_centers_by_prefix_d` / `purity_by_prefix_d`
     have the right length and their final entry matches the branch's own
     scalars exactly.
  9. `has_certified_centers` / `select_branch_dimensionality` behave
     correctly both on pure noise (nothing certifies) and on a branch with
     real, certifiable signal.
 10. The old v1.0 API surface and the v2.1/v2.2 MI-era `BranchResult` fields
     are genuinely gone, not just deprecated.

A note on test data construction: every fixture below is CATEGORICAL, which
is the only kind of data VSF accepts — each distinct value of a column is
one category, whether it arrives as a string, a bool or an integer
(`vsf.pmd.discretize_feature`). Earlier versions of these fixtures added
Gaussian jitter to binary columns to work around a bug in the quantile
binning of numeric features; that binning, and the bug with it, no longer
exist, so the fixtures use the raw category codes.
"""

from unittest import mock

import numpy as np
import pytest

import vsf.avr as avr_mod
from vsf.avr import (
    MAX_BRANCH_D,
    BranchEngine,
    BranchResult,
    discover_branches,
    select_branch_dimensionality,
)
from vsf.metrics import cell_codes


# ---------------------------------------------------------------------------
# Exhaustiveness + non-nested branches: XOR synergy a greedy chain would miss
# ---------------------------------------------------------------------------

def _xor_synergy_dataset(n=4000, seed=123):
    """
    Z = XOR(a, b). a and b are individually uncorrelated with Z (standard
    XOR secret-sharing property: each share alone carries zero information
    about the secret when the other share is uniform and independent), but
    jointly they determine Z exactly. `c` is a noisy DIRECT copy of Z (30%
    label flip) — related to Z, but not cleanly enough for any single cell
    to clear the default purity floor. Two pure-noise columns widen the
    search space so this isn't a trivially small combinatorial check.

    Every column is CATEGORICAL, which is the only kind of column VSF
    accepts: `a`, `b` and `c` are two-valued, and the two noise columns
    carry 5 and 3 unrelated levels respectively.
    """
    rng = np.random.default_rng(seed)
    a_bit = rng.integers(0, 2, size=n)
    b_bit = rng.integers(0, 2, size=n)
    Z = np.bitwise_xor(a_bit, b_bit)

    flip = rng.random(n) < 0.3
    c_bit = np.where(flip, 1 - Z, Z)

    noise1 = rng.integers(0, 5, size=n)
    noise2 = rng.integers(0, 3, size=n)

    X = np.column_stack([a_bit, b_bit, c_bit, noise1, noise2])
    feature_names = ["a", "b", "c", "n1", "n2"]
    return X, Z, feature_names


def test_exhaustive_search_finds_xor_synergy_a_greedy_chain_would_miss():
    X, Z, feature_names = _xor_synergy_dataset()
    branches = discover_branches(X, Z, feature_names=feature_names, max_d=2, cv_repeats=0)

    # d=1: `c` is the best available single feature (the XOR secret-sharing
    # property means `a`/`b` alone carry nothing), but even its 70%-accurate
    # signal never clears the default purity floor (tau=0.90) in any cell —
    # so the "best" d=1 branch still certifies nothing.
    assert branches[1].selected_feature_names == ["c"]
    assert branches[1].centers.coverage == 0.0
    assert branches[1].centers.n_centers == 0

    # d=2: the TRUE synergistic pair {a, b} must win, even though neither
    # of its members was selected at d=1. A greedy chain that extends the
    # d=1 winner ({c} -> {c, x}) can never find this pair — discovering it
    # requires genuinely re-examining combinations that don't contain `c`.
    assert set(branches[2].selected_feature_names) == {"a", "b"}

    # {a, b} recovers Z almost perfectly: most positives sit in near-pure
    # cells, decisively higher coverage than the d=1 branch (which has none).
    assert branches[2].centers.coverage > 0.8
    assert branches[2].centers.n_centers > 0
    assert branches[2].centers.purity_pooled > 0.95
    assert branches[2].centers.coverage > branches[1].centers.coverage


def test_branch_d1_winner_is_not_a_subset_of_branch_d2_winner():
    # The direct "branches need not be nested" claim, stated as its own
    # explicit assertion independent of the synergy narrative above.
    X, Z, feature_names = _xor_synergy_dataset()
    branches = discover_branches(X, Z, feature_names=feature_names, max_d=2, cv_repeats=0)
    d1_set = set(branches[1].selected_features)
    d2_set = set(branches[2].selected_features)
    assert not d1_set.issubset(d2_set)
    assert d1_set.isdisjoint(d2_set)


# ---------------------------------------------------------------------------
# Positive-class resolution (v2.3: always required, no MI-based fallback)
# ---------------------------------------------------------------------------

def test_discover_branches_raises_without_a_resolvable_positive_class():
    rng = np.random.default_rng(1)
    n = 500
    X = rng.integers(0, 3, size=(n, 3))
    Z = rng.integers(0, 4, size=n)  # 4-class target, no positive_class given
    with pytest.raises(ValueError, match="resolvable positive class"):
        discover_branches(X, Z, max_d=1)


def test_positive_class_selects_coverage_of_the_named_class_not_another_one():
    # A 3-class target where feature A almost perfectly identifies "hi" and
    # says nothing about "lo": pinning positive_class must change what the
    # search reports, not just silently accept the argument.
    rng = np.random.default_rng(42)
    n = 3000
    Z = rng.choice(["lo", "mid", "hi"], size=n, p=[0.5, 0.35, 0.15])
    noise = rng.integers(0, 5, size=n)
    A_bit = np.where(Z == "hi", 1, 0)
    # 1% label noise, not 5%: with a 15% "hi" prevalence, a 5% flip rate puts
    # enough non-"hi" rows into the A=1 category to drag its purity to ~77%,
    # below the default tau=0.90, so NOTHING would certify and the test would
    # be asserting on an empty report. At 1% the A=1 category is ~95% pure,
    # which is the "almost perfectly identifies hi" the test describes.
    flip = rng.random(n) < 0.01
    A = np.where(flip, 1 - A_bit, A_bit)
    X = np.column_stack([A, noise])
    names = ["A", "noise"]

    hi_branch = discover_branches(X, Z, feature_names=names, max_d=1,
                                  positive_class="hi", cv_repeats=0)[1]
    lo_branch = discover_branches(X, Z, feature_names=names, max_d=1,
                                  positive_class="lo", cv_repeats=0)[1]
    assert hi_branch.centers.n_centers > 0
    assert hi_branch.centers.coverage > lo_branch.centers.coverage
    # A is unrelated to "lo" by construction: nothing about it can certify.
    assert lo_branch.centers.coverage == 0.0
    assert lo_branch.centers.n_centers == 0


def test_positive_class_must_match_exactly_one_distinct_value():
    rng = np.random.default_rng(1)
    n = 300
    X = rng.integers(0, 3, size=(n, 2))
    Z = rng.integers(0, 4, size=n)
    with pytest.raises(ValueError, match=r"matches 0 distinct target values"):
        discover_branches(X, Z, max_d=1, positive_class=999)


def test_resolve_positive_indicator_auto_picks_the_higher_code():
    # Direct unit coverage of the resolution order stated in
    # `_resolve_positive_indicator`'s docstring: no `positive_class` and a
    # two-valued target -> the higher `np.unique` code is positive.
    Z = np.array(["e", "p", "e", "e", "p", "p", "e"])
    z_codes, n_rows = cell_codes(Z)
    assert n_rows == 2
    assert list(np.unique(Z)) == ["e", "p"]  # "p" is the higher code
    indicator = avr_mod._resolve_positive_indicator(Z, z_codes, n_rows, None)
    assert np.array_equal(indicator, (Z == "p").astype(np.int8))


def test_resolve_positive_indicator_rejects_a_positive_class_after_binning():
    # positive_class is only meaningful when raw values map 1:1 onto the
    # integer codes; once the target has been PMD-binned that mapping is
    # gone and the function must refuse rather than silently certify the
    # wrong class.
    z_codes, n_rows = cell_codes(np.array([0, 0, 1, 1] * 10))
    raw = np.array([0.11, 0.12, 0.13, 0.14] * 10)  # 4 raw values -> 2 bins
    with pytest.raises(ValueError, match="no longer correspond 1:1"):
        avr_mod._resolve_positive_indicator(raw, z_codes, n_rows, 0.11)


def test_negative_permutation_counts_are_rejected():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(100, 2))
    Z = rng.integers(0, 2, size=100)
    with pytest.raises(ValueError, match="non-negative"):
        discover_branches(X, Z, max_d=1, n_permutations_centers=-1)
    with pytest.raises(ValueError, match="non-negative"):
        discover_branches(X, Z, max_d=1, n_permutations_familywise_coverage=-1)


# ---------------------------------------------------------------------------
# Determinism: no randomness anywhere in the algorithm's own logic
# ---------------------------------------------------------------------------

def test_determinism_repeated_calls_produce_identical_output():
    X, Z, feature_names = _xor_synergy_dataset(seed=99)
    branches_1 = discover_branches(X, Z, feature_names=feature_names, max_d=3, cv_repeats=0)
    branches_2 = discover_branches(X, Z, feature_names=feature_names, max_d=3, cv_repeats=0)

    assert set(branches_1.keys()) == set(branches_2.keys())
    for d in branches_1:
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
    r1 = discover_branches(X, Z, max_d=2, cv_repeats=0)
    r2 = discover_branches(X, Z, max_d=2, cv_repeats=0)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Grid Capacity Limit coarsening path is genuinely invoked
# ---------------------------------------------------------------------------

def test_grid_capacity_coarsening_invoked_for_sparse_combination():
    # A single categorical feature with 25 distinct levels against n=200
    # samples: 25 occupied cells > n // 10 = 20 -> the capacity rule must
    # merge levels before scoring.
    rng = np.random.default_rng(0)
    n = 200
    high_card = np.array([f"cat_{i % 25}" for i in rng.permutation(n)])
    other = rng.integers(0, 3, size=n).astype(str)
    Z = rng.integers(0, 2, size=n)
    X = np.column_stack([high_card, other])

    # The search coarsens per column through `vsf.pmd.coarsen_column` (once
    # per (column, level count), cached in `_CandidateFactory`) rather than
    # per subset through `adaptively_coarsen_bins`; the partition is the
    # same, see `test_candidate_factory_matches_per_subset_coarsening`. The
    # capacity check itself is the occupancy count of the joint code
    # (`_CandidateFactory.capacity`), not a separate call.
    with mock.patch.object(
        avr_mod, "coarsen_column", wraps=avr_mod.coarsen_column
    ) as spy_coarsen:
        branches = discover_branches(X, Z, max_d=1, cv_repeats=0)

    assert spy_coarsen.called, (
        "coarsen_column must be invoked for the 25-level combination "
        "whose joint table exceeds the N/10 capacity limit"
    )
    # The branch must still be produced (coarsening degrades gracefully,
    # it doesn't crash or silently drop the combination from consideration).
    assert 1 in branches

    coarsened_levels = [
        int(np.unique(call.args[0]).shape[0]) for call in spy_coarsen.call_args_list
    ]
    assert all(levels >= 1 for levels in coarsened_levels)


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
        avr_mod, "coarsen_column", wraps=avr_mod.coarsen_column
    ) as spy_coarsen:
        discover_branches(X, Z, max_d=2, cv_repeats=0)

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

    branches = discover_branches(X, Z, max_d=MAX_BRANCH_D, cv_repeats=0)
    assert set(branches.keys()) == {1, 2}
    assert 3 not in branches
    assert 4 not in branches


def test_single_feature_dataset():
    rng = np.random.default_rng(3)
    n = 200
    X = rng.integers(0, 4, size=(n, 1))
    Z = rng.integers(0, 2, size=n)

    branches = discover_branches(X, Z, max_d=MAX_BRANCH_D, cv_repeats=0)
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
    branches = discover_branches(X, Z, max_d=1, cv_repeats=0)
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
    branches = discover_branches(X, Z, max_d=ok_max_d, cv_repeats=0)
    assert max(branches.keys()) == ok_max_d
    assert set(branches.keys()) == set(range(1, ok_max_d + 1))


def test_branch_result_fields_are_internally_consistent():
    rng = np.random.default_rng(10)
    n = 400
    X = rng.integers(0, 3, size=(n, 4))
    Z = rng.integers(0, 2, size=n)
    branches = discover_branches(X, Z, max_d=3, cv_repeats=0)
    for d, branch in branches.items():
        assert isinstance(branch, BranchResult)
        assert branch.d == d
        assert len(branch.selected_features) == d
        assert len(branch.selected_feature_names) == d
        assert 0.0 <= branch.centers.coverage <= 1.0 + 1e-9
        assert branch.centers.n_centers >= 0
        if branch.centers.n_centers == 0:
            assert branch.centers.coverage == 0.0


# ---------------------------------------------------------------------------
# coverage_by_prefix_d / n_centers_by_prefix_d / purity_by_prefix_d
# ---------------------------------------------------------------------------

def test_prefix_series_length_matches_branch_d_and_last_entry_matches_scalars():
    X, Z, feature_names = _xor_synergy_dataset()
    branches = discover_branches(X, Z, feature_names=feature_names, max_d=2, cv_repeats=0)
    for d, branch in branches.items():
        assert len(branch.coverage_by_prefix_d) == d
        assert len(branch.n_centers_by_prefix_d) == d
        assert len(branch.purity_by_prefix_d) == d
        # The final (full-d) prefix entry must reproduce the branch's own
        # reported scalars exactly (see `BranchResult`'s docstring).
        assert branch.coverage_by_prefix_d[-1] == pytest.approx(branch.centers.coverage)
        assert branch.n_centers_by_prefix_d[-1] == branch.centers.n_centers
        assert branch.purity_by_prefix_d[-1] == pytest.approx(branch.centers.purity_pooled)


def test_prefix_series_is_a_projection_not_the_independent_per_d_branch():
    # `coverage_by_prefix_d` answers "what would THIS branch's own axes read
    # if collapsed to k of them" — a DIFFERENT question from
    # `discover_branches`'s own per-d search, which independently
    # re-optimizes and need not select the same features (branches are not
    # required to be nested; see the XOR-synergy tests above). In this
    # fixture the d=2 branch's first-selected axis (one of {a, b}, both
    # individually useless by the XOR secret-sharing property) is a
    # materially weaker predictor in isolation than the independently
    # discovered d=1 branch ({c}) — even though d=1 itself certifies
    # nothing at the default tau.
    X, Z, feature_names = _xor_synergy_dataset()
    branches = discover_branches(X, Z, feature_names=feature_names, max_d=2, cv_repeats=0)
    branch_d2 = branches[2]
    assert set(branch_d2.selected_feature_names) == {"a", "b"}
    # First axis alone certifies nothing (XOR secret sharing), while the
    # full 2-axis branch certifies most of the class.
    assert branch_d2.coverage_by_prefix_d[0] == 0.0
    assert branch_d2.coverage_by_prefix_d[-1] > 0.8


def test_max_branch_d_constant_is_4():
    # This ceiling matches the display's actual spatial encoding (3
    # coordinate axes + 1 time/frame axis) — pinned here so a silent change
    # is caught.
    assert MAX_BRANCH_D == 4


# ---------------------------------------------------------------------------
# Familywise (look-elsewhere corrected) coverage p-value
# ---------------------------------------------------------------------------

def test_familywise_coverage_p_value_is_none_unless_requested():
    X, Z, feature_names = _xor_synergy_dataset()
    default_off = discover_branches(X, Z, feature_names=feature_names, max_d=1, cv_repeats=0)
    assert default_off[1].coverage_p_value_familywise is None

    requested = discover_branches(
        X, Z, feature_names=feature_names, max_d=1, cv_repeats=0,
        n_permutations_familywise_coverage=49, random_state=0,
    )
    assert requested[1].coverage_p_value_familywise is not None
    assert 0.0 <= requested[1].coverage_p_value_familywise <= 1.0


# ---------------------------------------------------------------------------
# has_certified_centers / select_branch_dimensionality
# ---------------------------------------------------------------------------

def test_has_certified_centers_is_false_on_pure_noise():
    rng = np.random.default_rng(77)
    n = 3000
    X = rng.integers(0, 3, size=(n, 3))
    Z = (rng.random(n) < 0.3).astype(int)
    branches = discover_branches(X, Z, max_d=2, cv_repeats=0)
    for branch in branches.values():
        assert branch.centers.n_centers == 0
        assert not branch.has_certified_centers()


def test_has_certified_centers_is_true_for_a_branch_with_real_signal():
    X, Z, feature_names = _xor_synergy_dataset()
    branches = discover_branches(
        X, Z, feature_names=feature_names, max_d=2,
        n_permutations_centers=199, random_state=0,
    )
    assert branches[2].has_certified_centers()
    assert not branches[1].has_certified_centers()  # nothing certifies at d=1


def test_select_branch_dimensionality_is_none_on_pure_noise():
    rng = np.random.default_rng(77)
    n = 3000
    X = rng.integers(0, 3, size=(n, 3))
    Z = (rng.random(n) < 0.3).astype(int)
    branches = discover_branches(X, Z, max_d=2)  # default cv_repeats=5
    assert select_branch_dimensionality(branches) is None


def test_select_branch_dimensionality_picks_the_smallest_sufficient_branch():
    X, Z, feature_names = _xor_synergy_dataset()
    branches = discover_branches(X, Z, feature_names=feature_names, max_d=2)  # default cv
    # d=1 has zero cross-validated coverage (nothing certifies), d=2's is
    # decisively positive -> the smallest SUFFICIENT dimensionality is 2.
    assert select_branch_dimensionality(branches) == 2


# ---------------------------------------------------------------------------
# BranchEngine: thin, stateless wrapper
# ---------------------------------------------------------------------------

def test_branch_engine_matches_direct_discover_branches_call():
    X, Z, feature_names = _xor_synergy_dataset(seed=55)
    engine = BranchEngine(max_d=2, cv_repeats=0)
    via_engine = engine.fit(X, Z, feature_names=feature_names)
    via_function = discover_branches(X, Z, feature_names=feature_names, max_d=2, cv_repeats=0)
    assert via_engine == via_function


def test_branch_engine_default_max_d_is_module_constant():
    engine = BranchEngine()
    assert engine.max_d == MAX_BRANCH_D


def test_branch_engine_is_stateless_across_fits():
    # Fitting on one dataset must not leave any state that influences a
    # later fit on a different dataset (no random_state mutation, no
    # history — see this module's docstring).
    engine = BranchEngine(max_d=2, cv_repeats=0)
    X1, Z1, names1 = _xor_synergy_dataset(seed=1)
    X2, Z2, names2 = _xor_synergy_dataset(seed=2)

    r1_first = engine.fit(X1, Z1, feature_names=names1)
    engine.fit(X2, Z2, feature_names=names2)
    r1_second = engine.fit(X1, Z1, feature_names=names1)
    assert r1_first == r1_second


# ---------------------------------------------------------------------------
# Old API surface must be genuinely gone, not just deprecated
# ---------------------------------------------------------------------------

def test_v1_and_v2_association_ranking_symbols_are_gone():
    for name in (
        # v1.0 "Adaptive Visual Routing" surface
        "AVREngine", "AVRResult", "Scenario", "d_star", "l_target",
        "l_feat", "xai_message", "selection_history",
        # v2.1/v2.2 MI-based ranking surface
        "Objective", "_score_candidate", "_null_floor", "is_significant",
    ):
        assert not hasattr(avr_mod, name), f"{name!r} unexpectedly still present in vsf.avr"


def test_branch_result_has_exactly_the_v23_coverage_only_field_set():
    fields = set(BranchResult.__dataclass_fields__)
    # v1.0 fields must never come back.
    v1_fields = {
        "scenario", "vir", "l_target", "l_feat", "xai_message",
        "selection_history", "n_significant_features", "nmi", "nmi_by_prefix_d",
    }
    # v2.1/v2.2 MI-era fields must never come back either — v2.3 removed the
    # association-ranking path entirely, it did not just stop reporting it.
    mi_era_fields = {
        "mi", "mi_null", "mi_adj", "u_adj", "h_target", "p_value",
        "p_value_familywise", "per_class",
        "mi_by_prefix_d", "mi_adj_by_prefix_d", "u_adj_by_prefix_d",
    }
    assert fields.isdisjoint(v1_fields)
    assert fields.isdisjoint(mi_era_fields)
    assert fields == {
        "d", "selected_features", "selected_feature_names", "centers",
        "coverage_by_prefix_d", "n_centers_by_prefix_d", "purity_by_prefix_d",
        "coverage_p_value_familywise",
    }


def test_discover_branches_has_no_objective_parameter():
    import inspect
    params = inspect.signature(discover_branches).parameters
    assert "objective" not in params
    assert "null" not in params
    assert "n_permutations" not in params  # renamed n_permutations_centers
    assert "n_permutations_familywise" not in params  # renamed *_coverage


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
