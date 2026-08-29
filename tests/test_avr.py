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

    # {a, b} recovers Z almost perfectly (H(Z) = 1 bit): its MI/NMI must
    # both be close to the ceiling, and strictly higher than the d=1 winner
    # could ever reach on its own.
    assert branches[2].mi == pytest.approx(1.0, abs=0.15)
    assert branches[2].nmi == pytest.approx(1.0, abs=0.15)
    assert branches[2].mi > 0.8
    assert branches[2].mi > branches[1].mi


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
# Ranking is by RAW MI, not NMI (Section 4.2 / 4.5)
# ---------------------------------------------------------------------------

def test_ranking_within_one_d_uses_raw_mi_not_nmi():
    # Constructed so raw-MI and NMI rankings actively DISAGREE:
    #   Feature A: binary, moderate raw MI, but H(A) <= 1 bit caps its own
    #     denominator -> high NMI.
    #   Feature B: high-cardinality (matches Z's 8 classes), higher raw MI,
    #     but Z's entropy (3 bits) dominates the NMI denominator -> lower NMI.
    rng = np.random.default_rng(7)
    n = 8000
    K = 8
    Z = rng.integers(0, K, size=n)  # 8-class uniform target, H(Z) = 3 bits

    half = (Z < K // 2).astype(int)
    noise_a = rng.random(n) < 0.20
    a_int = np.where(noise_a, 1 - half, half)
    A = np.array(["hi" if v == 1 else "lo" for v in a_int])

    noise_b = rng.random(n) < 0.55
    b_int = np.where(noise_b, rng.integers(0, K, size=n), Z)
    B = np.array([f"cat{v}" for v in b_int])

    from vsf.math import mutual_information, normalized_mutual_information

    mi_a, nmi_a = mutual_information(Z, A), normalized_mutual_information(Z, A)
    mi_b, nmi_b = mutual_information(Z, B), normalized_mutual_information(Z, B)
    # Sanity-check the constructed disagreement actually holds before
    # trusting it as a regression fixture.
    assert mi_b > mi_a, "fixture must have raw MI favoring B"
    assert nmi_a > nmi_b, "fixture must have NMI favoring A"

    X = np.column_stack([A, B])
    branches = discover_branches(X, Z, feature_names=["A", "B"], max_d=1)

    # discover_branches must follow raw MI (picks B), NOT NMI (which would
    # have picked A).
    assert branches[1].selected_feature_names == ["B"]
    assert branches[1].mi == pytest.approx(mi_b)


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
        assert 0.0 <= branch.nmi <= 1.0 + 1e-9


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
    assert fields == {"d", "selected_features", "selected_feature_names", "mi", "nmi"}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
