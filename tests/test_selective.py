"""vsf.selective: post-selection certificates, nested CV, schema stability."""
from __future__ import annotations

import itertools
from math import comb
from typing import Dict, Tuple

import numpy as np
import pytest

from vsf.avr import _exhaustive_search, _prepare_search, discover_branches
from vsf.centers import (
    CenterSpec,
    _cell_counts,
    crossvalidated_coverage,
    min_successes_to_certify,
    select_centers,
    stratified_repeated_kfold,
)
from vsf.selective import (
    MAX_ALPHA,
    certify_discovery,
    exact_upper_tail,
    family_test_count,
    nested_crossvalidation,
    random_halves,
    schema_stability,
)


def _noise(seed: int, n: int = 600, m: int = 6, levels: int = 3, p: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, levels, size=(n, m)).astype(str)
    Z = np.where(rng.random(n) < p, "1", "0")
    return X, Z


def _planted(seed: int, n: int = 900) -> Tuple[np.ndarray, np.ndarray]:
    """Cell x0 = 0 and x1 = 0 is 98 % positive; everything else 10 %."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, 4))
    inside = (X[:, 0] == 0) & (X[:, 1] == 0)
    p = np.where(inside, 0.98, 0.10)
    Z = np.where(rng.random(n) < p, "1", "0")
    return X.astype(str), Z


# ---------------------------------------------------------------------------
# exact binomial tail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 7, 30])
@pytest.mark.parametrize("tau", [0.3, 0.9])
def test_exact_upper_tail_matches_the_binomial_sum(n: int, tau: float) -> None:
    for k in range(0, n + 2):
        ref = sum(comb(n, j) * tau**j * (1 - tau) ** (n - j) for j in range(k, n + 1)) if k <= n else 0.0
        got = float(exact_upper_tail(k, n, tau)[0])
        assert got == pytest.approx(ref, rel=1e-11, abs=1e-300)


@pytest.mark.parametrize("tau", [0.6, 0.9])
@pytest.mark.parametrize("level", [0.05, 1e-3, 1e-6])
def test_exact_upper_tail_agrees_with_the_certification_threshold(tau: float, level: float) -> None:
    n = np.arange(1, 301)
    k_min = min_successes_to_certify(n, tau, level)
    for size, k in zip(n.tolist(), k_min.tolist()):
        if k <= size:
            assert exact_upper_tail(k, size, tau)[0] <= level * (1 + 1e-9)
        if 1 <= k - 1 <= size:
            assert exact_upper_tail(k - 1, size, tau)[0] > level * (1 - 1e-9)


def test_certifying_thresholds_lie_in_the_range_of_hoeffdings_bound() -> None:
    # The Poisson-binomial argument needs k >= n * tau + 1 at every level the
    # certificates use (<= MAX_ALPHA). Above 0.05 it fails for small cells.
    n = np.arange(1, 601)
    for tau in np.round(np.arange(0.05, 0.99, 0.02), 2):
        for level in (MAX_ALPHA, 1e-3, 1e-7):
            k = min_successes_to_certify(n, float(tau), level)
            assert np.all((k > n) | (k >= n * tau + 1 - 1e-9))
    k = min_successes_to_certify(n, 0.05, 0.2)
    assert not np.all((k > n) | (k >= n * 0.05 + 1 - 1e-9))


def test_exact_upper_tail_edges() -> None:
    assert exact_upper_tail(0, 5, 0.5)[0] == 1.0
    assert exact_upper_tail(6, 5, 0.5)[0] == 0.0
    assert exact_upper_tail(0, 0, 0.5)[0] == 1.0
    assert exact_upper_tail([1, 2], [3, 3], 0.5).shape == (2,)
    with pytest.raises(ValueError):
        exact_upper_tail(1, 1, 1.0)
    with pytest.raises(ValueError):
        exact_upper_tail(1, -1, 0.5)


# ---------------------------------------------------------------------------
# row-restricted search
# ---------------------------------------------------------------------------
def _factory(X: np.ndarray, Z: np.ndarray, spec: CenterSpec):
    prepared = _prepare_search(X, Z, None, "1", spec, "presence")
    assert prepared is not None
    return prepared


def test_search_on_all_rows_is_identical_to_the_unrestricted_search() -> None:
    X, Z = _noise(1)
    spec = CenterSpec(tau=0.5)
    factory, z, _ = _factory(X, Z, spec)
    a = _exhaustive_search(factory, z.astype(np.int64), 2, [1], spec, 3)[0]
    b = _exhaustive_search(factory, z.astype(np.int64), 2, [1], spec, 3, rows=np.arange(z.size))[0]
    assert a == b


def test_search_on_rows_equals_a_search_on_that_subset_when_capacity_does_not_bind() -> None:
    X, Z = _noise(2, n=800, m=5, levels=2)
    spec = CenterSpec(tau=0.5, min_samples=3)
    factory, z, _ = _factory(X, Z, spec)
    rows = np.sort(np.random.default_rng(0).choice(z.size, 500, replace=False))
    got = _exhaustive_search(factory, z.astype(np.int64), 2, [1], spec, 4, rows=rows)[0]
    sub_factory, sub_z, _ = _factory(X[rows], Z[rows], spec)
    ref = _exhaustive_search(sub_factory, sub_z.astype(np.int64), 2, [1], spec, 4)[0]
    assert got == ref


def test_search_rows_are_validated() -> None:
    X, Z = _noise(3)
    spec = CenterSpec(tau=0.5)
    factory, z, _ = _factory(X, Z, spec)
    with pytest.raises(ValueError):
        _exhaustive_search(factory, z.astype(np.int64), 2, [1], spec, 2, rows=np.array([z.size]))
    with pytest.raises(ValueError):
        _exhaustive_search(factory, z[:-1].astype(np.int64), 2, [1], spec, 2)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_random_halves_partition_the_rows_deterministically() -> None:
    a, b = random_halves(11, 5)
    assert a.size == 6 and b.size == 5
    assert np.array_equal(np.sort(np.concatenate([a, b])), np.arange(11))
    a2, b2 = random_halves(11, 5)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    with pytest.raises(ValueError):
        random_halves(1, 0)


@pytest.mark.parametrize("m", [1, 5])
def test_family_test_count_is_the_number_of_eligible_cells_of_every_schema(m: int) -> None:
    X, Z = _noise(4, n=300, m=5)
    spec = CenterSpec(tau=0.5)
    factory, _, _ = _factory(X, Z, spec)
    brute = 0
    for d in range(1, 5):
        for combo in itertools.combinations(range(5), d):
            codes, n_cells = factory.codes(combo)
            brute += int(np.count_nonzero(np.bincount(codes, minlength=n_cells) >= m))
    assert family_test_count(factory, 4, m) == brute


def test_schema_stability_summary() -> None:
    st = schema_stability([(1, 0), (0, 1), (2, 1)], (0, 1))
    assert st.n_resamples == 3
    assert st.share_equal_to_reference == pytest.approx(2 / 3)
    assert st.modal_schema == (0, 1) and st.modal_share == pytest.approx(2 / 3)
    assert st.n_distinct == 2
    assert st.mean_pairwise_jaccard == pytest.approx((1 + 1 / 3 + 1 / 3) / 3)
    single = schema_stability([(3,)], (3,))
    assert single.mean_pairwise_jaccard == 1.0 and single.share_equal_to_reference == 1.0


# ---------------------------------------------------------------------------
# family-wide Bonferroni
# ---------------------------------------------------------------------------
def _coverage_at_level(z: np.ndarray, codes: np.ndarray, n_cells: int, tau: float, level: float, m: int) -> Tuple[int, int]:
    k, n = _cell_counts(z, codes, n_cells)
    ok = (n >= m) & (exact_upper_tail(k, n, tau) <= level)
    return int(k[ok].sum()), int(ok.sum())


def test_family_bonferroni_tests_every_eligible_cell_at_alpha_over_t() -> None:
    X, Z = _planted(5)
    spec = CenterSpec(tau=0.8, alpha=0.05, min_samples=2)
    res = certify_discovery(X, Z, positive_class="1", center_spec=spec, method="family_bonferroni")
    factory, z, _ = _factory(X, Z, spec)
    t = family_test_count(factory, 4, 2)
    assert set(res.n_tests.values()) == {t}
    level = 0.05 / t
    for d, br in res.branches.items():
        codes, n_cells = factory.codes(br.features)
        k, n = _cell_counts(z, codes, n_cells)
        assert sorted(c.cell for c in br.cells) == np.nonzero(n >= 2)[0].tolist()
        for c in br.cells:
            assert (c.k, c.n) == (int(k[c.cell]), int(n[c.cell]))
            assert c.certified == (exact_upper_tail(c.k, c.n, 0.8)[0] <= level)
            assert c.p_adjusted == pytest.approx(min(1.0, c.p_value * t))
        # the reported schema maximises coverage of family-certified cells
        best = max(
            _coverage_at_level(z, *factory.codes(combo), 0.8, level, 2)[0]
            for combo in itertools.combinations(range(4), d)
        )
        assert sum(c.k for c in br.certified_cells) == best


def test_family_bonferroni_finds_the_planted_cell() -> None:
    X, Z = _planted(6)
    res = certify_discovery(
        X, Z, positive_class="1", center_spec=CenterSpec(tau=0.8), method="family_bonferroni"
    )
    br = res.branches[2]
    assert br.features == (0, 1)
    assert br.n_certified == 1
    assert br.certified_cells[0].k / br.certified_cells[0].n > 0.9
    assert br.coverage_eval == br.coverage_all_rows > 0.4
    assert "Bonferroni" in res.guarantee


# ---------------------------------------------------------------------------
# sample splitting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("multiplicity", ["all_branches", "per_branch"])
def test_split_tests_only_screened_cells_on_the_evaluation_half(multiplicity: str) -> None:
    X, Z = _planted(7)
    spec = CenterSpec(tau=0.8, min_samples=3)
    res = certify_discovery(
        X, Z, positive_class="1", center_spec=spec, method="split",
        branch_multiplicity=multiplicity, random_state=11,  # type: ignore[arg-type]
    )
    factory, z, _ = _factory(X, Z, spec)
    rows_a, rows_b = random_halves(z.size, 11)
    assert (res.search_rows, res.eval_rows) == (rows_a.size, rows_b.size)
    best = _exhaustive_search(factory, z.astype(np.int64), 2, [1], spec, 4, rows=rows_a)[0]
    candidates: Dict[int, list] = {}
    for d, (_, combo) in best.items():
        assert res.branches[d].features == tuple(combo)
        codes, n_cells = factory.codes(tuple(combo))
        k_a, n_a = _cell_counts(z[rows_a], codes[rows_a], n_cells)
        k_b, n_b = _cell_counts(z[rows_b], codes[rows_b], n_cells)
        screened = (n_a >= 3) & (k_a >= np.ceil(0.8 * n_a - 1e-9)) & (n_b >= 3)
        candidates[d] = np.nonzero(screened)[0].tolist()
        br = res.branches[d]
        assert sorted(c.cell for c in br.cells) == candidates[d]
        for c in br.cells:
            assert (c.k, c.n) == (int(k_b[c.cell]), int(n_b[c.cell]))
        assert br.n_eval == rows_b.size
    total = sum(len(v) for v in candidates.values())
    for d, br in res.branches.items():
        t = total if multiplicity == "all_branches" else len(candidates[d])
        assert res.n_tests[d] == t
        for c in br.cells:
            assert c.certified == (c.p_value <= 0.05 / t)
    assert res.branches[2].n_certified == 1


def test_split_screen_all_tests_every_cell_seen_in_both_halves() -> None:
    X, Z = _planted(8)
    res = certify_discovery(
        X, Z, positive_class="1", center_spec=CenterSpec(tau=0.8), method="split", screen="all",
    )
    assert len(res.branches[2].cells) == 9


def test_invalid_arguments_are_rejected() -> None:
    X, Z = _planted(9, n=200)
    with pytest.raises(ValueError):
        certify_discovery(X, Z, positive_class="1", center_spec=CenterSpec(tau=1.0))
    with pytest.raises(ValueError):
        certify_discovery(X, Z, positive_class="1", method="permutation")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        certify_discovery(X, Z, positive_class="1", screen="none")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        certify_discovery(X, Z, positive_class="1", max_d=5)
    with pytest.raises(ValueError):
        certify_discovery(X, Z, positive_class="1", center_spec=CenterSpec(tau=0.8, alpha=0.1))
    with pytest.raises(ValueError):  # tau not above the base rate
        certify_discovery(X, Z, positive_class="1", center_spec=CenterSpec(tau=0.05))


# ---------------------------------------------------------------------------
# calibration under the global null
# ---------------------------------------------------------------------------
# A null on which the uncorrected winner fails visibly: tau five points
# above the base rate, 4-level columns, every schema of d <= 4 scored.
_NULL = dict(n=1000, m=8, levels=4, p=0.40)
_NULL_TAU = 0.45


def _any_false_certificate(seed: int, method: str) -> bool:
    X, Z = _noise(seed, **_NULL)
    res = certify_discovery(
        X, Z, positive_class="1", center_spec=CenterSpec(tau=_NULL_TAU, alpha=0.05),
        method=method, random_state=seed,  # type: ignore[arg-type]
    )
    return any(br.n_certified > 0 for br in res.branches.values())


@pytest.mark.parametrize("method", ["family_bonferroni", "split"])
def test_null_false_certificates_are_rare(method: str) -> None:
    hits = sum(_any_false_certificate(s, method) for s in range(40))
    # Binomial(40, 0.05): P(X >= 7) < 0.02 even at the nominal rate.
    assert hits <= 6


def test_the_uncorrected_winner_is_what_needed_the_correction() -> None:
    # The defect this module exists for, on the same null: the reported
    # winner under per-schema Bonferroni certifies noise far above alpha.
    hits = 0
    for s in range(40):
        X, Z = _noise(s, **_NULL)
        br = discover_branches(
            X, Z, positive_class="1",
            center_spec=CenterSpec(tau=_NULL_TAU, rule="certified", alpha=0.05),
            n_permutations_centers=0, cv_repeats=0,
        )
        hits += any(b.centers.n_centers > 0 for b in br.values())
    # 10 of these 40 runs (25 %) against a nominal 5 %.
    assert hits >= 8


@pytest.mark.slow
@pytest.mark.parametrize("method", ["family_bonferroni", "split"])
def test_null_false_certificate_rate_is_at_most_alpha(method: str) -> None:
    runs = 300
    hits = sum(_any_false_certificate(1000 + s, method) for s in range(runs))
    assert hits / runs <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / runs)


# ---------------------------------------------------------------------------
# nested cross-validation
# ---------------------------------------------------------------------------
def test_nested_cv_fold_winners_and_coverages_are_recomputable() -> None:
    X, Z = _planted(10, n=600)
    spec = CenterSpec(tau=0.8)
    res = nested_crossvalidation(X, Z, positive_class="1", center_spec=spec, max_d=3, n_repeats=2, random_state=3)
    factory, z, _ = _factory(X, Z, spec)
    splits = stratified_repeated_kfold(z, 5, 2, 3)
    assert all(len(b.fold_winners) == len(splits) for b in res.branches.values())
    for i, (train, test) in enumerate(splits):
        best = _exhaustive_search(factory, z.astype(np.int64), 2, [1], spec, 3, rows=train)[0]
        for d, b in res.branches.items():
            combo = tuple(best[d][1])
            assert b.fold_winners[i] == combo
            codes, n_cells = factory.codes(combo)
            k_tr, n_tr = _cell_counts(z[train], codes[train], n_cells)
            mask, _ = select_centers(k_tr, n_tr, spec)
            k_te, _ = _cell_counts(z[test], codes[test], n_cells)
            expected = k_te[mask].sum() / k_te.sum()
            assert b.nested.per_split[i] == pytest.approx(expected)


def test_nested_cv_fixed_schema_is_the_existing_estimate_on_the_same_folds() -> None:
    X, Z = _planted(11, n=600)
    spec = CenterSpec(tau=0.8)
    res = nested_crossvalidation(X, Z, positive_class="1", center_spec=spec, max_d=2, n_repeats=3, random_state=4)
    factory, z, _ = _factory(X, Z, spec)
    for b in res.branches.values():
        codes, n_cells = factory.codes(b.full_data_winner)
        ref = crossvalidated_coverage(z, codes, n_cells, spec, n_splits=5, n_repeats=3, random_state=4)
        assert b.fixed_schema == ref


def test_nested_cv_on_a_planted_cell_is_stable_and_agrees_with_fixed() -> None:
    X, Z = _planted(12, n=1200)
    res = nested_crossvalidation(X, Z, positive_class="1", center_spec=CenterSpec(tau=0.8), max_d=2, n_repeats=2)
    b = res.branches[2]
    assert b.full_data_winner == (0, 1)
    assert b.stability.share_equal_to_reference == 1.0
    assert b.nested.per_split == b.fixed_schema.per_split
    assert res.select_dimensionality() == 2


def test_nested_cv_is_undetermined_below_the_power_floor() -> None:
    X, _ = _noise(13, n=300)
    Z = np.array(["1"] * 10 + ["0"] * 290)
    res = nested_crossvalidation(X, Z, positive_class="1", center_spec=CenterSpec(tau=0.5))
    assert res.undetermined_reason is not None
    assert res.branches == {}
    assert res.select_dimensionality() is None
