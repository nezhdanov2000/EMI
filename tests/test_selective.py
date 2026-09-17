"""vsf.selective: post-selection certificates, nested CV, schema stability."""
from __future__ import annotations

import itertools
from math import comb
from typing import Dict, Tuple

import numpy as np
import pytest

from vsf.avr import (
    _exhaustive_search,
    _prepare_search,
    apply_fitted_partition,
    compute_landscape,
    compute_tau_curves,
    discover_branches,
    family_cell_count,
    iter_branches_by_value,
    report_schema,
    resolve_center_spec,
)
from vsf.metrics import cell_codes
from vsf.pmd import discretize_dataset
from vsf.redundancy import collect_centers
from vsf.centers import (
    CenterSpec,
    _cell_counts,
    crossvalidated_coverage,
    min_successes_to_certify,
    min_successes_to_certify_heterogeneous,
    select_centers,
    stratified_repeated_kfold,
)
from vsf.selective import (
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


def _poisson_binomial_upper_tail(p: np.ndarray, k: int) -> float:
    """Exact P(sum of independent Bernoulli(p_i) >= k), by convolution."""
    dist = np.array([1.0])
    for pi in p:
        dist = np.convolve(dist, [1.0 - pi, pi])
    return float(dist[k:].sum()) if k < dist.size else 0.0


def test_plain_binomial_threshold_is_anti_conservative_for_heterogeneous_rows() -> None:
    # The counterexample in min_successes_to_certify_heterogeneous: the
    # binomial threshold certifies k = 1, and rows (0.03, 0, 0) of mean 0.01
    # reach k = 1 more often than the binomial tail says.
    k = int(min_successes_to_certify([3], 0.01, 0.0298)[0])
    assert k == 1
    binomial = 1 - 0.99 ** 3
    assert binomial <= 0.0298
    assert _poisson_binomial_upper_tail(np.array([0.03, 0.0, 0.0]), 1) > 0.0298
    # the guarded threshold refuses it
    assert int(min_successes_to_certify_heterogeneous([3], 0.01, 0.0298)[0]) == 2


@pytest.mark.parametrize("tau", [0.01, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("level", [0.3, 0.05, 1e-3])
def test_heterogeneous_threshold_bounds_every_mean_tau_poisson_binomial(tau: float, level: float) -> None:
    # Numerical illustration of the proposition (the proof is in the
    # docstring): for small n, rows at the boundary pbar = tau in several
    # heterogeneous shapes never exceed the level at the guarded threshold,
    # and the threshold always satisfies condition (i).
    rng = np.random.default_rng(0)
    for n in range(1, 13):
        k = int(min_successes_to_certify_heterogeneous([n], tau, level)[0])
        assert k >= int(np.ceil(n * tau)) + 1 or k == n + 1
        if k > n:
            continue
        shapes = [np.full(n, tau)]
        lumped = np.zeros(n)
        mass = n * tau
        for i in range(n):
            lumped[i] = min(1.0, mass)
            mass -= lumped[i]
        shapes.append(lumped)
        for _ in range(20):
            w = rng.random(n)
            q = w / w.sum() * n * tau
            if np.all(q <= 1.0):
                shapes.append(q)
        for q in shapes:
            assert _poisson_binomial_upper_tail(q, k) <= level * (1 + 1e-12)


def test_heterogeneous_threshold_is_the_binomial_one_where_the_guard_is_slack() -> None:
    n = np.arange(1, 2001)
    for tau in (0.3, 0.7, 0.9):
        for level in (0.05, 1e-3, 1e-7):
            plain = min_successes_to_certify(n, tau, level)
            guarded = min_successes_to_certify_heterogeneous(n, tau, level)
            assert np.all(guarded >= plain)
            # at levels this small the guard never binds for n * tau >= 1;
            # only the rounding margin can move a threshold, by at most one
            assert np.all(guarded - plain <= 1)
            assert np.mean(guarded == plain) > 0.99


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


def test_family_test_count_covers_search_and_displayed_partitions() -> None:
    # 300 rows: capacity 30, so the 4-level 3- and 4-column partitions are
    # coarsened for the search while the display draws them uncoarsened.
    X, Z = _noise(4, n=300, m=5, levels=4)
    spec = CenterSpec(tau=0.5)
    factory, _, _ = _factory(X, Z, spec)
    Xd, _ = discretize_dataset(X)
    row_sets = set()
    listed = 0
    coarsened = 0
    for d in range(1, 5):
        for combo in itertools.combinations(range(5), d):
            search_codes, n_search = factory.codes(combo)
            raw_codes, n_raw = cell_codes(Xd[:, combo])
            partitions = [search_codes] + ([raw_codes] if n_raw != n_search else [])
            coarsened += int(n_raw != n_search)
            for codes in partitions:
                codes = np.asarray(codes)
                for c in np.unique(codes):
                    listed += 1
                    row_sets.add(np.flatnonzero(codes == c).tobytes())
    assert coarsened > 0
    # one hypothesis per distinct row set: duplicates share n, k and purity
    assert len(row_sets) < listed
    assert family_test_count(factory, 4) == len(row_sets)
    assert family_cell_count(factory, 2) < family_cell_count(factory, 4)


def test_family_multiplicity_is_resolved_by_every_search_entry_point() -> None:
    X, Z = _planted(14, n=400)
    spec = CenterSpec(tau=0.8, rule="certified", multiplicity="family")
    assert not spec.is_resolved
    with pytest.raises(ValueError):
        spec.effective_alpha(3)
    factory, _, _ = _factory(X, Z, spec)
    t = family_cell_count(factory, 4)
    resolved = resolve_center_spec(factory, spec, 4)
    assert resolved.family_tests == t and resolved.effective_alpha(1) == pytest.approx(0.05 / t)
    assert resolve_center_spec(factory, resolved, 2) is resolved
    branches = discover_branches(X, Z, positive_class="1", center_spec=spec, n_permutations_centers=19)
    assert {b.centers.spec.family_tests for b in branches.values()} == {t}
    assert all(b.centers.alpha_effective == pytest.approx(0.05 / t) for b in branches.values())
    opened = report_schema(X, Z, [0, 1], positive_class="1", center_spec=spec)
    assert opened[2].centers.spec.family_tests == t
    land = compute_landscape(X, Z, positive_class="1", center_spec=spec)
    ref = compute_landscape(X, Z, positive_class="1", center_spec=resolved)
    assert np.array_equal(land.n_centers, ref.n_centers)
    curves = compute_tau_curves(X, Z, positive_class="1", center_spec=spec)
    assert curves["curves"]["2"]["n_centers"]
    catalog = collect_centers(X, Z, positive_class="1", center_spec=spec)
    assert catalog.spec.family_tests == t
    by_value = dict(iter_branches_by_value(X, Z, ["1"], center_spec=spec, n_permutations_centers=0, cv_repeats=0))
    assert by_value["1"][2].centers.spec.family_tests == t


def test_family_spec_validation() -> None:
    with pytest.raises(ValueError):
        CenterSpec(tau=0.8, rule="purity", multiplicity="family")
    with pytest.raises(ValueError):
        CenterSpec(tau=0.8, rule="certified", multiplicity="family", method="wilson")
    assert CenterSpec(tau=0.8, rule="certified", multiplicity="family", alpha=0.1).alpha == 0.1
    with pytest.raises(ValueError):
        CenterSpec(tau=0.8, rule="certified", multiplicity="family", family_tests=0)
    with pytest.raises(ValueError):
        CenterSpec(tau=0.8, rule="certified", family_tests=10)
    assert CenterSpec(tau=0.8, rule="certified", multiplicity="family", family_tests=10).effective_alpha(99) == pytest.approx(0.005)


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
    ok = (n >= m) & (k >= min_successes_to_certify_heterogeneous(n, tau, level))
    return int(k[ok].sum()), int(ok.sum())


def test_family_bonferroni_tests_every_eligible_cell_at_alpha_over_t() -> None:
    X, Z = _planted(5)
    spec = CenterSpec(tau=0.8, alpha=0.05, min_samples=2)
    res = certify_discovery(X, Z, positive_class="1", center_spec=spec, method="family_bonferroni")
    factory, z, _ = _factory(X, Z, spec)
    t = family_test_count(factory, 4)
    assert set(res.n_tests.values()) == {t}
    level = 0.05 / t
    for d, br in res.branches.items():
        codes, n_cells = factory.codes(br.features)
        k, n = _cell_counts(z, codes, n_cells)
        assert sorted(c.cell for c in br.cells) == np.nonzero(n >= 2)[0].tolist()
        for c in br.cells:
            assert (c.k, c.n) == (int(k[c.cell]), int(n[c.cell]))
            assert c.certified == (c.k >= min_successes_to_certify_heterogeneous([c.n], 0.8, level)[0])
            if c.certified:
                assert c.p_value <= level
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
    res = nested_crossvalidation(X, Z, positive_class="1", center_spec=spec, max_d=3, n_repeats=2,
                                 random_state=3, encoding="all_rows")
    assert res.encoding == "all_rows"
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


@pytest.mark.parametrize("spec", [CenterSpec(tau=0.8), CenterSpec(tau=0.8, rule="certified", multiplicity="family")])
def test_nested_cv_train_encoding_uses_nothing_from_the_test_rows(spec: CenterSpec) -> None:
    # 300 rows, 4-level columns: capacity binds, so the fold's merges matter.
    rng = np.random.default_rng(15)
    X = rng.integers(0, 4, size=(300, 5))
    inside = (X[:, 0] == 0) & (X[:, 1] < 2)
    Z = np.where(rng.random(300) < np.where(inside, 0.97, 0.15), "1", "0")
    X = X.astype(str)
    res = nested_crossvalidation(X, Z, positive_class="1", center_spec=spec, max_d=3,
                                 n_repeats=2, random_state=5)
    assert res.encoding == "train"
    factory, z, _ = _factory(X, Z, spec)
    splits = stratified_repeated_kfold(z, 5, 2, 5)
    for i, (train, test) in enumerate(splits):
        # The fold's procedure, rebuilt from the training rows ALONE: the
        # test rows' features are replaced by garbage that cannot matter.
        X_train_only = X.copy()
        X_train_only[test] = X[train[0]]
        fit, zt, _ = _factory(X_train_only[train], np.where(z[train] == 1, "1", "0"), spec)
        fold_spec = resolve_center_spec(fit, spec, 3)
        best = _exhaustive_search(fit, zt.astype(np.int64), 2, [1], fold_spec, 3)[0]
        for d, b in res.branches.items():
            combo = tuple(best[d][1])
            assert b.fold_winners[i] == combo
            train_codes, n_train_cells = fit.codes(combo)
            k_tr, n_tr = _cell_counts(zt, train_codes, n_train_cells)
            mask, _ = select_centers(k_tr, n_tr, fold_spec)
            # test rows scored through the fitted partition
            codes_all, n_all = apply_fitted_partition(fit, factory._raw, combo)
            # a training cell and its image are the same cell
            image = {}
            for tc, ac in zip(train_codes.tolist(), codes_all[train].tolist()):
                assert image.setdefault(tc, ac) == ac
            centre_images = {image[c] for c in np.nonzero(mask)[0].tolist()}
            k_te = sum(int(z[r]) for r in test.tolist() if int(codes_all[r]) in centre_images)
            expected = k_te / int(z[test].sum())
            assert b.nested.per_split[i] == pytest.approx(expected)


def test_apply_fitted_partition_reproduces_the_fitted_partition_and_maps_unseen_levels() -> None:
    rng = np.random.default_rng(16)
    X = rng.integers(0, 6, size=(400, 4)).astype(str)
    Z = np.where(rng.random(400) < 0.3, "1", "0")
    factory, _, _ = _factory(X, Z, CenterSpec(tau=0.5))
    train = np.flatnonzero(factory._raw[:, 0] != 5)  # level 5 of column 0 unseen in training
    fit = type(factory)(factory._raw[train], factory.bin_counts, int(train.size), ordered=factory.ordered)
    merged = 0
    for combo in itertools.combinations(range(4), 3):
        codes_fit, _ = fit.codes(combo)
        codes_all, _ = apply_fitted_partition(fit, factory._raw, combo)
        pairs = set(zip(codes_fit.tolist(), codes_all[train].tolist()))
        assert len(pairs) == len(set(codes_fit.tolist())) == len(set(codes_all[train].tolist()))
        k = fit.merge_levels(combo)
        if k is not None and k[0] < fit._levels[0]:
            merged += 1
            # the unseen level joins column 0's "other" group
            unseen = np.flatnonzero(factory._raw[:, 0] == 5)
            other = train[np.flatnonzero(~np.isin(factory._raw[train, 0], fit._orders[0][: k[0] - 1]))]
            same_rest = other[np.all(factory._raw[other][:, list(combo[1:])] == factory._raw[unseen[0], list(combo[1:])], axis=1)] \
                if 0 in combo and combo[0] == 0 else np.empty(0, dtype=np.int64)
            if same_rest.size:
                assert codes_all[unseen[0]] == codes_all[same_rest[0]]
    assert merged > 0


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


# ---------------------------------------------------------------------------
# /api/validate
# ---------------------------------------------------------------------------
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

import pandas as pd  # noqa: E402

import vsf.server as vsf_server  # noqa: E402
from vsf.server import _build_server  # noqa: E402


def _planted_frame(seed: int = 21, n: int = 900) -> pd.DataFrame:
    X, Z = _planted(seed, n)
    df = pd.DataFrame(X, columns=["a", "b", "c", "e"])
    df["target"] = np.where(Z == "1", "yes", "no")
    return df


class _Server:
    def __init__(self, df: pd.DataFrame) -> None:
        self.httpd = _build_server(df, host="127.0.0.1", port=0, prefetch=False)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]

    def post(self, body: dict) -> Tuple[int, dict]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/validate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def wait(self, body: dict, timeout: float = 60.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, data = self.post(body)
            assert status == 200, data
            if data["status"] != "running":
                return data
            time.sleep(0.05)
        raise AssertionError("validation did not finish")

    def close(self) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()


_BODY = {"target": "target", "criterion": "yes", "tau": 0.8, "alpha": 0.05,
         "method": "family_bonferroni", "repeats": 1}


def test_validate_endpoint_reports_the_library_results() -> None:
    df = _planted_frame()
    srv = _Server(df)
    try:
        status, first = srv.post(_BODY)
        assert status == 200 and first["status"] in ("running", "done")
        assert first["progress"]["total"] == 2 + 1 + 5
        data = srv.wait(_BODY)
        assert data["status"] == "done", data["error"]
        assert data["progress"]["done"] == data["progress"]["total"]
        result = data["result"]
    finally:
        srv.close()

    X = df[["a", "b", "c", "e"]].values
    Z = (df["target"] == "yes").astype(int).values
    spec = CenterSpec(tau=0.8, alpha=0.05)
    cert = certify_discovery(X, Z, ["a", "b", "c", "e"], positive_class=1, center_spec=spec,
                             method="family_bonferroni")
    nested = nested_crossvalidation(X, Z, ["a", "b", "c", "e"], positive_class=1,
                                    center_spec=spec, n_repeats=1)
    for d, br in cert.branches.items():
        got = result["certificate"]["branches"][str(d)]
        assert got["features"] == list(br.feature_names)
        assert got["n_certified"] == br.n_certified
        assert got["n_tested"] == len(br.cells)
        assert got["coverage_eval"] == pytest.approx(br.coverage_eval)
    assert result["certificate"]["n_tests"]["2"] == cert.n_tests[2]
    planted = result["certificate"]["branches"]["2"]["cells"][0]
    assert planted["conditions"] == [
        {"column": "a", "values": ["0"]}, {"column": "b", "values": ["0"]},
    ]
    for d, nb in nested.branches.items():
        got = result["nested"]["branches"][str(d)]
        assert got["nested"]["mean"] == pytest.approx(nb.nested.mean)
        assert got["fixed_schema"]["mean"] == pytest.approx(nb.fixed_schema.mean)
        assert got["stability"]["share_same"] == pytest.approx(nb.stability.share_equal_to_reference)
        assert got["winner"] == list(nb.full_data_winner_names)
    assert result["nested"]["d_star"] == nested.select_dimensionality()


def test_validate_endpoint_rejects_invalid_requests() -> None:
    srv = _Server(_planted_frame())
    try:
        for patch in (
            {"criterion": None},
            {"method": "westfall_young"},
            {"repeats": 0},
            {"repeats": 11},
            {"tau": 1.0},
            {"target": "nope"},
        ):
            status, data = srv.post({**_BODY, **patch})
            assert status == 400, (patch, data)
            assert data["error"]
    finally:
        srv.close()


def test_validate_endpoint_runs_one_job_at_a_time_and_retries_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    calls = {"n": 0}
    real = vsf_server.nested_crossvalidation

    def gated(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            gate.wait(timeout=30)
            raise RuntimeError("injected failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(vsf_server, "nested_crossvalidation", gated)
    srv = _Server(_planted_frame())
    try:
        status, data = srv.post(_BODY)
        assert status == 200 and data["status"] == "running"
        status, busy = srv.post({**_BODY, "tau": 0.85})
        assert status == 409 and busy["running"]["tau"] == 0.8
        status, same = srv.post(_BODY)  # polling the running job is not a conflict
        assert status == 200 and same["status"] == "running"
        gate.set()
        failed = srv.wait(_BODY)
        assert failed["status"] == "error" and "injected failure" in failed["error"]
        status, again = srv.post(_BODY)  # a poll keeps reporting the failure
        assert status == 200 and again["status"] == "error"
        status, restarted = srv.post({**_BODY, "retry": True})
        assert status == 200 and restarted["status"] in ("running", "done")
        retried = srv.wait(_BODY)
        assert retried["status"] == "done"
        assert calls["n"] == 2
    finally:
        gate.set()
        srv.close()


def test_the_page_colours_by_observed_share_only() -> None:
    """The interface offers no certificate colouring (decision of 2026-09-17)."""
    srv = _Server(_planted_frame(n=200))
    try:
        base = f"http://127.0.0.1:{srv.port}"
        with urllib.request.urlopen(base + "/", timeout=10) as r:
            html = r.read().decode("utf-8")
        with urllib.request.urlopen(base + "/static/js/app.js", timeout=10) as r:
            js = r.read().decode("utf-8")
    finally:
        srv.close()
    assert 'id="certMode"' not in html and 'id="validationPanel"' not in html
    assert "'/api/validate'" not in js
    assert "readCertMultiplicity" not in js


# ---------------------------------------------------------------------------
# multiplicity in the live application
# ---------------------------------------------------------------------------
def _post_json(srv: "_Server", path: str, body: dict) -> Tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{srv.port}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_analyze_certifies_over_the_family_by_default_under_the_certified_rule() -> None:
    df = _planted_frame()
    srv = _Server(df)
    try:
        body = {"target": "target", "criterion": "yes", "tau": 0.8, "rule": "certified"}
        status, fam = _post_json(srv, "/api/analyze", body)
        assert status == 200, fam
        status, legacy = _post_json(srv, "/api/analyze", {**body, "multiplicity": "bonferroni"})
        assert status == 200, legacy
        status, purity = _post_json(srv, "/api/analyze", {**body, "rule": "purity"})
        assert status == 200, purity
    finally:
        srv.close()

    X = df[["a", "b", "c", "e"]].values
    Z = (df["target"] == "yes").astype(int).values
    factory, _, _ = _prepare_search(X, Z, None, 1, CenterSpec(tau=0.8), "presence")
    t = family_cell_count(factory, 4)

    cert = fam["certificate"]
    assert cert["multiplicity"] == "family" and cert["valid_after_search"] is True
    assert cert["family_tests"] == t
    assert cert["per_cell_level"] == pytest.approx(0.05 / t)
    # a fully pure cell of n rows has p = tau ** n
    n_min = cert["min_certifiable_rows"]
    assert 0.8 ** n_min <= 0.05 / t < 0.8 ** (n_min - 1)
    ref = discover_branches(
        X, Z, positive_class=1, n_permutations_centers=0, cv_repeats=0,
        center_spec=CenterSpec(tau=0.8, rule="certified", multiplicity="family"),
    )
    for d, payload in fam["branches"].items():
        assert payload["certificate"]["multiplicity"] == "family"
        assert payload["certificate"]["family_tests"] == t
        assert payload["certificate"]["alpha_effective"] == pytest.approx(0.05 / t)
        assert payload["search_centers"]["n_centers"] == ref[int(d)].centers.n_centers
        # every green cell of the display is certified at the family level
        for cell in payload["centers"]["top"]:
            assert exact_upper_tail(cell["k"], cell["n"], 0.8)[0] <= 0.05 / t

    assert legacy["certificate"]["multiplicity"] == "bonferroni"
    assert legacy["certificate"]["valid_after_search"] is False
    assert legacy["certificate"]["family_tests"] is None
    assert purity["certificate"]["rule"] == "purity"
    assert purity["certificate"]["valid_after_search"] is False
    # the legacy per-schema certificate is never stricter than the family one
    for d in fam["branches"]:
        assert (legacy["branches"][d]["search_centers"]["coverage"]
                >= fam["branches"][d]["search_centers"]["coverage"] - 1e-12)


def test_certificate_request_validation_and_cache_separation() -> None:
    srv = _Server(_planted_frame(n=300))
    try:
        base = {"target": "target", "criterion": "yes", "tau": 0.8}
        for patch in (
            {"rule": "purity", "multiplicity": "family"},
            {"rule": "certified", "multiplicity": "holm"},
            {"rule": "certified", "multiplicity": "family", "tau": 0.0},
            {"rule": "certified", "tau": 1.0},
        ):
            for path in ("/api/analyze", "/api/landscape"):
                status, data = _post_json(srv, path, {**base, **patch})
                assert status == 400, (path, patch, data)
        status, a = _post_json(srv, "/api/landscape", {**base, "rule": "certified"})
        assert status == 200, a
        status, b = _post_json(srv, "/api/landscape", {**base, "rule": "certified", "multiplicity": "bonferroni"})
        assert status == 200, b
        assert a["n_zero"] >= b["n_zero"]
        assert len(srv.httpd.landscape_cache) == 2
    finally:
        srv.close()
