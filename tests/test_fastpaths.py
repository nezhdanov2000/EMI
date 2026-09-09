"""
Regression witnesses for the exact-equivalent fast paths (2026-09).

Every optimisation in this batch is a re-implementation, not a new
algorithm: the exhaustive search, the cell-code construction, the adaptive
coarsening, the continued-fraction evaluation, the binomial certification
thresholds and the renderer's cell statistics all had a simpler, slower
form before, and each test here pins the fast form against that simpler
form (re-stated inline where the original was deleted) on inputs chosen to
exercise ties, capacity coarsening, empty cells and both centre rules.
"""

from __future__ import annotations

import dataclasses
import itertools
from math import lgamma

import numpy as np
import pytest

import vsf.avr as avr_mod
import vsf.centers as centers_mod
from vsf.avr import (
    _CandidateFactory,
    _exhaustive_search,
    discover_branches,
    discover_branches_by_value,
)
from vsf.centers import (
    CenterSpec,
    _betainc,
    _betacf,
    center_report,
    center_summary,
    coverage_score,
    min_successes_to_certify,
    purity_bounds,
)
from vsf.metrics import cell_codes, dense_codes_from_flat
from vsf.pmd import adaptively_coarsen_bins, check_grid_capacity, discretize_dataset


# ---------------------------------------------------------------------------
# Reference forms
# ---------------------------------------------------------------------------

def _reference_subset_codes(X_discrete, bin_counts, n_samples, combo, ordered=None):
    """The straightforward form of `vsf.avr._subset_codes`: occupancy check, then merge."""
    X_S = X_discrete[:, combo]
    if not check_grid_capacity(X_S, n_samples):
        flags = None if ordered is None else [ordered[j] for j in combo]
        X_S = adaptively_coarsen_bins(X_S, n_samples, ordered=flags)
    return cell_codes(X_S)


def _reference_search(X_discrete, bin_counts, z_binary, spec, max_d):
    """The pre-2026-09 search loop: `coverage_score` per subset, first wins."""
    n = int(z_binary.shape[0])
    best = {}
    for d in range(1, max_d + 1):
        for combo in itertools.combinations(range(X_discrete.shape[1]), d):
            codes, n_cells = _reference_subset_codes(X_discrete, bin_counts, n, combo)
            key = coverage_score(z_binary, codes, n_cells, spec)
            if d not in best or key > best[d][0]:
                best[d] = (key, combo)
    return best


def _reference_betacf(a, b, x, max_iter=400, eps=3e-16, check_every=16):
    """The pre-2026-09 `_betacf`: whole-array Lentz, no compaction."""
    tiny = centers_mod._TINY
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < tiny, tiny, d)
    d = 1.0 / d
    h = d.copy()
    for m in range(1, max_iter + 1):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if m % check_every == 0 and np.all(np.abs(delta - 1.0) < eps):
            break
    return h


def _reference_min_successes_to_certify(n_values, tau, alpha_eff):
    """The pre-2026-09 per-element `lgamma` form."""
    n_arr = np.asarray(n_values, dtype=np.int64)
    out = np.zeros(n_arr.shape, dtype=np.int64)
    log_tau, log_1mtau = float(np.log(tau)), float(np.log1p(-tau))
    for n in np.unique(n_arr[n_arr > 0]).tolist():
        k = np.arange(n + 1, dtype=np.float64)
        log_choose = (
            lgamma(n + 1.0)
            - np.array([lgamma(v + 1.0) for v in k.tolist()])
            - np.array([lgamma(n - v + 1.0) for v in k.tolist()])
        )
        pmf = np.exp(log_choose + k * log_tau + (n - k) * log_1mtau)
        survival = np.cumsum(pmf[::-1])[::-1]
        hit = np.nonzero(survival <= alpha_eff)[0]
        out[n_arr == n] = int(hit[0]) if hit.size > 0 else int(n) + 1
    out[n_arr <= 0] = 1
    return out


def _mixed_dataset(seed: int, n: int = 600, m: int = 6):
    """Strings, a binary numeric, a 30-level integer (forces coarsening)."""
    rng = np.random.default_rng(seed)
    cols = [
        rng.choice(["a", "b", "c"], size=n),
        rng.choice(["x", "y"], size=n),
        rng.integers(0, 2, size=n).astype(str),
        rng.integers(0, 30, size=n).astype(str),
        rng.choice(["p", "q", "r", "s"], size=n),
        rng.integers(0, 12, size=n).astype(str),
    ][:m]
    X = np.column_stack(cols)
    # A target correlated with columns 0 and 4 so that centres exist, plus
    # enough noise that many subsets tie exactly on the first key components.
    z = ((X[:, 0] == "a") & (X[:, 4] == "p")) | (rng.random(n) < 0.05)
    return X, z.astype(np.int8)


# ---------------------------------------------------------------------------
# Cell codes and coarsening
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_nominal", [3, 257, 1 << 16, 1 << 25])
def test_dense_codes_from_flat_equals_unique_inverse(n_nominal):
    rng = np.random.default_rng(n_nominal)
    flat = rng.integers(0, n_nominal, size=5000)
    expected = np.unique(flat, return_inverse=True)[1].ravel()
    np.testing.assert_array_equal(dense_codes_from_flat(flat, n_nominal), expected)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_candidate_factory_matches_per_subset_coarsening(seed):
    X, _ = _mixed_dataset(seed)
    X_discrete, bin_counts = discretize_dataset(X)
    n = X.shape[0]
    factory = _CandidateFactory(X_discrete, bin_counts, n)
    seen = []
    for combo, codes, n_cells in factory.iter_candidates(4):
        ref_codes, ref_cells = _reference_subset_codes(X_discrete, bin_counts, n, combo)
        assert n_cells == ref_cells, combo
        np.testing.assert_array_equal(codes, ref_codes, err_msg=str(combo))
        direct, direct_cells = factory.codes(combo)
        np.testing.assert_array_equal(direct, ref_codes, err_msg=str(combo))
        assert direct_cells == ref_cells
        seen.append(combo)
    expected_order = [
        c for d in range(1, 5) for c in itertools.combinations(range(X.shape[1]), d)
    ]
    assert seen == expected_order
    # The dataset must actually have exercised the coarsening branch.
    from vsf.pmd import occupied_cells
    assert any(occupied_cells(X_discrete[:, list(c)]) > factory.capacity for c in expected_order)


# ---------------------------------------------------------------------------
# Exhaustive search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seed,spec",
    [
        (0, CenterSpec()),
        (1, CenterSpec(tau=1.0)),
        (2, CenterSpec(rule="certified", tau=0.7, alpha=0.1)),
        (3, CenterSpec(tau=0.6, min_samples=4)),
    ],
)
def test_exhaustive_search_matches_reference_argmax(seed, spec):
    X, z = _mixed_dataset(seed)
    X_discrete, bin_counts = discretize_dataset(X)
    factory = _CandidateFactory(X_discrete, bin_counts, X.shape[0])
    got = _exhaustive_search(factory, z.astype(np.int64), 2, [1], spec, 4)[0]
    ref = _reference_search(X_discrete, bin_counts, z, spec, 4)
    assert set(got) == set(ref)
    for d in ref:
        assert got[d][1] == ref[d][1], (d, got[d], ref[d])
        assert got[d][0] == ref[d][0], (d, got[d], ref[d])


def test_exhaustive_search_with_no_positives_matches_reference():
    X, z = _mixed_dataset(4)
    z[:] = 0
    X_discrete, bin_counts = discretize_dataset(X)
    factory = _CandidateFactory(X_discrete, bin_counts, X.shape[0])
    got = _exhaustive_search(factory, z.astype(np.int64), 2, [1], CenterSpec(), 2)[0]
    ref = _reference_search(X_discrete, bin_counts, z, CenterSpec(), 2)
    assert {d: v[1] for d, v in got.items()} == {d: v[1] for d, v in ref.items()}
    assert {d: v[0] for d, v in got.items()} == {d: v[0] for d, v in ref.items()}


def test_discover_branches_by_value_equals_per_value_discover_branches():
    rng = np.random.default_rng(7)
    X, _ = _mixed_dataset(7, n=400, m=5)
    target = np.where(X[:, 0] == "a", "alpha", np.where(X[:, 1] == "x", "beta", "gamma"))
    target = np.where(rng.random(X.shape[0]) < 0.1, "delta", target)
    values = ["alpha", "beta", "gamma", "delta", "absent"]
    by_value = discover_branches_by_value(
        X, target, values, max_d=3, n_permutations_centers=99, cv_repeats=2
    )
    absent = by_value["absent"]
    assert set(absent) == {1, 2, 3}
    assert all(br.centers.n_positive == 0 and br.centers.coverage == 0.0 for br in absent.values())
    for v in values[:-1]:
        z = (target.astype(str) == v).astype(int)
        expected = discover_branches(
            X, z, positive_class=1, max_d=3, n_permutations_centers=99, cv_repeats=2
        )
        got = by_value[v]
        assert set(got) == set(expected), v
        for d in expected:
            assert dataclasses.asdict(got[d]) == dataclasses.asdict(expected[d]), (v, d)


# ---------------------------------------------------------------------------
# Special functions and certification thresholds
# ---------------------------------------------------------------------------

def test_compacted_lentz_matches_whole_array_lentz():
    rng = np.random.default_rng(11)
    n = rng.integers(1, 30000, size=2000).astype(np.float64)
    k = np.clip(np.floor(rng.random(2000) * n), 1, n - 1)
    a, b = k, n - k + 1
    x = np.clip(rng.random(2000), 1e-4, 1 - 1e-4)
    swap = x >= (a + 1.0) / (a + b + 2.0)
    a_eff = np.where(swap, b, a)
    b_eff = np.where(swap, a, b)
    x_eff = np.where(swap, 1.0 - x, x)
    got = _betacf(a_eff, b_eff, x_eff)
    ref = _reference_betacf(a_eff, b_eff, x_eff)
    # A converged element is frozen instead of being multiplied, for as long
    # as the slowest element of the array still runs, by further factors
    # that are within `eps` of 1: each such multiply can move the reference
    # value by one ulp, and a few hundred of them accumulate to ~1e-13
    # relative. The compacted value is the one at the element's own
    # convergence; the two agree to that accumulated rounding, and the
    # 40-halving quantile bisection downstream resolves 1e-12, so no
    # reported bound moved on any shipped dataset.
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)
    # Scalar path (one element) is the same arithmetic in Python floats.
    for i in rng.integers(0, 2000, size=50):
        one = _betacf(a_eff[i : i + 1], b_eff[i : i + 1], x_eff[i : i + 1])
        assert one[0] == got[i]


def test_betainc_scalar_and_vector_paths_agree_exactly():
    rng = np.random.default_rng(5)
    n = rng.integers(1, 40000, size=300).astype(np.float64)
    k = np.clip(np.floor(rng.random(300) * n), 1, n - 1)
    a, b = k, n - k + 1
    x = np.clip(rng.random(300), 1e-3, 0.999)
    vec = _betainc(a, b, x)
    for i in range(300):
        one = _betainc(a[i : i + 1], b[i : i + 1], x[i : i + 1])[0]
        assert one == vec[i]


@pytest.mark.parametrize(
    "tau,alpha_eff", [(0.9, 0.05 / 300), (0.8, 1e-4), (0.99, 0.01), (0.5, 0.2)]
)
def test_min_successes_to_certify_matches_lgamma_reference(tau, alpha_eff):
    rng = np.random.default_rng(int(tau * 100))
    n = np.concatenate([rng.integers(0, 3000, size=300), [0, 1, 2, 3, 5000]])
    np.testing.assert_array_equal(
        min_successes_to_certify(n, tau, alpha_eff),
        _reference_min_successes_to_certify(n, tau, alpha_eff),
    )


def test_clopper_pearson_memo_returns_identical_values_on_repeat():
    rng = np.random.default_rng(3)
    n = rng.integers(1, 5000, size=400)
    k = np.floor(rng.random(400) * (n + 1)).astype(np.int64)
    k = np.minimum(k, n)
    centers_mod._CP_CACHE.clear()
    first = purity_bounds(k, n, 0.05 / 400)
    second = purity_bounds(k, n, 0.05 / 400)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    # And the memo never serves a value computed at a different alpha.
    other = purity_bounds(k, n, 0.05 / 40)
    assert not np.array_equal(first[0][k > 0], other[0][k > 0])


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [CenterSpec(), CenterSpec(rule="certified", tau=0.7)])
def test_center_summary_equals_center_report_scalars(spec):
    X, z = _mixed_dataset(9)
    X_discrete, bin_counts = discretize_dataset(X)
    factory = _CandidateFactory(X_discrete, bin_counts, X.shape[0])
    for combo in [(0,), (0, 4), (1, 3, 5), (0, 1, 2, 4)]:
        codes, n_cells = factory.codes(combo)
        report = center_report(z, codes, n_cells, spec, n_repeats=0)
        cov, k, pur = center_summary(z, codes, n_cells, spec)
        assert (cov, k, pur) == (report.coverage, report.n_centers, report.purity_pooled)


def test_center_report_without_cell_bounds_changes_only_bound_fields():
    X, z = _mixed_dataset(10)
    X_discrete, bin_counts = discretize_dataset(X)
    factory = _CandidateFactory(X_discrete, bin_counts, X.shape[0])
    codes, n_cells = factory.codes((0, 4))
    full = center_report(z, codes, n_cells, n_permutations=99)
    lean = center_report(z, codes, n_cells, n_permutations=99, cell_bounds=False)
    full_d, lean_d = dataclasses.asdict(full), dataclasses.asdict(lean)
    for name in ("centers", "max_purity_lower"):
        full_d.pop(name)
        lean_d.pop(name)
    assert full_d == lean_d
    assert lean.max_purity_lower == 0.0
    assert {c.cell for c in full.centers} == {c.cell for c in lean.centers}
    for c in lean.centers:
        assert (c.purity_lower, c.purity_upper) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Renderer statistics and lattice geometry
# ---------------------------------------------------------------------------

def test_full_cell_statistics_index_matches_row_wise_unique():
    from vsf.vis import _FullCellStatistics

    rng = np.random.default_rng(12)
    n = 700
    cols = [
        rng.choice(["10", "9", "b", "a", "nan"], size=n),
        rng.integers(0, 15, size=n).astype(str),
        rng.choice(["x", "y", "z"], size=n),
    ]
    pos = rng.random(n) < 0.3
    stats = _FullCellStatistics(cols, pos, CenterSpec())
    matrix = np.stack([np.asarray(c).astype(str) for c in cols], axis=1)
    uniq, codes = np.unique(matrix, axis=0, return_inverse=True)
    expected_index = {tuple(row): i for i, row in enumerate(uniq.tolist())}
    assert stats.index == expected_index
    np.testing.assert_array_equal(
        stats.n_per_cell, np.bincount(codes.ravel(), minlength=len(expected_index))
    )
    np.testing.assert_array_equal(
        stats.k_per_cell,
        np.bincount(codes.ravel(), weights=pos.astype(float), minlength=len(expected_index)).astype(int),
    )


@pytest.mark.parametrize("n_cell", [1, 2, 7, 8, 9, 26, 27, 28, 100, 1001])
def test_vectorised_lattice_offsets_equal_scalar_loop(n_cell):
    cx, cy, cz = np.int64(3), np.int64(1), np.int64(2)
    S = int(np.ceil(n_cell ** (1.0 / 3.0)))
    step = 0.0 if S <= 1 else min(0.052, 0.78 / max(S - 1, 1))
    # scalar reference, exactly as the renderer used to loop
    ref = np.zeros((n_cell, 3))
    for rank in range(n_cell):
        i, j, k = rank % S, (rank // S) % S, rank // (S * S)
        ref[rank] = (
            np.round(cx + (i - (S - 1) / 2.0) * step, 4),
            np.round(cy + (j - (S - 1) / 2.0) * step, 4),
            np.round(cz + (k - (S - 1) / 2.0) * step, 4),
        )
    rank = np.arange(n_cell, dtype=np.int64)
    i, j, k = rank % S, (rank // S) % S, rank // (S * S)
    half = (S - 1) / 2.0
    got = np.column_stack([
        np.round(cx + (i - half) * step, 4),
        np.round(cy + (j - half) * step, 4),
        np.round(cz + (k - half) * step, 4),
    ])
    np.testing.assert_array_equal(got, ref)


def test_candidate_factory_matches_reference_with_ordered_columns():
    from vsf.pmd import ordered_columns
    rng = np.random.default_rng(21)
    n = 400
    X = np.empty((n, 4), dtype=object)
    X[:, 0] = rng.normal(size=n)                      # continuous: 400 levels, ordered
    X[:, 1] = rng.integers(0, 40, size=n)              # integer, ordered
    X[:, 2] = rng.choice(["p", "q", "r", "s", "t"], size=n)
    X[:, 3] = rng.choice(["x", "y"], size=n)
    ordered = ordered_columns(X)
    assert ordered == [True, True, False, False]
    X_discrete, bin_counts = discretize_dataset(X)
    factory = _CandidateFactory(X_discrete, bin_counts, n, ordered=ordered)
    for combo, codes, n_cells in factory.iter_candidates(4):
        ref_codes, ref_cells = _reference_subset_codes(X_discrete, bin_counts, n, combo, ordered)
        assert n_cells == ref_cells, combo
        np.testing.assert_array_equal(codes, ref_codes, err_msg=str(combo))
        assert n_cells <= factory.capacity or all(
            len(np.unique(X_discrete[:, j])) == 1 for j in combo
        )
