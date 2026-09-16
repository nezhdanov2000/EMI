"""experiments/baselines.py: budget-matched comparison harness (phase 3)."""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType
from typing import List

import numpy as np
import pytest

pytest.importorskip("sklearn")

from vsf.avr import _CandidateFactory, _prepare_search  # noqa: E402
from vsf.centers import CenterSpec  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vsf_experiments_baselines", _ROOT / "experiments" / "baselines.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bl = _load()


def _two_region_data(seed: int, n: int = 1200):
    """Positives at (x0=0, x1=0) and at (x2=0, x3=0): two rules on different columns, no common 2-D grid."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, 5))
    a = (X[:, 0] == 0) & (X[:, 1] == 0)
    b = (X[:, 2] == 0) & (X[:, 3] == 0)
    z = np.where(rng.random(n) < np.where(a | b, 0.97, 0.03), 1, 0)
    return X.astype(str), z


def _factory(X, z, tau=0.9, m=1):
    prepared = _prepare_search(X, z, None, 1, CenterSpec(tau=tau, min_samples=m), "presence")
    assert prepared is not None
    return prepared


def test_budgeted_greedy_matches_brute_force_on_small_instances() -> None:
    rng = np.random.default_rng(0)
    for _ in range(30):
        n_train = 40
        cands = [
            bl._Candidate(positives=np.flatnonzero(rng.random(n_train) < 0.2), cost=int(rng.integers(1, 4)), key=i)
            for i in range(7)
        ]
        res = bl.budgeted_greedy(cands, [1, 3, 5], n_train)
        for budget, (chosen, covered) in res.items():
            assert sum(cands[i].cost for i in chosen) <= budget
            union = set()
            for i in chosen:
                union |= set(cands[i].positives.tolist())
            assert covered == len(union)
            best = 0
            for r in range(len(cands) + 1):
                for sub in itertools.combinations(range(len(cands)), r):
                    if sum(cands[i].cost for i in sub) <= budget:
                        u = set()
                        for i in sub:
                            u |= set(cands[i].positives.tolist())
                        best = max(best, len(u))
            # (1 - 1/e) / 2 guarantee
            assert covered >= (1 - 1 / np.e) / 2 * best - 1e-9


def test_select_vsf_is_the_best_schema_prefix_by_brute_force() -> None:
    X, z = _two_region_data(1, n=600)
    factory, zb, names = _factory(X, z)
    train = np.arange(0, 600, 2)
    fit = _CandidateFactory(factory._raw[train], factory.bin_counts, train.size, ordered=factory.ordered)
    res = bl.select_vsf(fit, factory._raw, train, zb[train], 0.9, 1, [2, 4, 8], names)
    zt = zb[train].astype(np.int64)
    for budget in (2, 4, 8):
        best = 0
        for d in range(1, 5):
            for combo in itertools.combinations(range(5), d):
                codes, nc = fit.codes(combo)
                t = np.bincount(codes * 2 + zt, minlength=2 * nc).reshape(nc, 2)
                k, n = t[:, 1], t.sum(axis=1)
                q = np.sort(k[bl._qualifying(k, n, 0.9, 1)])[::-1]
                best = max(best, int(q[: budget // d].sum()))
        assert res.train_coverage[budget] == pytest.approx(best / zt.sum())
        groups = res.groups[budget]
        assert sum(g.cost for g in groups) <= budget
        # groups of one schema are disjoint and equally costly
        assert len({g.cost for g in groups}) <= 1
        if groups:
            stacked = np.vstack([g.members for g in groups])
            assert stacked.sum(axis=0).max() <= 1
            covered_train = int((stacked.any(axis=0)[train] & (zt == 1)).sum())
            assert covered_train == best


def test_greedy_chain_is_nested() -> None:
    X, z = _two_region_data(2, n=600)
    factory, zb, _ = _factory(X, z)
    fit = _CandidateFactory(factory._raw, factory.bin_counts, factory.n_samples, ordered=factory.ordered)
    chain = bl.greedy_chain(fit, zb, CenterSpec(tau=0.9))
    assert [len(c) for c in chain] == [1, 2, 3, 4]
    for a, b in zip(chain, chain[1:]):
        assert set(a) < set(b)


def test_rules_describe_what_one_grid_cannot() -> None:
    X, z = _two_region_data(3)
    res = bl.run_comparison(X, z, 1, tau=0.9, m=5, n_repeats=1, budgets=(4,), methods=("vsf", "rules"))
    vsf = res["summary"]["vsf"][4]["coverage"].mean
    rules = res["summary"]["rules"][4]["coverage"].mean
    # two 2-condition rules capture both regions; a 2-D grid holds only one
    # region's cell, a 4-D grid needs one cell per region at cost 4 each
    # at most ~0.90 of positives lie in the two regions (3 % background)
    assert rules > 0.85
    assert vsf < 0.6
    assert rules - vsf > 0.3


def test_run_comparison_respects_budgets_and_pairs_folds() -> None:
    X, z = _two_region_data(4, n=500)
    res = bl.run_comparison(X, z, 1, tau=0.8, m=3, n_repeats=1, budgets=(1, 2, 4))
    for mth in bl.METHODS:
        for b in (1, 2, 4):
            entry = res["summary"][mth][b]
            assert entry["cost_mean"] <= b
            assert len(entry["coverage"].per_split) == 5
    with pytest.raises(ValueError):
        bl.run_comparison(X, z, 1, tau=0.8, m=3, methods=("svm",))


def test_evaluate_groups_counts_the_union_once() -> None:
    z = np.array([1, 1, 0, 1, 0, 1])
    test = np.array([0, 1, 2, 3])
    g1 = bl.Group(members=np.array([1, 1, 0, 0, 0, 0], bool), cost=1, label="a")
    g2 = bl.Group(members=np.array([0, 1, 1, 0, 0, 0], bool), cost=2, label="b")
    cov, pur, spent = bl.evaluate_groups([g1, g2], z, test)
    assert cov == pytest.approx(2 / 3) and pur == pytest.approx(2 / 3) and spent == 3
    cov, pur, spent = bl.evaluate_groups([], z, test)
    assert cov == 0.0 and np.isnan(pur) and spent == 0


def test_pure_union_keeps_training_purity_of_the_union() -> None:
    rng = np.random.default_rng(7)
    X = rng.integers(0, 3, size=(800, 5))
    p = np.where((X[:, 0] == 0) | (X[:, 1] == 0), 0.8, 0.1)
    z = (rng.random(800) < p).astype(int)
    factory, zb, names = _factory(X.astype(str), z, tau=0.7)
    train = np.arange(0, 800, 2)
    res = bl.select_rules(factory._raw, train, zb[train], 0.7, 5, [8, 16], names, 3, pure_union=True)
    free = bl.select_rules(factory._raw, train, zb[train], 0.7, 5, [8, 16], names, 3)
    for b in (8, 16):
        groups = res.groups[b]
        assert groups
        inside = np.zeros(800, dtype=bool)
        for g in groups:
            inside |= g.members
        rows = train[inside[train]]
        assert zb[rows].mean() >= 0.7 - 1e-9
        assert res.train_coverage[b] <= free.train_coverage[b] + 1e-12
    with pytest.raises(ValueError):
        bl.budgeted_greedy([bl._Candidate(positives=np.array([0]), cost=1, key=0)], [1], 2, min_union_purity=0.5)


def test_net_coverage_is_the_purity_constrained_lagrangian() -> None:
    # pure union: net coverage equals coverage
    assert bl.net_coverage(30, 30, 60, 0.9) == pytest.approx(0.5)
    # union exactly at tau adds nothing; below tau is negative; above is positive
    assert bl.net_coverage(90, 100, 200, 0.9) == pytest.approx(0.0)
    assert bl.net_coverage(80, 100, 200, 0.9) < 0.0
    assert bl.net_coverage(95, 100, 200, 0.9) > 0.0
    # sign agrees with purity >= tau for every small count
    for n in range(1, 25):
        for k in range(n + 1):
            for tau in (0.5, 0.7, 0.9):
                u = bl.net_coverage(k, n, 10, tau)
                assert (u >= -1e-12) == (k >= tau * n - 1e-12)
    assert bl.net_coverage(0, 0, 0, 0.7) == 0.0
    with pytest.raises(ValueError):
        bl.net_coverage(1, 1, 1, 1.0)
    with pytest.raises(ValueError):
        bl.net_coverage(2, 1, 1, 0.5)


def test_description_stability() -> None:
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([0, 1, 1, 0], bool)
    empty = np.zeros(4, bool)
    assert bl.description_stability([a, a, a]) == pytest.approx(1.0)
    assert bl.description_stability([a, ~a]) == pytest.approx(0.0)
    assert bl.description_stability([a, b]) == pytest.approx(1 / 3)
    # (a, empty) counts as 0, (empty, empty) is skipped
    assert bl.description_stability([a, empty, empty]) == pytest.approx(0.0)
    assert np.isnan(bl.description_stability([empty, empty]))
    assert np.isnan(bl.description_stability([a]))


def test_split_outcome_matches_evaluate_groups_and_summary_fields() -> None:
    X, z = _two_region_data(5, n=500)
    out = bl.run_split(X, z, 1, 0.8, 3, split_index=0, n_repeats=1, budgets=(2, 4))
    for mth, per in out.items():
        for b, o in per.items():
            assert o.inside.shape == (500,)
            assert 0 <= o.k_in <= o.n_in and o.n_pos > 0
            if o.n_in:
                assert o.net_coverage(0.8) == pytest.approx(
                    (o.k_in - 4.0 * (o.n_in - o.k_in)) / o.n_pos)
            assert o.seconds >= 0.0
    res = bl.run_comparison(X, z, 1, tau=0.8, m=3, n_repeats=1, budgets=(4,))
    for mth in bl.METHODS:
        e = res["summary"][mth][4]
        assert e["net_coverage"].mean <= e["coverage"].mean + 1e-12
        assert 0.0 <= e["stability"] <= 1.0
        assert 0.0 <= e["purity_reaches_tau"] <= 1.0


def test_tree_grid_never_does_worse_on_training_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    X, z = _two_region_data(6, n=800)
    factory, zb, _ = _factory(X, z, tau=0.9, m=5)
    train = np.arange(0, 800, 2)
    full = bl.select_tree(factory._raw, train, zb[train], 0.9, 5, [2, 4, 8])
    monkeypatch.setattr(bl, "TREE_GRID", (("entropy", None),))
    single = bl.select_tree(factory._raw, train, zb[train], 0.9, 5, [2, 4, 8])
    for b in (2, 4, 8):
        assert full.train_coverage[b] >= single.train_coverage[b] - 1e-12
        assert sum(g.cost for g in full.groups[b]) <= b


def test_certification_threshold_matches_the_product() -> None:
    from vsf.centers import min_successes_to_certify_heterogeneous

    thr = bl.certification_threshold(300, 0.9, 0.05 / 1000)
    assert thr.shape == (301,)
    ref = np.asarray(min_successes_to_certify_heterogeneous(np.arange(1, 301), 0.9, 0.05 / 1000))
    assert np.array_equal(thr[1:], ref)
    # condition (i) of Proposition 3 and "never" encoded as n + 1
    n = np.arange(1, 301)
    assert np.all((thr[1:] >= np.floor(0.9 * n) + 1) | (thr[1:] == n + 1))
    with pytest.raises(ValueError):
        bl._qualifying(np.array([3]), np.array([400]), 0.9, 1, thr)


def test_certified_selection_is_a_stricter_filter_and_rejects_noise() -> None:
    X, z = _two_region_data(8, n=1200)
    factory, zb, names = _factory(X, z, tau=0.8, m=1)
    train = np.arange(0, 1200, 2)
    fit = _CandidateFactory(factory._raw[train], factory.bin_counts, train.size, ordered=factory.ordered)
    from vsf.avr import family_cell_count

    thr = bl.certification_threshold(train.size, 0.8, 0.05 / family_cell_count(fit, 4))
    zt = zb[train]
    selectors = (
        lambda t: bl.select_vsf(fit, factory._raw, train, zt, 0.8, 1, [4, 8], names, threshold=t),
        lambda t: bl.select_rules(factory._raw, train, zt, 0.8, 1, [4, 8], names, 4, threshold=t),
    )
    for select in selectors:
        free, cert = select(None), select(thr)
        for b in (4, 8):
            assert cert.train_coverage[b] <= free.train_coverage[b] + 1e-12
            for g in cert.groups[b]:
                rows = train[g.members[train]]
                assert int(zb[rows].sum()) >= thr[rows.size]
            assert cert.groups[b], "the two planted regions are large enough to certify"
    # pure noise: nothing is certified by the post-selection threshold
    rng = np.random.default_rng(9)
    Xn = rng.integers(0, 3, size=(600, 5)).astype(str)
    zn = (rng.random(600) < 0.5).astype(int)
    out = bl.run_split(Xn, zn, 1, 0.8, 1, split_index=0, n_repeats=1, budgets=(4, 16),
                       methods=("vsf", "rules"), selection="certified")
    for mth in ("vsf", "rules"):
        for b in (4, 16):
            assert out[mth][b].conditions == 0
    with pytest.raises(ValueError):
        bl.run_split(Xn, zn, 1, 0.8, 1, split_index=0, n_repeats=1, selection="bonferroni")
