"""vsf.partial: partial-axis centres on one schema."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from vsf.avr import _prepare_search, discover_branches, resolve_center_spec
from vsf.centers import CenterSpec, coverage_score
from vsf.partial import (
    _Cand,
    _SubCellStore,
    crossvalidate_partial,
    pack_disjoint,
    partial_centres_for_schema,
    search_partial,
)

_TITANIC = Path(__file__).resolve().parents[1] / "benchmark_data" / "titanic.csv"


def _misaligned(n: int = 3000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Truth = (c0=0 & c1=1) | (c2=2 & c3=0): two conjunctions on disjoint column pairs."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, 8))
    truth = ((X[:, 0] == 0) & (X[:, 1] == 1)) | ((X[:, 2] == 2) & (X[:, 3] == 0))
    z = (rng.random(n) < np.where(truth, 0.95, 0.10)).astype(int)
    return X, z


def _titanic() -> Tuple[np.ndarray, np.ndarray]:
    if not _TITANIC.exists():  # pragma: no cover
        pytest.skip("benchmark_data/titanic.csv not present")
    df = pd.read_csv(_TITANIC)
    X = df.drop(columns=["survived"])
    return X.values, (df["survived"].astype(str) == "died").astype(int).values


# --------------------------------------------------------------------------
# pack_disjoint
# --------------------------------------------------------------------------
def _cand(sub, cell, rows, z, order=0) -> _Cand:
    rows = np.asarray(rows, dtype=bool)
    return _Cand(tuple(sub), cell, int(rows.sum()), int(z[rows].sum()), order, rows)


def test_pack_disjoint_is_disjoint_and_maximal() -> None:
    rng = np.random.default_rng(1)
    n = 60
    z = rng.integers(0, 2, n)
    cands = []
    for i in range(40):
        rows = rng.random(n) < 0.15
        if rows.any():
            cands.append(_cand((i % 5,), i, rows, z, order=i))
    chosen = pack_disjoint(cands)
    union = np.zeros(n, dtype=bool)
    for c in chosen:
        assert not np.any(union & c.rows)
        union |= c.rows
    # maximal: every unchosen candidate hits the union
    chosen_ids = {(c.sub_schema, c.cell) for c in chosen}
    for c in cands:
        if (c.sub_schema, c.cell) not in chosen_ids:
            assert np.any(union & c.rows)


def test_pack_disjoint_prefers_more_positives_then_fewer_conditions() -> None:
    z = np.array([1, 1, 1, 1, 0, 0])
    rows_a = np.array([1, 1, 1, 1, 0, 0], dtype=bool)  # 4 positives, 2 conditions
    rows_b = np.array([1, 1, 0, 0, 0, 0], dtype=bool)  # 2 positives, 1 condition
    rows_c = np.array([1, 1, 1, 1, 0, 0], dtype=bool)  # same rows as a, 1 condition
    chosen = pack_disjoint([_cand((0, 1), 0, rows_a, z), _cand((2,), 0, rows_b, z), _cand((3,), 0, rows_c, z)])
    assert [c.sub_schema for c in chosen] == [(3,)]


# --------------------------------------------------------------------------
# one schema
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["purity", "certified"])
def test_partial_dominates_full_grid_on_every_schema(rule: str) -> None:
    X, z = _misaligned(n=1500)
    spec = CenterSpec(tau=0.7, min_samples=5, rule=rule, multiplicity="family") if rule == "certified" \
        else CenterSpec(tau=0.7, min_samples=5)
    factory, zb, _ = _prepare_search(X, z, None, None, spec, "presence")
    spec_r = resolve_center_spec(factory, spec, 4)
    store = _SubCellStore(factory, zb.astype(np.int64), spec_r, None)
    for d in (1, 2, 3, 4):
        for combo in itertools.combinations(range(6), d):
            codes, n_cells = factory.codes(combo)
            full_cov = coverage_score(zb, codes, n_cells, spec_r)[0]
            sel = partial_centres_for_schema(X, z, combo, spec, max_d=4)
            assert sel.coverage >= full_cov - 1e-12
            # every centre is a real centre of its sub-schema and centres are disjoint
            union = np.zeros(len(z), dtype=bool)
            for c in sel.centres:
                assert set(c.sub_schema) <= set(combo)
                assert c.purity >= spec.tau - 1e-12
                sub_codes, _ = factory.codes(c.sub_schema)
                rows = np.asarray(sub_codes) == c.cell
                assert int(rows.sum()) == c.n and int(zb[rows].sum()) == c.k
                assert not np.any(union & rows)
                union |= rows
            assert sel.n_sel == int(union.sum())
            assert sel.k_sel == int(zb[union].sum())


def test_partial_equals_full_grid_when_no_sub_schema_has_centres() -> None:
    # One 2D truth cell only: sub-schemas of {0,1} have no cell of purity >= 0.9
    rng = np.random.default_rng(3)
    X = rng.integers(0, 3, size=(2000, 4))
    truth = (X[:, 0] == 0) & (X[:, 1] == 1)
    z = (rng.random(2000) < np.where(truth, 0.97, 0.05)).astype(int)
    spec = CenterSpec(tau=0.9, min_samples=5)
    factory, zb, _ = _prepare_search(X, z, None, None, spec, "presence")
    codes, n_cells = factory.codes((0, 1))
    full = coverage_score(zb, codes, n_cells, spec)
    sel = partial_centres_for_schema(X, z, (0, 1), spec)
    assert sel.coverage == pytest.approx(full[0])
    assert all(c.sub_schema == (0, 1) for c in sel.centres)


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
def test_search_partial_finds_misaligned_truth_where_full_grid_cannot() -> None:
    X, z = _misaligned(n=3000)
    spec = CenterSpec(tau=0.7, min_samples=10)
    full = discover_branches(X, z, max_d=4, center_spec=spec, n_permutations_centers=0, cv_splits=2, cv_repeats=1)
    part = search_partial(X, z, spec, max_d=4)
    # Without a condition budget the full grid can still take every qualifying
    # 4D cell (9 per rule, 4 conditions each); partial centres reach the same
    # coverage with two 2-condition centres, so the gain here is description
    # cost, not coverage. Coverage gains appear once small cells fail the
    # size floor or the certificate (next test) or a budget binds (phase 3).
    assert part[4].coverage > 0.6
    assert part[4].coverage >= full[4].centers.coverage - 1e-12
    assert part[4].conditions < 4 * full[4].centers.n_centers / 2
    assert len(part[4].centres) < full[4].centers.n_centers
    assert set(part[4].schema) >= {0, 1, 2, 3}
    subs = {c.sub_schema for c in part[4].centres}
    # the second rule overlaps the first on rows satisfying both; the disjoint
    # packing therefore takes it as 3-axis pieces, never as the overlapping (2, 3) cell
    assert (0, 1) in subs and (2, 3) not in subs
    # at every d the partial search is at least the full search on the full search's schema
    for d in (1, 2, 3, 4):
        assert part[d].coverage >= full[d].centers.coverage - 1e-12
    # a schema that fixes all axes scores its standard coverage: d=2 winners coincide
    assert part[2].coverage == pytest.approx(full[2].centers.coverage)


def test_search_partial_family_certificate_uses_same_family() -> None:
    X, z = _misaligned(n=3000)
    spec = CenterSpec(tau=0.7, min_samples=10, rule="certified", multiplicity="family", alpha=0.05)
    part = search_partial(X, z, spec, max_d=4)
    full = discover_branches(X, z, max_d=4, center_spec=spec, n_permutations_centers=0, cv_splits=2, cv_repeats=1)
    for d in (1, 2, 3, 4):
        assert part[d].coverage >= full[d].centers.coverage - 1e-12
    # under the certificate the 4D cells (~30 rows each) cannot certify; the
    # 2-axis centre of the same schema can: 0.62 against 0.10 on this data
    assert part[4].coverage > full[4].centers.coverage + 0.3


def test_search_partial_titanic_runs_and_is_consistent() -> None:
    X, z = _titanic()
    spec = CenterSpec(tau=0.7, min_samples=10)
    part = search_partial(X, z, spec, max_d=3)
    full = discover_branches(X, z, max_d=3, center_spec=spec, n_permutations_centers=0, cv_splits=2, cv_repeats=1)
    for d in (1, 2, 3):
        assert 0.0 <= part[d].coverage <= 1.0
        assert part[d].coverage >= full[d].centers.coverage - 1e-12
        assert part[d].purity >= spec.tau - 1e-12 or not part[d].centres


# --------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------
def test_crossvalidate_partial_train_only_encoding() -> None:
    X, z = _misaligned(n=2000)
    spec = CenterSpec(tau=0.7, min_samples=10)
    cv = crossvalidate_partial(X, z, spec, max_d=4, n_splits=3, n_repeats=1)
    assert set(cv) == {1, 2, 3, 4}
    for d, r in cv.items():
        assert r.coverage.shape == (3,)
        assert np.all((r.coverage >= 0) & (r.coverage <= 1))
    assert cv[4].mean_coverage > 0.55
    assert np.nanmean(cv[4].purity) > 0.7
