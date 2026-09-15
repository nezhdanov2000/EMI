"""
Redundant centres (`vsf.redundancy`, Project_Master_Document.md Section 4.11).

What is pinned here:
  * the catalogue holds exactly the centres the landscape counts, schema by
    schema, under both rules and both directions;
  * the stored similarity graph equals a brute-force computation of every
    pair (no pair above the floor is skipped by the size band);
  * the two guarantees of the leader grouping, checked by brute force at
    several thresholds, and threshold 1 reproducing the distinct row sets;
  * a centre's description selects exactly its rows, including a schema the
    grid-capacity rule coarsened;
  * a planted near-duplicate characteristic is grouped at 0.8 and split at 0.99;
  * the per-member numbers of `group_detail` are internally consistent;
  * the two endpoints.
"""

from __future__ import annotations

import itertools
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from vsf import redundancy as red
from vsf.avr import compute_landscape
from vsf.centers import CenterSpec, clopper_pearson_lower
from vsf.redundancy import PAIR_FLOOR, collect_centers
from vsf.server import _build_server

_TITANIC = Path(__file__).resolve().parents[1] / "benchmark_data" / "titanic.csv"


def _df(seed: int = 0, n: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.choice(["x", "y", "z"], size=n)
    b = rng.choice(["0", "1"], size=n)
    c = rng.choice(["p", "q", "r", "s"], size=n)
    d = rng.choice(["m", "n"], size=n)
    e = rng.choice(["u", "v", "w"], size=n)
    pos = ((a == "z") & (b == "1")) | ((c == "s") & (e == "w")) | (rng.random(n) < 0.04)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "target": np.where(pos, "yes", "no")})


def _xz(df: pd.DataFrame, target: str = "target", value: str = "yes") -> Tuple[np.ndarray, np.ndarray, list]:
    X = df.drop(columns=[target])
    return X.values, (df[target].astype(str) == value).astype(int).values, list(X.columns)


def _titanic() -> Tuple[np.ndarray, np.ndarray, list]:
    if not _TITANIC.exists():  # pragma: no cover - the benchmark file ships with the repository
        pytest.skip("benchmark_data/titanic.csv not present")
    return _xz(pd.read_csv(_TITANIC), "survived", "died")


def _rows(cat: red.CenterCatalog) -> np.ndarray:
    """(S, N) boolean row membership of every distinct set."""
    return np.unpackbits(cat.set_bits, axis=1, count=cat.n_samples).astype(bool)


def _exact_similarity(cat: red.CenterCatalog) -> np.ndarray:
    R = _rows(cat).astype(np.int64)
    inter = R @ R.T
    big = np.maximum(cat.set_n[:, None], cat.set_n[None, :])
    return inter / big


# ---------------------------------------------------------------------------
# The catalogue is the landscape's centres
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule, tau, direction", [
    ("purity", 0.85, "presence"),
    ("certified", 0.8, "presence"),
    ("purity", 0.9, "absence"),
])
def test_catalogue_holds_exactly_the_centres_the_landscape_counts(rule, tau, direction):
    X, z, names = _xz(_df(1))
    spec = CenterSpec(tau=tau, rule=rule)
    cat = collect_centers(X, z, names, positive_class=1, center_spec=spec, direction=direction)
    ref = compute_landscape(X, z, names, positive_class=1, center_spec=spec, direction=direction)
    L = cat.landscape
    assert L.features == ref.features
    np.testing.assert_array_equal(L.n_centers, ref.n_centers)
    np.testing.assert_array_equal(L.k_sel, ref.k_sel)
    np.testing.assert_array_equal(L.n_sel, ref.n_sel)
    # min_rows = 1: every centre of every schema is in the catalogue
    per_schema = np.bincount(cat.center_schema, minlength=len(cat.schema_features))
    np.testing.assert_array_equal(per_schema, ref.n_centers[cat.schema_landscape_index])
    assert int(ref.n_centers.sum()) == cat.n_centers
    # and each set's (n, k) is the cell's own count of the searched indicator
    R = _rows(cat)
    np.testing.assert_array_equal(R.sum(axis=1), cat.set_n)
    np.testing.assert_array_equal((R & cat.z.astype(bool)[None, :]).sum(axis=1), cat.set_k)


def test_min_rows_drops_only_small_centres_and_keeps_the_landscape():
    X, z, names = _titanic()
    spec = CenterSpec(tau=0.7)
    full = collect_centers(X, z, names, positive_class=1, center_spec=spec, min_rows=1)
    cut = collect_centers(X, z, names, positive_class=1, center_spec=spec, min_rows=10)
    assert cut.set_n.min() >= 10
    assert cut.n_centers == int(np.count_nonzero(full.set_n[full.center_set] >= 10))
    np.testing.assert_array_equal(cut.landscape.n_centers, full.landscape.n_centers)


def test_per_level_lower_bound_equals_clopper_pearson():
    rng = np.random.default_rng(3)
    n = rng.integers(1, 400, size=300)
    k = rng.integers(0, n + 1)
    alpha = rng.choice([0.05, 0.05 / 7, 0.05 / 40, 1e-4], size=300)
    got = red._clopper_pearson_lower_per_level(k, n, alpha)
    ref = np.array([float(clopper_pearson_lower(int(kk), int(nn), float(aa))) for kk, nn, aa in zip(k, n, alpha)])
    np.testing.assert_array_equal(got, ref)


# ---------------------------------------------------------------------------
# Similarity graph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("min_rows", [1, 10])
def test_pair_graph_equals_brute_force(min_rows, monkeypatch):
    # Small blocks and chunks force the blocked path through many band edges.
    monkeypatch.setattr(red, "_BLOCK", 37)
    monkeypatch.setattr(red, "_CHUNK", 53)
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=min_rows)
    S = _exact_similarity(cat)
    np.fill_diagonal(S, 0.0)
    expected = {(i, j) for i, j in zip(*np.nonzero(S >= PAIR_FLOOR)) if i < j}
    got = set()
    for u in range(cat.n_sets):
        nb, inter, sim = cat.neighbours(u)
        for v, w, s in zip(nb.tolist(), inter.tolist(), sim.tolist()):
            assert s == pytest.approx(S[u, v], abs=1e-15)
            assert w == int(round(S[u, v] * max(cat.set_n[u], cat.set_n[v])))
            if u < v:
                got.add((u, v))
    assert got == expected
    assert cat.n_pairs == len(expected)
    assert cat.n_sets == len({r.tobytes() for r in cat.set_bits})  # sets are distinct


def test_distinct_set_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(red, "MAX_DISTINCT_SETS", 5)
    X, z, names = _titanic()
    with pytest.raises(ValueError, match="distinct centre row sets"):
        collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7))


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t", [0.5, 0.7, 0.8, 0.9, 0.97, 1.0])
def test_leader_grouping_guarantees(t):
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=1)
    S = _exact_similarity(cat)
    g = cat.group(t)
    leaders = g.leaders
    assert np.all(g.leader_of_set[leaders] == leaders)
    # (1) every member lies at least t inside its representative and vice versa
    members = np.nonzero(g.leader_of_set != np.arange(cat.n_sets))[0]
    lead = g.leader_of_set[members]
    assert np.all(S[members, lead] >= t - 1e-12)
    np.testing.assert_allclose(g.sim_to_leader[members], S[members, lead], rtol=0, atol=1e-15)
    # (2) no two representatives are t-duplicates
    SL = S[np.ix_(leaders, leaders)].copy()
    np.fill_diagonal(SL, 0.0)
    assert np.all(SL < t - 1e-12)
    # the representative is the best-ranked set of its group
    for u in leaders.tolist():
        grp = np.nonzero(g.leader_of_set == u)[0]
        assert cat.set_rank[u] == cat.set_rank[grp].min()
    # a member joins its MOST similar eligible representative
    for m in members.tolist():
        eligible = [l for l in leaders.tolist() if cat.set_rank[l] < cat.set_rank[m] and S[m, l] >= t - 1e-12]
        assert S[m, g.leader_of_set[m]] == pytest.approx(max(S[m, l] for l in eligible), abs=1e-15)


def test_threshold_one_groups_exactly_the_identical_row_sets():
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7))
    g = cat.group(1.0)
    assert g.leaders.shape[0] == cat.n_sets
    page = cat.groups_page(1.0, limit=10_000)
    assert page["n_groups"] == cat.n_sets
    assert sum(gr["n_members"] for gr in page["groups"]) == cat.n_centers
    assert all(gr["n_distinct"] == 1 for gr in page["groups"])


def test_representative_rank_rule_prefers_fewer_characteristics_then_the_lower_bound():
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7))
    order = cat.rank_order
    d = cat.set_d[order]
    lo = cat.set_lower[order]
    assert np.all(np.diff(d) >= 0)
    same_d = np.diff(d) == 0
    assert np.all(np.diff(lo)[same_d] <= 1e-15)
    # the best centre of a set has the set's minimal d and, among those, the largest level
    for u in range(cat.n_sets):
        cs = np.nonzero(cat.center_set == u)[0]
        dd = cat.schema_d[cat.center_schema[cs]]
        b = int(cat.set_best_center[u])
        assert cat.schema_d[cat.center_schema[b]] == dd.min()
        aa = cat.schema_alpha[cat.center_schema[cs[dd == dd.min()]]]
        assert cat.schema_alpha[cat.center_schema[b]] == aa.max()


def test_grouping_is_deterministic():
    X, z, names = _titanic()
    a = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=5)
    b = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=5)
    assert json.dumps(a.groups_page(0.8, limit=50)) == json.dumps(b.groups_page(0.8, limit=50))


def test_threshold_outside_the_stored_range_is_rejected():
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7))
    for bad in (0.49, 1.01, -1.0):
        with pytest.raises(ValueError, match="threshold"):
            cat.group(bad)
    with pytest.raises(ValueError):
        cat.groups_page(0.8, sort="size")


def test_planted_near_duplicate_characteristic():
    """
    `b2` copies `b` except on 3 % of rows: the centres built on `b2` must
    join the groups of their `b` twins at 0.8 and stay apart at 0.99.
    """
    rng = np.random.default_rng(11)
    n = 2000
    a = rng.choice(["x", "y", "z"], size=n)
    b = rng.choice(["0", "1", "2"], size=n)
    flip = rng.random(n) < 0.03
    b2 = np.where(flip, rng.choice(["0", "1", "2"], size=n), b)
    noise = rng.choice(["p", "q"], size=n)
    pos = ((b == "1") & (rng.random(n) < 0.97)) | (rng.random(n) < 0.05)
    df = pd.DataFrame({"a": a, "b": b, "b2": b2, "noise": noise, "target": np.where(pos, "yes", "no")})
    X, z, names = _xz(df)
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.85), max_d=1)
    single = {cat.schema_features[q][0]: q for q in range(len(cat.schema_features))}
    assert set(single) == {1, 2}  # b = 1 and b2 = 1 are the only 1D centres

    def set_of(feature: int) -> int:
        c = np.nonzero(cat.center_schema == single[feature])[0]
        assert c.shape[0] == 1
        return int(cat.center_set[c[0]])

    ub, ub2 = set_of(1), set_of(2)
    g8, g99 = cat.group(0.8), cat.group(0.99)
    assert g8.leader_of_set[ub] == g8.leader_of_set[ub2]
    assert g99.leader_of_set[ub] != g99.leader_of_set[ub2]
    # the cleaner copy (`b`, higher lower bound) represents the pair
    assert g8.leader_of_set[ub2] == ub


# ---------------------------------------------------------------------------
# Descriptions and member numbers
# ---------------------------------------------------------------------------

def _selects(cat: red.CenterCatalog, names: list, conditions: list) -> np.ndarray:
    m = np.ones(cat.n_samples, dtype=bool)
    for cond in conditions:
        j = names.index(cond["feature"])
        m &= np.isin(cat.category_labels(j), cond["values"])
    return m


def test_description_selects_exactly_the_centre_rows_including_coarsened_schemas():
    # N = 300 -> capacity 30 occupied cells; two 12-level columns exceed it
    # in 2D, so the search partition merges levels and descriptions list
    # several values.
    rng = np.random.default_rng(5)
    n = 300
    u = rng.integers(0, 12, size=n)
    v = rng.choice([f"L{i}" for i in range(12)], size=n)
    w = rng.choice(["a", "b"], size=n)
    pos = (u < 3) | (rng.random(n) < 0.05)
    df = pd.DataFrame({"u": u, "v": v, "w": w, "target": np.where(pos, "yes", "no")})
    X, z, names = _xz(df)
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.8), max_d=2)
    R = _rows(cat)
    merged_seen = False
    for c in range(cat.n_centers):
        cond = cat.describe(c)
        merged_seen |= any(e["merged"] for e in cond)
        np.testing.assert_array_equal(_selects(cat, names, cond), R[cat.center_set[c]])
    assert merged_seen
    # ordered (numeric) column: values listed in numeric order
    for c in range(cat.n_centers):
        for e in cat.describe(c):
            if e["feature"] == "u" and len(e["values"]) > 1:
                assert [int(x) for x in e["values"]] == sorted(int(x) for x in e["values"])


def test_descriptions_with_missing_values_and_colliding_labels():
    """
    A numeric column with gaps, a nominal column holding both NaN and the
    literal string "missing", and a mixed column holding 2.0 beside "2":
    every category keeps one unique label and every description selects
    exactly its cell.
    """
    rng = np.random.default_rng(8)
    n = 400
    age = rng.choice([1.0, 2.0, 3.5, np.nan], size=n)
    tag = rng.choice(np.array(["a", "missing", None, np.nan], dtype=object), size=n)
    mix = rng.choice(np.array([2.0, "2", "x"], dtype=object), size=n)
    pos = (age == 1.0) | (rng.random(n) < 0.1)
    df = pd.DataFrame({"age": age, "tag": tag, "mix": mix, "target": np.where(pos, "yes", "no")})
    X, z, names = _xz(df)
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.8))
    assert cat.factory.bin_counts == [4, 3, 3]  # NaN is one level; None and NaN are one level
    for j in range(3):
        labels = cat._column_labels(j)
        assert len(set(labels)) == len(labels)
    assert "missing" in cat._column_labels(0)
    assert sorted(cat._column_labels(1)) == ["'missing'", "a", "missing"]
    assert sorted(cat._column_labels(2)) == ["'2'", "2.0", "x"]
    R = _rows(cat)
    assert cat.n_centers > 0
    for c in range(cat.n_centers):
        cond = cat.describe(c)
        np.testing.assert_array_equal(_selects(cat, names, cond), R[cat.center_set[c]])
        assert all(len(e["values"]) == 1 for e in cond)  # no capacity merging at N = 400 here


def test_catalogue_is_safe_to_share_between_threads():
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=5)
    errors: list = []

    def work(seed: int) -> None:
        rng = np.random.default_rng(seed)
        try:
            for _ in range(20):
                t = float(rng.uniform(0.5, 1.0))
                cat.group(t)
                cat.describe(int(rng.integers(cat.n_centers)))
        except Exception as exc:  # pragma: no cover - the failure being tested for
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []


def test_titanic_descriptions_select_their_rows():
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=10)
    R = _rows(cat)
    for c in range(0, cat.n_centers, 7):
        np.testing.assert_array_equal(_selects(cat, names, cat.describe(c)), R[cat.center_set[c]])


def test_group_detail_numbers_are_consistent():
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=10)
    R = _rows(cat)
    zb = cat.z.astype(bool)
    page = cat.groups_page(0.8, sort="members", limit=5)
    assert page["histogram"]["below_floor"] + page["histogram"]["identical"] + sum(page["histogram"]["counts"]) == cat.n_centers
    for gr in page["groups"]:
        det = cat.group_detail(0.8, gr["group"], limit=500)
        rep = det["representative"]
        assert det["total"] == gr["n_members"] - 1
        rr = R[rep["set"]]
        assert rep["n"] == int(rr.sum()) and rep["k"] == int((rr & zb).sum())
        assert rep["purity_lower"] <= rep["purity"]
        union = np.zeros_like(rr)
        for m in det["members"]:
            mr = R[m["set"]]
            union |= mr
            inter = int((mr & rr).sum())
            assert m["intersection"]["n"] == inter
            assert m["intersection"]["k"] == int((mr & rr & zb).sum())
            assert m["similarity"] == pytest.approx(inter / max(m["n"], rep["n"]))
            assert m["similarity"] >= 0.8 - 1e-12
            assert m["share_in_representative"] == pytest.approx(inter / m["n"])
            assert m["share_of_representative"] == pytest.approx(inter / rep["n"])
            assert m["only_here"]["n"] == m["n"] - inter
            assert m["only_in_representative"]["n"] == rep["n"] - inter
            s0 = min(m["n"], rep["n"]) / cat.n_samples
            assert m["chance_similarity"] == pytest.approx(s0)
            assert m["similarity_above_chance"] == pytest.approx((m["similarity"] - s0) / (1 - s0))
            assert m["similarity_above_chance"] <= m["similarity"] + 1e-12
            assert m["identical"] == (m["set"] == rep["set"])
        union |= rr
        assert det["union"]["n"] == int(union.sum())
        assert det["union"]["k"] == int((union & zb).sum())


def test_group_filters_by_dimensionality_and_landscape_cell():
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=10)
    all_ = cat.groups_page(0.8, limit=10_000)
    only2 = cat.groups_page(0.8, d=2, limit=10_000)
    assert 0 < only2["total"] <= all_["total"]
    assert all(gr["schemas_by_d"][1] > 0 for gr in only2["groups"])
    cell = only2["groups"][0]["landscape_cells"][0]
    got = cat.groups_page(0.8, cell=(cell["d"], cell["ix"], cell["iy"]), limit=10_000)
    assert got["total"] >= 1
    for gr in got["groups"]:
        assert any(c["d"] == cell["d"] and c["ix"] == cell["ix"] and c["iy"] == cell["iy"] for c in gr["landscape_cells"])
    with pytest.raises(ValueError):
        cat.group_detail(0.8, -1)


# ---------------------------------------------------------------------------
# Branch view
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("min_rows", [1, 10])
def test_branch_view_lists_every_direct_alternative_of_every_branch_centre(min_rows):
    X, z, names = _titanic()
    spec = CenterSpec(tau=0.7)
    cat = collect_centers(X, z, names, positive_class=1, center_spec=spec, min_rows=min_rows)
    S = _exact_similarity(cat)
    R = _rows(cat)
    ref = compute_landscape(X, z, names, positive_class=1, center_spec=spec)
    for feats in ([2, 4], [4, 2], [5], [0, 1, 3]):
        combo = tuple(sorted(feats))
        li = ref.features.index(combo)
        v = cat.branch_view(feats, 0.8, limit=10_000)
        assert v["features"] == list(combo)
        assert v["n_centers"] == int(ref.n_centers[li]) == len(v["anchors"])
        assert v["coverage"] == pytest.approx(ref.k_sel[li] / ref.n_positive)
        assert [a["n"] for a in v["anchors"]] == sorted((a["n"] for a in v["anchors"]), reverse=True)
        for a in v["anchors"]:
            if a["n"] < min_rows:
                assert a["compared"] is False and a["alternatives"] == []
                continue
            assert a["compared"] is True
            u = a["set"]
            np.testing.assert_array_equal(_selects(cat, names, a["conditions"]), R[u])
            expected = {int(c) for c in np.nonzero((S[cat.center_set, u] >= 0.8 - 1e-12))[0]}
            q = cat.schema_of_features[combo]
            own = int(np.nonzero((cat.center_schema == q) & (cat.center_cell == a["cell"]))[0][0])
            assert cat.center_set[own] == u
            expected.discard(own)
            got = {m["center"] for m in a["alternatives"]}
            assert got == expected and a["total"] == len(expected)
            for m in a["alternatives"]:
                assert m["similarity"] == pytest.approx(S[m["set"], u])
                assert m["similarity"] >= 0.8 - 1e-12
            sims = [m["similarity"] for m in a["alternatives"]]
            assert sims == sorted(sims, reverse=True)
            if a["alternatives"]:
                ds = [m["d"] for m in a["alternatives"]]
                assert a["simplest"]["d"] == min(ds)
    with pytest.raises(ValueError):
        cat.branch_view([2, 2], 0.8)
    with pytest.raises(ValueError):
        cat.branch_view([99], 0.8)
    with pytest.raises(ValueError):
        cat.branch_view([2], 0.8, anchor=10_000)


def test_histogram_scopes_count_the_right_centres():
    """
    Three scopes over the same axis. "all": every centre's nearest other
    centre. "branch": the same value for the centres of one schema only.
    "center": ONE centre against every other centre, whose bars at or above
    the threshold must be exactly that centre's listed alternatives.
    """
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=10)
    S = _exact_similarity(cat)
    per_set = np.where(cat.set_multiplicity > 1, 1.0, cat.set_nn_sim)

    def parts(h):
        return h["identical"] + h["below_floor"] + sum(h["counts"])

    def at_or_above(h, t):
        return h["identical"] + sum(c for c, e in zip(h["counts"], h["edges"]) if e >= t - 1e-12)

    everything = cat.nearest_neighbour_histogram()
    assert everything["scope"] == "all"
    assert everything["n_centers"] == cat.n_centers == parts(everything)

    combo = (2, 4)
    q = cat.schema_of_features[combo]
    branch_centers = np.nonzero(cat.center_schema == q)[0]
    view = cat.branch_view(list(combo), 0.8, limit=10_000)
    hb = view["histogram"]
    assert hb["scope"] == "branch"
    assert hb["n_centers"] == int(branch_centers.shape[0]) == parts(hb)
    assert hb["n_centers"] < everything["n_centers"]
    # the branch bars are the branch's slice of the global ones
    assert at_or_above(hb, 0.8) == int(np.count_nonzero(per_set[cat.center_set[branch_centers]] >= 0.8 - 1e-12))
    assert hb["identical"] == int(np.count_nonzero(per_set[cat.center_set[branch_centers]] >= 1.0 - 1e-12))

    for a in view["anchors"]:
        if not a["compared"]:
            assert "histogram" not in a
            continue
        hc = a["histogram"]
        assert hc["scope"] == "center"
        assert hc["n_centers"] == cat.n_centers - 1 == parts(hc)
        # the threshold on this card's own axis reproduces its alternatives
        assert at_or_above(hc, view["threshold"]) == a["total"]
        assert hc["identical"] == a["n_identical"]
        own = int(np.nonzero((cat.center_schema == q) & (cat.center_cell == a["cell"]))[0][0])
        sims = np.delete(S[cat.center_set, a["set"]], own)
        assert hc["below_floor"] == int(np.count_nonzero(sims < PAIR_FLOOR - 1e-12))

    with pytest.raises(ValueError):
        cat.center_similarity_histogram(cat.n_centers)
    with pytest.raises(ValueError):
        cat.center_similarity_histogram(-1)


def test_kinship_is_two_independent_facts_and_the_filter_partitions_the_alternatives():
    """
    `lineage` compares the two schemas' COLUMN sets, `relation` the two ROW
    sets, and the second is measured rather than inferred from the first.
    The "related"/"unrelated" filter must partition the unfiltered list
    exactly, and its counts must not depend on which side is shown.
    """
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=10)
    R = _rows(cat)
    combo = (2, 4)
    full = cat.branch_view(list(combo), 0.8, limit=10_000)
    rel = cat.branch_view(list(combo), 0.8, limit=10_000, kinship="related")
    unrel = cat.branch_view(list(combo), 0.8, limit=10_000, kinship="unrelated")
    assert full["kinship"] == "all" and rel["kinship"] == "related"
    seen_lineage, seen_relation, inconsistent = set(), set(), 0
    for a, ar, au in zip(full["anchors"], rel["anchors"], unrel["anchors"]):
        assert a["cell"] == ar["cell"] == au["cell"]
        if not a["compared"]:
            continue
        counts = a["kinship_counts"]
        # the split is over the unfiltered list and is the same in all three
        assert counts == ar["kinship_counts"] == au["kinship_counts"]
        assert counts["related"] + counts["unrelated"] == a["total"]
        assert ar["total"] == counts["related"] and au["total"] == counts["unrelated"]
        ids = {m["center"] for m in a["alternatives"]}
        assert {m["center"] for m in ar["alternatives"]} | {m["center"] for m in au["alternatives"]} == ids
        assert not ({m["center"] for m in ar["alternatives"]} & {m["center"] for m in au["alternatives"]})
        s_b = set(combo)
        for m in a["alternatives"]:
            k = m["kinship"]
            s_a = set(m["schema_features"])
            expect = ("same_schema" if s_a == s_b else "child" if s_b < s_a
                      else "parent" if s_a < s_b else "unrelated")
            assert k["lineage"] == expect
            # the row relation is the measured one, not the inferred one
            ra, rb = R[m["set"]], R[a["set"]]
            inter = int((ra & rb).sum())
            expect_rel = ("identical" if inter == ra.sum() == rb.sum()
                          else "inside" if inter == ra.sum()
                          else "contains" if inter == rb.sum() else "crossing")
            assert k["relation"] == expect_rel
            nested = k["lineage"] in ("child", "parent")
            assert (m in ar["alternatives"]) is nested
            ok = (k["relation"] in ("identical", "inside") if k["lineage"] == "child"
                  else k["relation"] in ("identical", "contains") if k["lineage"] == "parent" else True)
            assert k["consistent"] is ok
            inconsistent += 0 if ok else 1
            seen_lineage.add(k["lineage"])
            seen_relation.add(k["relation"])
    # the branch is compared against the whole family, so both sides occur
    assert {"child", "unrelated"} <= seen_lineage
    assert {"identical", "inside"} <= seen_relation
    with pytest.raises(ValueError):
        cat.branch_view(list(combo), 0.8, kinship="nested")


def test_the_histograms_are_drawn_on_the_population_the_filter_leaves():
    """
    A picture beside a filtered list must count the same centres the list
    does. Under a kinship filter both the per-centre strip and the
    nearest-neighbour histogram drop the centres the filter excludes, and
    the fast nearest-neighbour walk must agree with the dense computation.
    """
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=10)
    combo = (2, 4)
    for kinship in ("all", "related", "unrelated"):
        v = cat.branch_view(list(combo), 0.8, limit=10_000, kinship=kinship)
        for a in v["anchors"]:
            if not a["compared"]:
                continue
            h = a["histogram"]
            # the strip counts exactly the centres this card may be compared with
            expected = int(cat._comparison_mask(
                int(np.nonzero((cat.center_schema == cat.schema_of_features[combo])
                               & (cat.center_cell == a["cell"]))[0][0]), kinship).sum())
            assert h["n_centers"] == expected
            assert h["identical"] + h["below_floor"] + sum(h["counts"]) == expected
            # and its bars at or above the threshold are this card's own list
            at_or_above = h["identical"] + sum(
                c for c, e in zip(h["counts"], h["edges"]) if e >= v["threshold"] - 1e-12)
            assert at_or_above == a["total"]
        # the branch histogram is the maximum of each card's strip
        maxima = []
        for a in v["anchors"]:
            if not a["compared"]:
                continue
            c = int(np.nonzero((cat.center_schema == cat.schema_of_features[combo])
                               & (cat.center_cell == a["cell"]))[0][0])
            vals = cat.center_similarity_values(c, kinship)
            maxima.append(float(vals.max()) if vals.size else 0.0)
        assert v["histogram"] == cat._similarity_histogram(np.asarray(maxima), "branch")

        # the pooled picture IS the elementwise sum of the cards' strips, and
        # its bars at or above the threshold are the branch's whole list
        strips = [a["histogram"] for a in v["anchors"] if a["compared"]]
        pooled = v["histogram_pairs"]
        assert pooled["scope"] == "branch_pairs"
        assert pooled["counts"] == [sum(s["counts"][i] for s in strips) for i in range(len(pooled["counts"]))]
        assert pooled["below_floor"] == sum(s["below_floor"] for s in strips)
        assert pooled["identical"] == sum(s["identical"] for s in strips)
        assert pooled["n_centers"] == sum(s["n_centers"] for s in strips)
        assert pooled["identical"] + sum(
            c for c, e in zip(pooled["counts"], pooled["edges"]) if e >= v["threshold"] - 1e-12
        ) == sum(a["total"] for a in v["anchors"] if a["compared"])
        # the solid part of that picture is the sum of the strips ON SCREEN:
        # every compared centre with no filter, and otherwise only the centres
        # whose cards survive it
        listed_strips = [
            a["histogram"] for a in v["anchors"]
            if a["compared"] and (kinship == "all" or a["total"] > 0)
        ]
        shown = v["histogram_pairs_listed"]
        assert v["n_centers_listed"] == len(listed_strips)
        assert shown["counts"] == [sum(s["counts"][i] for s in listed_strips) for i in range(len(shown["counts"]))]
        assert shown["below_floor"] == sum(s["below_floor"] for s in listed_strips)
        assert shown["identical"] == sum(s["identical"] for s in listed_strips)
        assert shown["n_centers"] == sum(s["n_centers"] for s in listed_strips)
        # solid never exceeds the whole, and equals it when nothing is hidden
        assert all(a >= b for a, b in zip(pooled["counts"], shown["counts"]))
        assert shown["n_centers"] <= pooled["n_centers"]
        if len(listed_strips) == len(strips):
            assert shown == pooled
        # paging one card must not shrink any branch-level picture
        first = next(a for a in v["anchors"] if a["compared"])
        page = cat.branch_view(list(combo), 0.8, limit=3, anchor=first["cell"], kinship=kinship)
        assert page["histogram_pairs"] == pooled and page["histogram"] == v["histogram"]
        assert page["histogram_pairs_listed"] == shown
        assert page["n_centers_listed"] == v["n_centers_listed"]

    # the early-exit walk used for the whole family equals the dense route
    for kinship in ("related", "unrelated"):
        dense = np.asarray([
            (lambda w: float(w.max()) if w.size else 0.0)(cat.center_similarity_values(c, kinship))
            for c in range(cat.n_centers)
        ])
        fast = cat.nearest_neighbour_histogram(None, "all", kinship)
        assert fast == cat._similarity_histogram(dense, "all")
        assert fast != cat.nearest_neighbour_histogram(None, "all", "all")


def test_level_coarsening_breaks_the_child_parent_row_nesting():
    """
    The reason `relation` may not be inferred from `lineage`: a schema's
    partition refines its sub-schema's ONLY while no shared column was
    coarsened to fit the grid capacity (Section 4.3). On titanic this fails
    for a measurable share of the nested schema pairs, and the resulting
    pairs are reported as inconsistent rather than as sub-cells.
    """
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.9), min_rows=10)
    F = cat.factory
    broken = []
    for parent in itertools.combinations(range(F.n_features), 3):
        for extra in range(F.n_features):
            if extra in parent:
                continue
            child = tuple(sorted(parent + (extra,)))
            cc, _ = F.codes(child)
            pc, _ = F.codes(parent)
            order = np.argsort(cc, kind="stable")
            bounds = np.flatnonzero(np.diff(cc[order])) + 1
            if any(np.unique(pc[g]).size > 1 for g in np.split(order, bounds)):
                broken.append((parent, child))
    assert broken, "expected the capacity rule to coarsen at least one shared column"
    # and an inconsistent pair is classified as such, never as a sub-cell
    parent, child = broken[0]
    q_child = cat.schema_of_features.get(child)
    if q_child is not None:
        for c in np.nonzero(cat.center_schema == q_child)[0].tolist():
            u = int(cat.center_set[c])
            for ref in range(cat.n_sets):
                k = cat._kinship(c, parent, int(cat.set_n[u]), int(cat.set_n[ref]), cat.intersection(u, ref))
                assert k["lineage"] == "child"
                assert k["consistent"] == (k["relation"] in ("identical", "inside"))


def test_branch_view_anchor_pages_one_centre():
    X, z, names = _titanic()
    cat = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.7), min_rows=10)
    full = cat.branch_view([2, 4], 0.6, limit=10_000)
    a = max((a for a in full["anchors"] if a["compared"]), key=lambda a: a["total"])
    pages = []
    for off in range(0, a["total"], 3):
        p = cat.branch_view([2, 4], 0.6, limit=3, offset=off, anchor=a["cell"])
        assert len(p["anchors"]) == 1 and p["anchors"][0]["cell"] == a["cell"]
        pages += [m["center"] for m in p["anchors"][0]["alternatives"]]
    assert pages == [m["center"] for m in a["alternatives"]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class _Server:
    def __init__(self, df):
        self.httpd = _build_server(df, host="127.0.0.1", port=0, translations=None, prefetch=False)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]

    def post(self, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def close(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()


def test_centre_group_endpoints():
    df = _df(6)
    srv = _Server(df)
    base = {"target": "target", "criterion": "yes", "tau": 0.85}
    try:
        status, page = srv.post("/api/centers/groups", dict(base, threshold=0.8, min_rows=3, limit=5))
        assert status == 200, page
        assert page["threshold"] == 0.8 and page["min_rows"] == 3 and page["tau"] == 0.85
        assert len(page["groups"]) <= 5 and page["total"] == page["n_groups"]
        X, z, names = _xz(df)
        ref = collect_centers(X, z, names, positive_class=1, center_spec=CenterSpec(tau=0.85), min_rows=3)
        assert page["groups"] == ref.groups_page(0.8, limit=5)["groups"]
        status, page9 = srv.post("/api/centers/groups", dict(base, threshold=0.95, min_rows=3))
        assert status == 200 and len(srv.httpd.centers_cache) == 1  # the threshold only regroups
        gid = page["groups"][0]["group"]
        status, det = srv.post("/api/centers/group", dict(base, threshold=0.8, min_rows=3, group=gid))
        assert status == 200, det
        assert det["group"] == gid and det["total"] == page["groups"][0]["n_members"] - 1
        status, filt = srv.post("/api/centers/groups", dict(base, threshold=0.8, min_rows=3, filter_d=2,
                                                            cell={"d": None, "ix": 0, "iy": 0}))
        assert status == 200 and filt["filter_d"] == 2 and filt["cell"] == {"d": None, "ix": 0, "iy": 0}
        assert srv.post("/api/centers/groups", dict(base, threshold=0.3))[0] == 400
        assert srv.post("/api/centers/groups", dict(base, min_rows=0))[0] == 400
        assert srv.post("/api/centers/groups", dict(base, filter_d=7))[0] == 400
        assert srv.post("/api/centers/groups", dict(base, cell={"ix": 1}))[0] == 400
        assert srv.post("/api/centers/groups", dict(base, cell={"d": None, "ix": 10, "iy": 0}))[0] == 400
        assert srv.post("/api/centers/groups", dict(base, filter_d=[2]))[0] == 400
        assert srv.post("/api/centers/group", dict(base))[0] == 400
        assert srv.post("/api/centers/group", dict(base, group=10_000_000))[0] == 400
        assert srv.post("/api/centers/groups", dict(base, tau=0.01))[0] == 400  # below base rate
        status, br = srv.post("/api/centers/branch", dict(base, threshold=0.8, min_rows=3, features=[0, 1], limit=50))
        assert status == 200, br
        assert br == dict(ref.branch_view([0, 1], 0.8), **{k: br[k] for k in (
            "target", "criterion", "tau", "rule", "alpha", "min_samples", "also")}, direction="presence")
        assert srv.post("/api/centers/branch", dict(base, features=[]))[0] == 400
        assert srv.post("/api/centers/branch", dict(base, features=["a"]))[0] == 400
        assert srv.post("/api/centers/branch", dict(base, features=[0, 0]))[0] == 400
    finally:
        srv.close()
