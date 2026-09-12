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
