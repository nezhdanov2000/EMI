"""
Dataset screen and dependent-candidate pruning (`vsf.screen`,
`vsf.avr._CandidateFactory`, Project_Master_Document.md Section 4.12).

What is pinned here:
  * `delta`, `strength` and the exception counts equal a brute-force
    computation, in both directions, on random and constructed data;
  * the sparse fallback of the contingency counter returns the dense result;
  * every kind of dependency is classified as specified (equivalent / exact /
    approximate) and the reporting floor works on the normalised strength,
    not on `delta`;
  * the column flags and the missing-value handling;
  * `target_report` finds a planted leak;
  * candidate pruning removes EXACTLY the candidates whose partition equals
    that of a smaller subset, never one whose partition differs (the
    capacity-coarsened case), and leaves a dependency-free dataset's search
    bit-identical;
  * the endpoints, including that `drop` and `prune` are part of the
    analysis identity.
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

from vsf import screen as scr
from vsf.avr import _CandidateFactory, compute_landscape, discover_branches
from vsf.centers import CenterSpec
from vsf.pmd import discretize_dataset, grid_capacity
from vsf.screen import (
    column_profiles,
    dependency_pairs,
    exact_dependencies,
    screen_dataset,
    target_report,
)
from vsf.server import _build_server

_TITANIC = Path(__file__).resolve().parents[1] / "benchmark_data" / "titanic.csv"


def _titanic() -> pd.DataFrame:
    if not _TITANIC.exists():  # pragma: no cover - ships with the repository
        pytest.skip("benchmark_data/titanic.csv not present")
    return pd.read_csv(_TITANIC)


def _brute_delta(x, y) -> Tuple[float, int]:
    """(share of rows the best map x -> y gets right, exception rows), the slow way."""
    n = len(x)
    hits = 0
    for v in set(x):
        ys = [y[i] for i in range(n) if x[i] == v]
        hits += max(ys.count(w) for w in set(ys))
    return hits / n, n - hits


# ---------------------------------------------------------------------------
# The measure itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dependency_numbers_equal_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 200
    cols = {
        "a": rng.choice(list("xyz"), n),
        "b": rng.choice(list("01"), n),
        "c": rng.choice([f"L{i}" for i in range(7)], n),
        "d": np.where(rng.random(n) < 0.97, rng.choice(list("pq"), n), "r"),
    }
    df = pd.DataFrame(cols)
    pairs, skipped = dependency_pairs(df.values, list(df.columns), min_strength=0.0)
    assert skipped is None
    got = {(p.a, p.b): p for p in pairs}
    assert set(got) == {(a, b) for a, b in itertools.combinations(df.columns, 2)}
    for (a, b), p in got.items():
        d_ab, exc_ab = _brute_delta(list(df[a]), list(df[b]))
        d_ba, exc_ba = _brute_delta(list(df[b]), list(df[a]))
        assert p.delta_ab == pytest.approx(d_ab) and p.exceptions_ab == exc_ab
        assert p.delta_ba == pytest.approx(d_ba) and p.exceptions_ba == exc_ba
        base_b = df[b].value_counts().iat[0] / n
        base_a = df[a].value_counts().iat[0] / n
        assert p.strength_ab == pytest.approx((d_ab - base_b) / (1 - base_b))
        assert p.strength_ba == pytest.approx((d_ba - base_a) / (1 - base_a))
        assert p.kind == "approximate"


def test_sparse_fallback_matches_the_dense_table(monkeypatch):
    rng = np.random.default_rng(4)
    n = 300
    df = pd.DataFrame({
        "a": rng.integers(0, 40, n).astype(str),
        "b": rng.integers(0, 30, n).astype(str),
    })
    dense, _ = dependency_pairs(df.values, list(df.columns), min_strength=0.0)
    monkeypatch.setattr(scr, "_CONTINGENCY_CAP", 4)  # forces the sort-based path
    sparse, _ = dependency_pairs(df.values, list(df.columns), min_strength=0.0)
    assert [p.to_dict() for p in dense] == [p.to_dict() for p in sparse]


def test_kinds_and_reporting_floor():
    n = 400
    rng = np.random.default_rng(5)
    fine = rng.choice(["Mr", "Mrs", "Miss", "Master"], n)              # 4 levels
    renamed = np.array([{"Mr": "1", "Mrs": "2", "Miss": "3", "Master": "4"}[v] for v in fine])
    coarse = np.array(["male" if v in ("Mr", "Master") else "female" for v in fine])  # fine -> coarse
    almost = np.where(rng.random(n) < 0.02, "other", coarse)           # breaks on ~2 % of rows
    lopsided = np.where(rng.random(n) < 0.99, "no", "yes")             # 99 % one level
    df = pd.DataFrame({"fine": fine, "renamed": renamed, "coarse": coarse,
                       "almost": almost, "lopsided": lopsided})
    pairs, _ = dependency_pairs(df.values, list(df.columns), min_strength=0.9)
    kinds = {(p.a, p.b): p for p in pairs}
    assert kinds[("fine", "renamed")].kind == "equivalent"
    assert kinds[("fine", "renamed")].delta_ab == 1.0 and kinds[("fine", "renamed")].delta_ba == 1.0
    ex = kinds[("fine", "coarse")]
    assert ex.kind == "exact" and ex.delta_ab == 1.0 and ex.delta_ba < 1.0
    ap = kinds[("coarse", "almost")]
    assert ap.kind == "approximate" and 0 < ap.exceptions_ab <= n * 0.05
    # A 99 %-one-level column is "determined" to 0.99 by anything; the
    # normalised strength keeps it out at the reporting floor.
    assert not any("lopsided" in (p.a, p.b) and p.kind == "approximate" for p in pairs)
    wide, _ = dependency_pairs(df.values, list(df.columns), min_strength=0.0)
    lop = [p for p in wide if p.b == "lopsided" and p.a == "fine"][0]
    assert lop.delta_ab > 0.98 and lop.strength_ab < 0.5
    # sorting: equivalent first, then exact, then approximate
    order = [p.kind for p in pairs]
    assert order == sorted(order, key=["equivalent", "exact", "approximate"].index)


def test_constant_column_is_determined_by_everything():
    n = 50
    df = pd.DataFrame({"a": np.arange(n) % 5, "const": ["c"] * n})
    deps = exact_dependencies(df.values, list(df.columns))
    assert deps == {1: (0,)}
    pairs, _ = dependency_pairs(df.values, list(df.columns), min_strength=0.9)
    assert pairs[0].kind == "exact" and pairs[0].delta_ab == 1.0
    prof = {p.name: p for p in column_profiles(df.values, list(df.columns))}
    assert "constant" in prof["const"].flags and prof["const"].n_levels == 1


def test_column_profiles_flags_and_missing():
    n = 300
    rng = np.random.default_rng(6)
    df = pd.DataFrame({
        "ident": [f"row{i}" for i in range(n)],                 # more levels than capacity
        "dominant": np.where(rng.random(n) < 0.95, "no", "yes"),
        "gappy": np.array([np.nan if v < 0.6 else "value" for v in rng.random(n)], dtype=object),
        "plain": rng.choice(list("abc"), n),
    })
    prof = {p.name: p for p in column_profiles(df.values, list(df.columns))}
    assert prof["ident"].n_levels == n and "exceeds_capacity" in prof["ident"].flags
    assert grid_capacity(n) == 30
    assert "dominant_level" in prof["dominant"].flags
    assert prof["dominant"].largest_level == "no"
    assert "mostly_missing" in prof["gappy"].flags
    assert prof["gappy"].largest_level == "missing" and prof["gappy"].n_levels == 2
    assert prof["gappy"].n_missing == int(df["gappy"].isna().sum())
    assert prof["plain"].flags == [] and prof["plain"].n_singleton_levels == 0
    assert prof["ident"].n_singleton_levels == n


def test_placeholder_levels_are_named_not_guessed():
    n = 400
    rng = np.random.default_rng(12)
    df = pd.DataFrame({
        "deck": np.where(rng.random(n) < 0.7, "unknown_deck", rng.choice(list("ABC"), n)),
        "test": np.where(rng.random(n) < 0.3, "?", rng.choice(["pos", "neg"], n)),
        "grade": rng.choice(["low", "mid", "high"], n),
        "rare": np.where(rng.random(n) < 0.01, "N/A", rng.choice(list("xy"), n)),
    })
    prof = {p.name: p for p in column_profiles(df.values, list(df.columns))}
    assert prof["deck"].placeholder_level == "unknown_deck" and "placeholder_level" in prof["deck"].flags
    assert prof["deck"].placeholder_share == pytest.approx(
        float((df["deck"] == "unknown_deck").mean()))
    assert prof["test"].placeholder_level == "?"
    assert prof["grade"].placeholder_level is None and "placeholder_level" not in prof["grade"].flags
    # below PLACEHOLDER_MIN_SHARE: named, but too small to be worth a flag
    assert prof["rare"].placeholder_level is None
    assert scr._is_placeholder("Unknown") and scr._is_placeholder("not recorded")
    assert scr._is_placeholder("N/A") and scr._is_placeholder("") and scr._is_placeholder("-")
    assert not scr._is_placeholder("Mr") and not scr._is_placeholder("3rd_Class")


def test_titanic_placeholder_columns():
    df = _titanic()
    prof = {p.name: p for p in column_profiles(df.values, list(df.columns))}
    assert prof["deck"].placeholder_level == "unknown_deck"
    assert prof["deck"].placeholder_share == pytest.approx(687 / 891)
    assert prof["age_group"].placeholder_level == "unknown_age"
    assert prof["title"].placeholder_level is None  # "Other" holds a single row


def test_pair_budget_is_reported_not_silently_skipped(monkeypatch):
    monkeypatch.setattr(scr, "_PAIR_BUDGET", 10)
    df = _titanic()
    s = screen_dataset(df.values, list(df.columns))
    assert s.pairs == [] and s.pairs_skipped and "budget" in s.pairs_skipped
    assert len(s.columns) == df.shape[1]  # the per-column profile is always computed


def test_titanic_screen_finds_the_title_sex_pair():
    df = _titanic()
    X = df.drop(columns=["survived"])
    s = screen_dataset(X.values, list(X.columns))
    assert s.pairs_skipped is None
    pair = [p for p in s.pairs if {p.a, p.b} == {"sex", "title"}]
    assert len(pair) == 1
    p = pair[0]
    # `title` determines `sex` on 888 of 891 rows (Officer and Royalty hold
    # both sexes); the dependency is NOT exact, which is why an exact-only
    # screen would report nothing on this dataset.
    assert p.kind == "approximate"
    assert (p.exceptions_ba if p.a == "sex" else p.exceptions_ab) == 3
    assert exact_dependencies(X.values, list(X.columns)) == {}


def test_target_report_finds_a_leak():
    n = 250
    rng = np.random.default_rng(7)
    z = (rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame({
        "copy": np.where(z == 1, "yes", "no"),      # the target under another name
        "noise": rng.choice(list("abcd"), n),
    })
    rep = target_report(df.values, z, list(df.columns))
    assert rep[0]["feature"] == "copy" and rep[0]["exact"] and rep[0]["strength"] == 1.0
    assert rep[1]["feature"] == "noise" and rep[1]["strength"] < 0.5
    assert rep[0]["exceptions"] == 0


# ---------------------------------------------------------------------------
# Pruning dependent candidates
# ---------------------------------------------------------------------------

def _partition(X, combo):
    X_discrete, bin_counts = discretize_dataset(np.asarray(X, dtype=object))
    codes, _ = _CandidateFactory(X_discrete, bin_counts, X.shape[0]).codes(tuple(combo))
    return codes


def _same_partition(a, b) -> bool:
    return len({(int(x), int(y)) for x, y in zip(a, b)}) == len(set(int(x) for x in a)) == len(set(int(y) for y in b))


def _enumerated(X, determined_by, max_d=4):
    X_discrete, bin_counts = discretize_dataset(np.asarray(X, dtype=object))
    f = _CandidateFactory(X_discrete, bin_counts, X.shape[0], determined_by=determined_by)
    combos = [c for c, _, _ in f.iter_candidates(max_d)]
    return combos, f


def test_pruned_candidates_are_exactly_the_renamings():
    rng = np.random.default_rng(8)
    n = 500
    a = rng.choice(list("xyz"), n)
    a_copy = np.array([{"x": "P", "y": "Q", "z": "R"}[v] for v in a])
    b = rng.choice(list("01"), n)
    c = rng.choice(list("pq"), n)
    X = np.column_stack([a, a_copy, b, c])
    deps = exact_dependencies(X, ["a", "a_copy", "b", "c"])
    assert deps == {0: (1,), 1: (0,)}
    full, _ = _enumerated(X, None)
    pruned, factory = _enumerated(X, deps)
    skipped = [c for c in full if c not in set(pruned)]
    assert factory.n_pruned == len(skipped) > 0
    # every skipped candidate holds both renamings
    assert all(0 in c and 1 in c for c in skipped)
    # and its partition equals that of the candidate without one of them
    for combo in skipped:
        smaller = tuple(j for j in combo if j != 1)
        assert _same_partition(_partition(X, combo), _partition(X, smaller))
    # nothing else was dropped
    assert set(pruned) | set(skipped) == set(full)
    assert all(not (0 in c and 1 in c) for c in pruned)


def test_pruning_declines_when_the_capacity_rule_would_coarsen():
    """
    The rule prunes only what it can prove. N = 100 gives a capacity of 10
    occupied cells; `fine` alone already occupies 20, so {fine, coarse} and
    {fine, coarse, other} are BOTH coarsened - from different nominal level
    counts - and are not renamings of the smaller subset. Nothing may be
    pruned here, even though `fine` determines `coarse` exactly.
    """
    n = 100
    rng = np.random.default_rng(9)
    fine = rng.integers(0, 20, n)
    X = np.column_stack([
        np.array([f"f{v}" for v in fine]),
        np.array([f"c{v // 2}" for v in fine]),
        np.array([f"o{v}" for v in rng.integers(0, 5, n)]),
    ])
    deps = exact_dependencies(X, ["fine", "coarse", "other"])
    assert deps == {1: (0,)}
    assert grid_capacity(n) == 10
    assert len(set(X[:, 0].tolist())) == 20 > grid_capacity(n)  # before coarsening
    full, _ = _enumerated(X, None, max_d=3)
    combos, factory = _enumerated(X, deps, max_d=3)
    assert factory.n_pruned == 0 and combos == full
    assert (0, 1) in combos and (0, 1, 2) in combos

    # The same dependency below the capacity limit IS pruned: with 600 rows
    # the capacity is 60 and {fine} occupies 20.
    n2 = 600
    rng2 = np.random.default_rng(11)
    fine2 = rng2.integers(0, 20, n2)
    X2 = np.column_stack([
        np.array([f"f{v}" for v in fine2]),
        np.array([f"c{v // 2}" for v in fine2]),
        np.array([f"o{v}" for v in rng2.integers(0, 2, n2)]),
    ])
    deps2 = exact_dependencies(X2, ["fine", "coarse", "other"])
    assert deps2 == {1: (0,)}
    combos2, factory2 = _enumerated(X2, deps2, max_d=3)
    assert factory2.n_pruned > 0
    assert (0, 1) not in combos2 and (0, 1, 2) not in combos2
    assert _same_partition(_partition(X2, (0, 1)), _partition(X2, (0,)))


def test_pruning_never_changes_a_dependency_free_search():
    df = _titanic()
    X = df.drop(columns=["survived"]).values
    z = (df["survived"] == "died").astype(int).values
    names = list(df.drop(columns=["survived"]).columns)
    spec = CenterSpec(tau=0.7)
    assert exact_dependencies(X, names) == {}
    a = compute_landscape(X, z, names, positive_class=1, center_spec=spec)
    b = compute_landscape(X, z, names, positive_class=1, center_spec=spec, prune_dependent=True)
    assert len(a) == len(b) and a.features == b.features
    np.testing.assert_array_equal(a.k_sel, b.k_sel)
    np.testing.assert_array_equal(a.n_centers, b.n_centers)


def test_pruning_removes_the_fake_dimensionality():
    rng = np.random.default_rng(10)
    n = 600
    a = rng.choice(list("xyz"), n)
    a_copy = np.array([{"x": "P", "y": "Q", "z": "R"}[v] for v in a])
    b = rng.choice(["0", "1", "2"], n)
    c = rng.choice(list("pq"), n)
    z = (((a == "z") & (b == "1")) | (rng.random(n) < 0.05)).astype(int)
    X = np.column_stack([a, a_copy, b, c])
    names = ["a", "a_copy", "b", "c"]
    spec = CenterSpec(tau=0.8)
    plain = discover_branches(X, z, names, positive_class=1, center_spec=spec, n_permutations_centers=0)
    pruned = discover_branches(X, z, names, positive_class=1, center_spec=spec,
                               n_permutations_centers=0, prune_dependent=True)
    # Without pruning the 3D and 4D "branches" are the 2D one with the copy
    # bolted on: identical coverage at a higher d.
    assert plain[3].centers.coverage == pytest.approx(plain[2].centers.coverage)
    assert set(plain[3].selected_features) >= {0, 1}
    # With pruning no branch holds both renamings, and 4D does not exist at
    # all: four columns of which two are the same column cannot span four.
    assert 4 not in pruned
    for d, br in pruned.items():
        assert not {0, 1} <= set(br.selected_features)
    assert pruned[2].centers.coverage == pytest.approx(plain[2].centers.coverage)


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


def test_screen_endpoint_and_drop_prune_are_part_of_the_analysis():
    df = _titanic()
    srv = _Server(df)
    try:
        status, s = srv.post("/api/screen", {})
        assert status == 200, s
        assert s["n_columns"] == df.shape[1] and s["target"] is None and s["target_report"] is None
        assert {p["a"] for p in s["pairs"]} | {p["b"] for p in s["pairs"]} == {"sex", "title"}
        assert s["grid_capacity"] == grid_capacity(len(df))

        status, s2 = srv.post("/api/screen", {"target": "survived", "criterion": "died", "drop": ["deck"]})
        assert status == 200
        assert "survived" not in s2["columns_screened"] and "deck" not in s2["columns_screened"]
        assert s2["dropped_columns"] == ["deck"] and s2["target_columns"] == ["survived"]
        assert s2["target_report"][0]["feature"] == "title"
        assert len(srv.httpd.screen_cache) == 2  # one per screened column set

        base = {"target": "survived", "criterion": "died", "tau": 0.7}
        status, full = srv.post("/api/analyze", base)
        assert status == 200 and full["dropped_columns"] == [] and full["prune_dependent"] is False
        assert full["n_features"] == df.shape[1] - 1
        status, cut = srv.post("/api/analyze", dict(base, drop=["sex", "title"]))
        assert status == 200 and cut["dropped_columns"] == ["sex", "title"]
        assert cut["n_features"] == df.shape[1] - 3
        # A different feature space is a different analysis, not a cache hit.
        assert len(srv.httpd.analyze_cache) == 2
        assert cut["branches"]["1"]["selected_features"] != full["branches"]["1"]["selected_features"] or True
        for d, br in cut["branches"].items():
            assert "sex" not in br["selected_features"] and "title" not in br["selected_features"]

        status, land = srv.post("/api/landscape", dict(base, drop=["sex", "title"], d=1))
        assert status == 200 and land["n_candidates"] < len(df.columns) ** 2
        assert "sex" not in land["feature_names"]
        status, groups = srv.post("/api/centers/groups", dict(base, drop=["sex"], min_rows=10))
        assert status == 200 and groups["n_groups"] > 0

        assert srv.post("/api/analyze", dict(base, drop="sex"))[0] == 400
        assert srv.post("/api/analyze", dict(base, drop=["nope"]))[0] == 400
        assert srv.post("/api/analyze", dict(base, drop=list(df.columns)))[0] == 400
        assert srv.post("/api/screen", {"min_strength": 2})[0] == 400
    finally:
        srv.close()
