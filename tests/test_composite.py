"""
Composite targets: a conjunction of two or more (column, value) pairs as
the indicator searched (`/api/target`, `/api/analyze` with `also`, the
landscape family). The search itself is untouched - a conjunction is one
more 0/1 column - so these tests pin the two things that ARE new: the
indicator is the exact conjunction, and every column of the target leaves
the feature space.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from vsf.avr import discover_branches, report_schema
from vsf.centers import CenterSpec
from vsf.server import _MAX_TARGET_CONJUNCTS, _Target, _analyze_key, _build_server, _resolve_target, _target_arrays


def _df(seed: int = 0, n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.choice(["x", "y", "z"], size=n)
    b = rng.choice(["0", "1"], size=n)
    c = rng.choice(["p", "q", "r", "s"], size=n)
    d = rng.choice(["m", "n"], size=n)
    e = rng.choice(["u", "v", "w"], size=n)
    # survived depends on (c, e); sex is b. "survived ∧ b = 1" concentrates on (c, e) too.
    pos = ((c == "s") & (e == "w")) | ((c == "p") & (e == "u") & (b == "1")) | (rng.random(n) < 0.05)
    return pd.DataFrame({"survived": np.where(pos, "yes", "no"), "a": a, "b": b, "c": c, "d": d, "e": e})


class _Server:
    def __init__(self, df, prefetch=False):
        self.httpd = _build_server(df, host="127.0.0.1", port=0, translations=None, prefetch=prefetch)
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


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_resolve_target_builds_the_conjunction_and_drops_its_columns():
    df = _df()
    t, err = _resolve_target(df, {"target": "survived", "criterion": "yes", "also": [["b", "1"], ["d", "m"]]}, "survived")
    assert err is None
    assert t == _Target("survived", "yes", (("b", "1"), ("d", "m")))   # sorted, hashable
    X_df, Z, sort_Z = _target_arrays(df, t)
    expected = ((df["survived"] == "yes") & (df["b"] == "1") & (df["d"] == "m")).astype(int).values
    assert np.array_equal(Z, expected)
    assert list(X_df.columns) == ["a", "c", "e"]
    assert np.array_equal(sort_Z, df["survived"].values)
    # order of `also` does not change the key
    t2, _ = _resolve_target(df, {"target": "survived", "criterion": "yes", "also": [["d", "m"], ["b", "1"]]}, "survived")
    assert _analyze_key(t2.params()) == _analyze_key(t.params())
    # a plain target has no `also` in its params, so old keys are unchanged
    t3, _ = _resolve_target(df, {"target": "survived", "criterion": "yes"}, "survived")
    assert t3.params() == {"target_col": "survived", "criterion": "yes"}


@pytest.mark.parametrize("req, fragment", [
    ({"target": "survived", "criterion": "yes", "also": [["b", "1"], ["b", "0"]]}, "twice"),
    ({"target": "survived", "criterion": "yes", "also": [["survived", "no"]]}, "twice"),
    ({"target": "survived", "criterion": "yes", "also": [["b", "7"]]}, "never takes"),
    ({"target": "survived", "criterion": "yes", "also": [["nope", "1"]]}, "not a column"),
    ({"target": "survived", "also": [["b", "1"]]}, "explicit criterion"),
    ({"target": "survived", "criterion": "yes", "also": [["b", "1"], ["d", "m"], ["a", "x"]]}, "at most"),
    ({"target": "survived", "criterion": "yes", "also": "b=1"}, "list of"),
    ({"target": "survived", "criterion": "yes", "also": [["b"]]}, "list of"),
])
def test_resolve_target_rejects_malformed_conjunctions(req, fragment):
    t, err = _resolve_target(_df(), req, "survived")
    assert t is None and fragment in err
    assert _MAX_TARGET_CONJUNCTS == 3


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def test_target_endpoint_reports_the_conjunction_before_any_search():
    df = _df(1)
    srv = _Server(df)
    try:
        status, info = srv.post("/api/target", {"target": "survived", "criterion": "yes", "also": [["b", "1"]]})
        assert status == 200, info
        n_pos = int(((df["survived"] == "yes") & (df["b"] == "1")).sum())
        assert info["n_positive"] == n_pos and info["n_samples"] == len(df)
        assert info["share"] == pytest.approx(n_pos / len(df))
        assert info["target_display"] == "survived = yes ∧ b = 1"
        assert info["target_columns"] == ["survived", "b"]
        assert info["feature_names"] == ["a", "c", "d", "e"] and info["n_features"] == 4
        status, plain = srv.post("/api/target", {"target": "survived", "criterion": "yes"})
        assert plain["also"] == [] and plain["feature_names"] == ["a", "b", "c", "d", "e"]
        assert plain["n_positive"] > info["n_positive"]
        assert srv.post("/api/target", {"target": "survived", "also": [["b", "1"]]})[0] == 400
    finally:
        srv.close()


def test_analyze_with_a_composite_target_equals_the_search_on_the_conjunction():
    df = _df(2)
    srv = _Server(df, prefetch=True)
    base = {"target": "survived", "criterion": "yes", "also": [["b", "1"]], "tau": 0.45}
    try:
        status, res = srv.post("/api/analyze", base)
        assert status == 200, res
        assert res["also"] == [["b", "1"]] and res["target_columns"] == ["survived", "b"]
        assert res["target_display"] == "survived = yes ∧ b = 1"
        Z = ((df["survived"] == "yes") & (df["b"] == "1")).astype(int).values
        assert res["n_positive"] == int(Z.sum())
        X_df = df.drop(columns=["survived", "b"])
        ref = discover_branches(
            X_df.values, Z, list(X_df.columns), max_d=4, positive_class=1,
            center_spec=CenterSpec(tau=0.45), random_state=42, n_permutations_centers=0,
        )
        for d_str, br in res["branches"].items():
            r = ref[int(d_str)]
            assert br["search_centers"]["coverage"] == pytest.approx(r.centers.coverage)
            assert br["search_centers"]["n_centers"] == r.centers.n_centers
            assert br["selected_feature_indices"] == list(r.selected_features)
            assert not ({"survived", "b"} & set(r.selected_feature_names))
        # no sibling prefetch for a conjunction; a plain target starts one
        assert srv.httpd.prefetch_thread is None
        assert len(srv.httpd.analyze_cache) == 1
        status, plain = srv.post("/api/analyze", {"target": "survived", "criterion": "yes", "tau": 0.45})
        assert status == 200 and plain["also"] == [] and plain["target_display"] == "survived = yes"
        assert plain["n_positive"] > res["n_positive"]
        assert len(srv.httpd.analyze_cache) >= 2   # a different key
        # opening a schema from the landscape uses the SAME feature indexing
        status, ls = srv.post("/api/landscape", dict(base, d=None))
        assert status == 200 and ls["feature_names"] == ["a", "c", "d", "e"] and ls["also"] == [["b", "1"]]
        assert ls["n_total"] == 15   # C(4,1) + C(4,2) + C(4,3) + C(4,4)
        cells = [(iy, ix) for iy in range(10) for ix in range(10) if ls["counts"][iy][ix]]
        iy, ix = cells[-1]
        status, cell = srv.post("/api/landscape/cell", dict(base, d=None, ix=ix, iy=iy))
        feats = cell["schemas"][0]["features"]
        status, opened = srv.post("/api/analyze", dict(base, features=feats))
        assert status == 200, opened
        assert opened["schema"]["feature_names"] == [["a", "c", "d", "e"][j] for j in feats]
        assert opened["branches"][str(len(feats))]["search_centers"]["coverage"] == pytest.approx(cell["schemas"][0]["coverage"])
        ref2 = report_schema(X_df.values, Z, feats, list(X_df.columns), positive_class=1,
                             center_spec=CenterSpec(tau=0.45), n_permutations_centers=0)
        assert opened["branches"][str(len(feats))]["search_centers"]["coverage"] == pytest.approx(ref2[len(feats)].centers.coverage)
        # tau-curves and absence on the conjunction
        status, curves = srv.post("/api/landscape/curves", base)
        assert status == 200 and curves["also"] == [["b", "1"]] and set(curves["curves"]) == {"1", "2", "3", "4"}
        assert curves["n_positive"] == int(Z.sum())
        status, absent = srv.post("/api/analyze", dict(base, direction="absence", tau=0.95))
        assert status == 200, absent
        assert absent["target_display"] == "not (survived = yes ∧ b = 1)"
        assert absent["n_positive"] == len(df) - int(Z.sum())
        assert srv.post("/api/analyze", dict(base, also=[["b", "1"], ["b", "0"]]))[0] == 400
    finally:
        srv.close()
