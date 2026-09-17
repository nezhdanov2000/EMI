"""
Rules view (`vsf.rules.enumerate_rules`, `/api/rules`, Section 4.15): every
cell of the given schemas as a conjunctive rule, with its generalisations
computed from the data.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from vsf.centers import CenterSpec, select_centers
from vsf.rules import enumerate_rules
from vsf.server import _build_server


def _df(seed: int = 0, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], size=n)
    age = rng.choice(["child", "adult", "senior"], size=n)
    fare = rng.choice(["low", "high"], size=n)
    p = np.where(sex == "male", 0.85, 0.2)
    p = np.where((sex == "male") & (age == "child"), 0.6, p)
    z = rng.random(n) < p
    return pd.DataFrame({"survived": np.where(z, "yes", "no"), "sex": sex, "age": age, "fare": fare})


def _hand_stats(df, cols, values):
    mask = np.ones(len(df), dtype=bool)
    for c, v in zip(cols, values):
        mask &= (df[c].astype(str) == v).values
    n = int(mask.sum())
    k = int(((df["survived"] == "yes").values & mask).sum())
    return n, k


def test_rules_are_the_cells_with_parents_from_the_data():
    df = _df()
    X = df[["sex", "age", "fare"]].values
    Z = (df["survived"] == "yes").astype(int).values
    names = ["sex", "age", "fare"]
    spec = CenterSpec(tau=0.8)
    out = enumerate_rules(X, Z, [[0], [0, 1], [0, 1, 2]], names, spec)
    rules = out["rules"]
    assert out["n_samples"] == len(df) and out["n_positive"] == int(Z.sum())
    # one rule per occupied cell of each schema
    assert sum(1 for r in rules if r["d"] == 1) == 2
    assert sum(1 for r in rules if r["d"] == 2) == 6
    assert sum(1 for r in rules if r["d"] == 3) == 12
    # sorted by purity, then rows
    keys = [(-r["purity"], -r["n"], r["d"]) for r in rules]
    assert keys == sorted(keys)
    for r in rules:
        cols = [c["column"] for c in r["conditions"]]
        vals = [c["value"] for c in r["conditions"]]
        n, k = _hand_stats(df, cols, vals)
        assert (r["n"], r["k"]) == (n, k) and r["purity"] == pytest.approx(k / n)
        assert r["features"] == [names.index(c) for c in cols]
        if r["d"] == 1:
            assert r["parents"] == [] and r["best_generalisation"] is None
            continue
        assert len(r["parents"]) == r["d"]
        for p, g in enumerate(r["parents"]):
            keep = [i for i in range(r["d"]) if i != p]
            n_p, k_p = _hand_stats(df, [cols[i] for i in keep], [vals[i] for i in keep])
            assert g["features"] == [names.index(cols[i]) for i in keep]
            assert (g["n"], g["k"]) == (n_p, k_p) and g["purity"] == pytest.approx(k_p / n_p)
        # best generalisation is the purest proper sub-conjunction
        best = r["best_generalisation"]
        assert best["purity"] >= max(g["purity"] for g in r["parents"])
    # the informative case: male ∧ child is LOWER than male, and is kept
    mc = next(r for r in rules if r["d"] == 2 and [c["value"] for c in r["conditions"]] == ["male", "child"])
    male = next(r for r in rules if r["d"] == 1 and r["conditions"][0]["value"] == "male")
    assert mc["purity"] < male["purity"]
    assert mc["parents"][1]["purity"] == pytest.approx(male["purity"])   # parent without `age = child`


def test_rules_certification_matches_select_centers_and_min_rows_filters():
    df = _df(1)
    X = df[["sex", "age", "fare"]].values
    Z = (df["survived"] == "yes").astype(int).values
    spec = CenterSpec(tau=0.7, min_samples=5)
    out = enumerate_rules(X, Z, [[0, 1]], ["sex", "age", "fare"], spec)
    ks = np.array([r["k"] for r in out["rules"]]); ns = np.array([r["n"] for r in out["rules"]])
    cert, _ = select_centers(ks, ns, spec)
    assert [bool(c) for c in cert] == [r["certified"] for r in out["rules"]]
    small = enumerate_rules(X, Z, [[0, 1, 2]], ["sex", "age", "fare"], spec, min_rows=55)
    assert small["rules"] and all(r["n"] >= 55 for r in small["rules"])
    assert len(small["rules"]) < len(enumerate_rules(X, Z, [[0, 1, 2]], ["sex", "age", "fare"], spec)["rules"])
    with pytest.raises(ValueError):
        enumerate_rules(X, Z, [[0, 0]], ["sex", "age", "fare"], spec)
    with pytest.raises(ValueError):
        enumerate_rules(X, Z, [[7]], ["sex", "age", "fare"], spec)


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


def test_rules_endpoint():
    df = _df(2)
    srv = _Server(df)
    base = {"target": "survived", "criterion": "yes", "tau": 0.8}
    try:
        status, a = srv.post("/api/analyze", base)
        assert status == 200, a
        schemas = [a["branches"][str(d)]["selected_feature_indices"] for d in a["branch_dims"]]
        status, out = srv.post("/api/rules", dict(base, schemas=schemas, min_rows=1))
        assert status == 200, out
        assert out["schemas"] == schemas and out["feature_names"] == ["sex", "age", "fare"]
        assert out["tau"] == 0.8 and out["direction"] == "presence" and out["also"] == []
        ref = enumerate_rules(df[["sex", "age", "fare"]].values, (df["survived"] == "yes").astype(int).values,
                              schemas, ["sex", "age", "fare"], CenterSpec(tau=0.8))
        assert out["rules"] == json.loads(json.dumps(ref["rules"]))
        assert len(srv.httpd.rules_cache) == 1
        srv.post("/api/rules", dict(base, schemas=schemas, min_rows=1))
        assert len(srv.httpd.rules_cache) == 1        # cached
        status, absent = srv.post("/api/rules", dict(base, schemas=schemas, direction="absence", tau=0.95))
        assert status == 200 and absent["n_positive"] == len(df) - ref["n_positive"]
        r0 = absent["rules"][0]
        assert r0["k"] == r0["n"] - next(r["k"] for r in out["rules"] if r["conditions"] == r0["conditions"])
        assert srv.post("/api/rules", dict(base, schemas=[]))[0] == 400
        assert srv.post("/api/rules", dict(base, schemas=[[0, 9]]))[0] == 400
        assert srv.post("/api/rules", dict(base, schemas=[[0, 0]]))[0] == 400
        assert srv.post("/api/rules", dict(base, schemas=schemas, min_rows=0))[0] == 400
        assert srv.post("/api/rules", {"target": "survived", "schemas": schemas})[0] == 400  # needs a criterion
    finally:
        srv.close()
