"""
`vsf.free_rules`: the selection behind the app's "Free rules" overlay
(`/api/landscape/free_rules`) is the `rules_disjoint` baseline of
`experiments/baselines.py` run on all rows - pinned here against the
baseline itself, on the same data - plus the endpoint and the session log.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from vsf.avr import _prepare_search, compute_landscape, family_cell_count
from vsf.centers import CenterSpec
from vsf.free_rules import (
    FreeRuleFamily,
    budgeted_greedy,
    certification_threshold,
    rule_candidates,
    select_free_rules,
)
from vsf.server import _build_server

_ROOT = Path(__file__).resolve().parents[1]


def _baselines() -> ModuleType:
    """
    `experiments/baselines.py`. It imports scikit-learn, rulelist and pydl8.5
    only inside the selectors that need them, so `select_rules` loads with
    the core dependencies alone and this test runs in the locked CI job too.
    """
    name = "vsf_experiments_baselines_for_free_rules"
    spec = importlib.util.spec_from_file_location(name, _ROOT / "experiments" / "baselines.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _two_region_data(seed: int, n: int = 1200) -> Tuple[np.ndarray, np.ndarray]:
    """Positives at (x0=0, x1=0) and at (x2=0, x3=0): two rules on different columns."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, 5))
    a = (X[:, 0] == 0) & (X[:, 1] == 0)
    b = (X[:, 2] == 0) & (X[:, 3] == 0)
    z = np.where(rng.random(n) < np.where(a | b, 0.97, 0.03), 1, 0)
    return X.astype(str), z


def _titanic() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(_ROOT / "benchmark_data" / "titanic.csv", dtype=str, keep_default_na=False)
    X_df = df.drop(columns=["survived"])
    return X_df.values, (df["survived"] == "survived").astype(int).values, list(X_df.columns)


def _cases() -> List[Tuple[str, np.ndarray, np.ndarray, Optional[List[str]], float, int]]:
    X1, z1 = _two_region_data(3)
    X2, z2 = _two_region_data(11, n=600)
    Xt, zt, names_t = _titanic()
    return [
        ("two_regions", X1, z1, None, 0.9, 1),
        ("two_regions_m20", X2, z2, None, 0.8, 20),
        ("titanic", Xt, zt, names_t, 0.9, 1),
        ("titanic_m20", Xt, zt, names_t, 0.7, 20),
    ]


_BUDGETS = (1, 2, 3, 4, 8, 16, 32)


@pytest.mark.parametrize("rule", ["purity", "certified"])
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c[0])
def test_free_rules_are_the_rules_disjoint_baseline_on_the_same_rows(case, rule: str) -> None:
    """
    For every budget: the same rules, in the same order, with the same
    members, cost and covered count as `baselines.select_rules(disjoint=True)`
    given every row as its training fold - under the observed-purity rule and
    under the family certificate (`run_split(selection="certified")`'s
    threshold).
    """
    bl = _baselines()
    _, X, z, names, tau, m = case
    multiplicity = "family" if rule == "certified" else "bonferroni"
    spec = CenterSpec(tau=tau, min_samples=m, rule=rule, multiplicity=multiplicity)
    prepared = _prepare_search(X, z, names, 1, CenterSpec(tau=tau, min_samples=m), "presence")
    assert prepared is not None
    factory, zb, feature_names = prepared
    X_all = factory._raw.astype(np.int64, copy=False)
    n = int(zb.shape[0])
    train = np.arange(n)
    threshold = None
    if rule == "certified":
        threshold = bl.certification_threshold(n, tau, 0.05 / family_cell_count(factory, 4))
    baseline = bl.select_rules(
        X_all, train, zb, tau, m, _BUDGETS, feature_names, 4, threshold=threshold, disjoint=True,
    )
    family = FreeRuleFamily(X, z, names, 1, spec)
    n_pos = int(zb.sum())
    non_empty = 0
    for budget in _BUDGETS:
        ours = family.select(budget)
        groups = baseline.groups[budget]
        assert len(ours.rules) == len(groups)
        assert ours.conditions == sum(g.cost for g in groups) <= budget
        for rule_, group in zip(ours.rules, groups):
            members = np.zeros(n, dtype=bool)
            members[rule_.rows] = True
            assert np.array_equal(members, group.members)
            assert rule_.cost == group.cost
            assert rule_.n == int(group.members.sum())
            assert rule_.k == int(zb[group.members].sum())
            # label = "col=v & col=v" over the rule's columns, in order
            assert [part.split("=")[0] for part in group.label.split(" & ")] == [
                feature_names[j] for j in rule_.features
            ]
        assert ours.coverage == pytest.approx(baseline.train_coverage[budget], abs=1e-12)
        assert ours.k_covered == round(baseline.train_coverage[budget] * max(1, n_pos))
        non_empty += bool(ours.rules)
        # one-shot entry point = the family's selection
        direct = select_free_rules(X, z, budget, names, 1, spec)
        assert [r.features for r in direct.rules] == [r.features for r in ours.rules]
        assert [r.rows.tolist() for r in direct.rules] == [r.rows.tolist() for r in ours.rules]
    if rule == "purity":
        assert non_empty > 0  # the comparison is not vacuous


def test_baselines_module_uses_the_library_functions() -> None:
    """No copy: the baseline's names are the library's objects."""
    bl = _baselines()
    assert bl._rule_candidates is rule_candidates
    assert bl.budgeted_greedy is budgeted_greedy
    assert bl.certification_threshold is certification_threshold


def test_selected_rules_are_disjoint_qualifying_and_within_budget() -> None:
    X, z, names = _titanic()
    for tau, m in ((0.9, 1), (0.7, 20)):
        spec = CenterSpec(tau=tau, min_samples=m)
        for budget in (1, 5, 8, 20):
            out = select_free_rules(X, z, budget, names, 1, spec)
            assert out.conditions <= budget
            seen = np.zeros(out.n_samples, dtype=int)
            for r in out.rules:
                seen[r.rows] += 1
                assert r.n >= m and r.k >= np.ceil(tau * r.n - 1e-9)
                assert 1 <= r.cost <= 4
                # the rows are exactly those matching the conjunction
                first = r.rows[0]
                match = np.all(X[:, list(r.features)] == X[first, list(r.features)], axis=1)
                assert np.array_equal(np.flatnonzero(match), r.rows)
                assert r.k == int(z[r.rows].sum())
                assert r.new_positives == r.k  # disjoint: nothing is counted twice
            assert seen.max(initial=0) <= 1
            assert out.k_covered == int(z[seen == 1].sum())
            assert out.n_covered == int((seen == 1).sum())
            assert out.purity >= tau - 1e-12 or not out.rules


def test_budget_and_certificate_validation() -> None:
    X, z = _two_region_data(5, n=300)
    with pytest.raises(ValueError):
        select_free_rules(X, z, 0, None, 1, CenterSpec(tau=0.9))
    with pytest.raises(ValueError):  # per-schema correction has no meaning for a free rule
        select_free_rules(X, z, 4, None, 1, CenterSpec(tau=0.9, rule="certified", multiplicity="bonferroni"))
    with pytest.raises(ValueError):
        select_free_rules(X, z, 4, None, 1, CenterSpec(tau=0.9), max_len=5)


def test_landscape_locate_agrees_with_cell_listing() -> None:
    X, z = _two_region_data(7, n=900)
    L = compute_landscape(X, z, feature_names=list("abcde"), positive_class=1, center_spec=CenterSpec(tau=0.9))
    drawn = 0
    for i, feats in enumerate(L.features):
        loc = L.locate(list(reversed(feats)))
        if L.n_centers[i] == 0:
            assert loc is None
            continue
        drawn += 1
        assert loc is not None and loc["d"] == len(feats)
        in_d = L.cell(len(feats), loc["ix"], loc["iy"], limit=1000)["schemas"]
        in_all = L.cell(None, loc["ix"], loc["iy_all"], limit=1000)["schemas"]
        assert list(feats) in [s["features"] for s in in_d]
        assert list(feats) in [s["features"] for s in in_all]
    assert drawn > 0
    assert L.locate([0, 1, 2, 3, 4]) is None  # not a scored schema


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class _Server:
    def __init__(self, df: pd.DataFrame, **kwargs: object) -> None:
        self.httpd = _build_server(df, host="127.0.0.1", port=0, translations=None, prefetch=False, **kwargs)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]

    def post(self, path: str, body: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def close(self) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()


def _titanic_df() -> pd.DataFrame:
    return pd.read_csv(_ROOT / "benchmark_data" / "titanic.csv", dtype=str, keep_default_na=False)


def test_free_rules_endpoint_matches_the_library_and_the_landscape() -> None:
    df = _titanic_df()
    X, z, names = _titanic()
    srv = _Server(df)
    try:
        base = {"target": "survived", "criterion": "survived", "tau": 0.9, "min_samples": 5, "rule": "purity"}
        status, out = srv.post("/api/landscape/free_rules", dict(base, budget=8))
        assert status == 200, out
        ref = select_free_rules(X, z, 8, names, 1, CenterSpec(tau=0.9, min_samples=5))
        assert out["budget"] == 8 and out["conditions"] == ref.conditions <= 8
        assert out["coverage"] == pytest.approx(ref.coverage)
        assert out["k_covered"] == ref.k_covered and out["n_covered"] == ref.n_covered
        assert out["n_positive"] == int(z.sum()) and out["n_samples"] == len(df)
        assert out["evaluated_on"] == "all_rows"
        assert out["feature_names"] == names
        assert [r["features"] for r in out["rules"]] == [list(r.features) for r in ref.rules]
        L = compute_landscape(X, z, feature_names=names, positive_class=1,
                              center_spec=CenterSpec(tau=0.9, min_samples=5))
        for row, r in zip(out["rules"], ref.rules):
            assert (row["n"], row["k"], row["new_positives"], row["cost"]) == (r.n, r.k, r.new_positives, r.cost)
            assert row["purity"] == pytest.approx(r.k / r.n)
            assert row["coverage_gain"] == pytest.approx(r.new_positives / int(z.sum()))
            # the conditions describe exactly the rule's rows
            mask = np.ones(len(df), dtype=bool)
            for cond in row["conditions"]:
                assert names[cond["feature"]] == cond["column"]
                mask &= (df[cond["column"]] == cond["value"]).values
            assert np.array_equal(np.flatnonzero(mask), r.rows)
            assert row["landscape"] == L.locate(r.features)
        assert any(row["landscape"] is not None for row in out["rules"])

        # default budget is 8, never the branch's own condition count
        status, default = srv.post("/api/landscape/free_rules", base)
        assert status == 200 and default["budget"] == 8
        assert [r["features"] for r in default["rules"]] == [r["features"] for r in out["rules"]]

        # a different budget reuses the cached candidates
        status, small = srv.post("/api/landscape/free_rules", dict(base, budget=2))
        assert status == 200 and small["conditions"] <= 2
        assert small["coverage"] <= out["coverage"] + 1e-12
        assert len(srv.httpd.free_rules_cache) == 1

        # tau, the size floor and the certified flag all reach the selection
        status, strict = srv.post("/api/landscape/free_rules", dict(base, tau=0.8, rule="certified", budget=8))
        assert status == 200 and strict["rule"] == "certified"
        ref_c = select_free_rules(
            X, z, 8, names, 1, CenterSpec(tau=0.8, min_samples=5, rule="certified", multiplicity="family"))
        assert [r["features"] for r in strict["rules"]] == [list(r.features) for r in ref_c.rules]
        assert strict["k_covered"] == ref_c.k_covered
        status, big_m = srv.post("/api/landscape/free_rules", dict(base, min_samples=40, budget=8))
        assert status == 200 and all(r["n"] >= 40 for r in big_m["rules"])

        for bad in (0, -1, 65, "x", 2.5, None):
            assert srv.post("/api/landscape/free_rules", dict(base, budget=bad))[0] == (200 if bad is None else 400)
        assert srv.post("/api/landscape/free_rules", dict(base, direction="absence", tau=0.9))[0] == 400
        assert srv.post("/api/landscape/free_rules", dict(base, tau=0.01))[0] == 400  # below base rate
    finally:
        srv.close()


def test_session_log_records_events_in_order(tmp_path: Path) -> None:
    log = tmp_path / "sessions" / "study.jsonl"
    srv = _Server(_titanic_df(), session_log=log)
    try:
        status, info = srv.post("/api/session/log", {"session": "s1", "event": "session_start", "data": {"participant": "P07"}})
        assert status == 200 and info == {"logged": True}
        srv.post("/api/session/log", {"session": "s1", "event": "free_rules_toggle", "data": {"on": True, "budget": 8}})
        srv.post("/api/session/log", {"session": "s1", "event": "free_rules_toggle", "data": {"on": False}})
        assert srv.post("/api/session/log", {"session": "s1"})[0] == 400           # no event
        assert srv.post("/api/session/log", {"event": "x" * 200, "session": "s1"})[0] == 400
        assert srv.post("/api/session/log", {"event": "e", "session": "s1", "data": [1]})[0] == 400
        lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert [r["event"] for r in lines] == ["session_start", "free_rules_toggle", "free_rules_toggle"]
        assert [r["seq"] for r in lines] == [1, 2, 3]
        assert lines[1]["data"] == {"on": True, "budget": 8} and lines[2]["data"] == {"on": False}
        assert all(r["session"] == "s1" and r["ts"].endswith("+00:00") for r in lines)
        assert [r["ts"] for r in lines] == sorted(r["ts"] for r in lines)
    finally:
        srv.close()


def test_session_log_off_is_reported_not_silent() -> None:
    srv = _Server(_titanic_df())
    try:
        status, info = srv.post("/api/session/log", {"session": "s1", "event": "free_rules_toggle", "data": {"on": True}})
        assert status == 200 and info == {"logged": False}
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# Page: the feature is off until switched on
# ---------------------------------------------------------------------------

def test_page_ships_the_toggle_off_with_a_budget_of_eight() -> None:
    import re
    from vsf.server import _read_webapp_asset

    html = _read_webapp_asset("index.html")
    toggle = re.search(r'<input[^>]*id="freeRulesToggle"[^>]*>', html)
    assert toggle is not None and "checked" not in toggle.group(0)
    budget = re.search(r'<input[^>]*id="freeRulesBudget"[^>]*>', html)
    assert budget is not None and 'value="8"' in budget.group(0) and 'max="64"' in budget.group(0)
    panel = re.search(r'<aside[^>]*id="freeRulesPanel"[^>]*>', html)
    assert panel is not None and "display: none" in panel.group(0)
    js = _read_webapp_asset("static/js/app.js")
    assert "computed on all rows, not checked on new data" in js
    assert "logSessionEvent('free_rules_toggle'" in js
    # the budget is the field's, never the selected branch's condition count
    request = js[js.index("function freeRulesRequest()"):js.index("// What the session log says")]
    assert "budget: freeRulesBudget()" in request and "conditions" not in request
