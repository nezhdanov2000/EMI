"""
Absence search (`vsf.avr.Direction`, 2026-09).

The absence of a value is the presence of its complement: every number an
absence search reports is the corresponding number of a presence search on
the inverted indicator. These tests pin that identity end to end (library,
per-column search, server response bytes), the base-rate invariant that
makes either direction well-posed, and the reporting details that differ
between the two questions (labels, direction field, the mass headline).
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

import vsf
from vsf.avr import (
    base_rate_reason,
    discover_branches,
    discover_branches_by_value,
    iter_branches_by_value,
)
from vsf.centers import CenterSpec
from vsf.server import _build_server


def _df(seed: int = 0, n: int = 900) -> pd.DataFrame:
    """A three-valued target where the dominant value ("common", ~80%) is
    absent exactly where a == "z" or (b == "1" and c == "q")."""
    rng = np.random.default_rng(seed)
    a = rng.choice(["x", "y", "z"], size=n, p=[0.45, 0.45, 0.10])
    b = rng.choice(["0", "1"], size=n)
    c = rng.choice(["p", "q", "r"], size=n)
    absent = (a == "z") | ((b == "1") & (c == "q"))
    target = np.where(absent, rng.choice(["rare1", "rare2"], size=n), "common")
    target = np.where(~absent & (rng.random(n) < 0.03), "rare1", target)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": rng.choice(["m", "n"], size=n), "target": target})


# ---------------------------------------------------------------------------
# Identity: absence(X) == presence(not X)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [CenterSpec(tau=0.95), CenterSpec(rule="certified", tau=0.8)])
def test_absence_equals_presence_of_the_complement(spec):
    df = _df()
    X = df[["a", "b", "c", "d"]].values
    names = ["a", "b", "c", "d"]
    z = (df["target"] == "common").astype(int).values
    absence = discover_branches(
        X, z, feature_names=names, positive_class=1, center_spec=spec,
        direction="absence", n_permutations_centers=99, cv_repeats=2,
    )
    presence_of_not = discover_branches(
        X, 1 - z, feature_names=names, positive_class=1, center_spec=spec,
        direction="presence", n_permutations_centers=99, cv_repeats=2,
    )
    assert set(absence) == set(presence_of_not) == {1, 2, 3, 4}
    for d in absence:
        assert dataclasses.asdict(absence[d]) == dataclasses.asdict(presence_of_not[d])
    # The planted structure: a alone frees 10% of the rows, a+b+c the rest.
    assert absence[1].selected_feature_names == ["a"]
    assert absence[1].centers.mass > 0.05
    assert absence[3].centers.coverage > absence[1].centers.coverage
    # What the reported numbers mean under absence.
    assert absence[1].centers.prevalence == pytest.approx(1 - z.mean())


def test_absence_of_a_common_value_is_well_posed_where_its_presence_is_not():
    df = _df()
    X = df[["a", "b", "c", "d"]].values
    z = (df["target"] == "common").astype(int).values
    assert z.mean() > 0.7
    with pytest.raises(ValueError, match="not above the base rate"):
        discover_branches(X, z, positive_class=1, center_spec=CenterSpec(tau=0.7), cv_repeats=0)
    branches = discover_branches(
        X, z, positive_class=1, center_spec=CenterSpec(tau=0.7),
        direction="absence", cv_repeats=0, n_permutations_centers=0,
    )
    assert branches[1].centers.n_centers > 0


def test_direction_must_be_presence_or_absence():
    df = _df()
    with pytest.raises(ValueError, match="direction"):
        discover_branches(
            df[["a", "b"]].values, (df["target"] == "common").astype(int).values,
            positive_class=1, direction="sideways", cv_repeats=0,
        )


# ---------------------------------------------------------------------------
# Base-rate invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tau,prevalence,direction,void",
    [
        (0.9, 0.9, "presence", True),
        (0.9, 0.95, "presence", True),
        (0.9, 0.89, "presence", False),
        (0.9, 0.1, "absence", False),
        (0.05, 0.1, "absence", True),
        (0.5, 0.5, "presence", True),
        (0.5001, 0.5, "presence", False),
    ],
)
def test_base_rate_reason(tau, prevalence, direction, void):
    reason = base_rate_reason(CenterSpec(tau=tau), prevalence, direction)
    assert (reason is not None) == void
    if void:
        assert "base rate" in reason
        if direction == "presence":
            assert "absence" in reason  # the remedy is named


def test_per_column_search_skips_void_values_and_reports_why():
    df = _df()
    X = df[["a", "b", "c", "d"]].values
    values = ["common", "rare1", "rare2"]
    skipped = {}
    by_value = discover_branches_by_value(
        X, df["target"].values, values, center_spec=CenterSpec(tau=0.7),
        direction="presence", cv_repeats=0, n_permutations_centers=0, skipped=skipped,
    )
    assert set(skipped) == {"common"}
    assert by_value["common"] == {}
    assert by_value["rare1"] and by_value["rare2"]
    # Under absence "common" is the well-posed one; the rare values are
    # nearly everywhere-absent, so tau = 0.7 is below THEIR complement's base
    # rate and they are the ones skipped.
    skipped = {}
    by_value = discover_branches_by_value(
        X, df["target"].values, values, center_spec=CenterSpec(tau=0.7),
        direction="absence", cv_repeats=0, n_permutations_centers=0, skipped=skipped,
    )
    assert set(skipped) == {"rare1", "rare2"}
    assert by_value["common"]


def test_per_column_absence_search_equals_per_value_absence_search():
    df = _df(1)
    X = df[["a", "b", "c", "d"]].values
    names = ["a", "b", "c", "d"]
    values = ["common", "rare1", "rare2"]
    spec = CenterSpec(tau=0.97)
    by_value = dict(iter_branches_by_value(
        X, df["target"].values, values, feature_names=names, center_spec=spec,
        direction="absence", n_permutations_centers=49, cv_repeats=2,
    ))
    for v in values:
        z = (df["target"].astype(str) == v).astype(int).values
        ref = discover_branches(
            X, z, feature_names=names, positive_class=1, center_spec=spec,
            direction="absence", n_permutations_centers=49, cv_repeats=2,
        )
        assert set(ref) == set(by_value[v]), v
        for d in ref:
            assert dataclasses.asdict(ref[d]) == dataclasses.asdict(by_value[v][d]), (v, d)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

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
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def close(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()


def test_server_absence_response_matches_the_library_absence_search():
    df = _df(2)
    srv = _Server(df)
    try:
        status, body_abs = srv.post("/api/analyze", {"target": "target", "criterion": "common", "direction": "absence"})
        assert status == 200, body_abs[:200]
        abs_payload = json.loads(body_abs)
        assert abs_payload["direction"] == "absence"
        assert abs_payload["certificate"]["tau"] == 0.9
        X = df[["a", "b", "c", "d"]].values
        z = (df["target"] == "common").astype(int).values
        ref = discover_branches(X, z, feature_names=["a", "b", "c", "d"], positive_class=1,
                                direction="absence", n_permutations_centers=999)
        for d, br in ref.items():
            got = abs_payload["branches"][str(d)]["centers"]
            assert got["coverage"] == pytest.approx(br.centers.coverage)
            assert got["mass"] == pytest.approx(br.centers.mass)
            assert got["n_centers"] == br.centers.n_centers
            assert got["prevalence"] == pytest.approx(1 - z.mean())
        first = abs_payload["branches"][abs_payload["default_branch"]]
        assert first["target_name"].startswith("target")
        assert "≠" in first["target_name"]
        assert set(first["target_labels"]) <= {"target = common", "not target = common"}
        assert "mass_by_d" in first["view_metrics"]
    finally:
        srv.close()


def test_server_rejects_void_threshold_and_absence_without_criterion():
    df = _df(3)
    srv = _Server(df)
    try:
        status, body = srv.post("/api/analyze", {"target": "target", "criterion": "common", "tau": 0.7})
        assert status == 400
        assert "base rate" in json.loads(body)["error"]
        status, body = srv.post("/api/analyze", {"target": "target", "direction": "absence"})
        assert status == 400
        status, body = srv.post("/api/analyze", {"target": "target", "criterion": "common", "direction": "up"})
        assert status == 400
    finally:
        srv.close()


def test_server_cache_separates_the_two_directions():
    df = _df(4)
    srv = _Server(df)
    try:
        s1, b1 = srv.post("/api/analyze", {"target": "target", "criterion": "rare1", "direction": "presence"})
        s2, b2 = srv.post("/api/analyze", {"target": "target", "criterion": "rare1", "direction": "absence", "tau": 0.99})
        assert s1 == 200 and s2 == 200
        assert b1 != b2
        assert len(srv.httpd.analyze_cache) == 2
        assert json.loads(b1)["direction"] == "presence"
        assert json.loads(b2)["direction"] == "absence"
    finally:
        srv.close()


def test_scan_in_absence_mode_reports_direction_and_skipped_pairs():
    df = _df(5)[["a", "b", "target"]]
    srv = _Server(df)
    try:
        # tau = 0.8: "common" (~80% of rows) has a complement base rate of
        # ~20% and is searchable; each rare value is absent from ~85% of the
        # rows, above tau, so its absence search is void and must be skipped
        # with a reason rather than reported as trivially "free".
        status, body = srv.post("/api/scan/start", {
            "coverage_threshold": 0, "tau": 0.8, "direction": "absence", "n_permutations": 49,
        })
        assert status == 200, body
        deadline = time.time() + 120
        while True:
            with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/api/scan/status", timeout=30) as r:
                job = json.loads(r.read())
            if job["status"] in ("done", "error", "cancelled"):
                break
            assert time.time() < deadline
            time.sleep(0.05)
        assert job["status"] == "done", job.get("error")
        assert job["direction"] == "absence"
        assert all(r["direction"] == "absence" for r in job["results"])
        skipped = {(s["column"], s["value"]) for s in job["skipped"]}
        assert ("target", "rare1") in skipped and ("target", "rare2") in skipped
        assert all("base rate" in s["reason"] for s in job["skipped"])
        assert job["n_skipped"] == len(job["skipped"])
        tested = {(r["column"], r["value"]) for r in job["results"]}
        assert ("target", "common") in tested
    finally:
        srv.close()


def test_export_full_dashboard_absence_mode():
    df = _df(6)
    html = vsf.export_full_dashboard(df, target="target", criterion="common", direction="absence")
    assert '"direction": "absence"' in html
    assert "\\u2260" in html or "≠" in html
    with pytest.raises(ValueError):
        vsf.export_full_dashboard(df, target="target", criterion=None, direction="absence")
