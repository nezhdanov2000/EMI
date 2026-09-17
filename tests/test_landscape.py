"""
Solution landscape (`vsf.avr.Landscape`, Section 4.9): every candidate the
exhaustive search scored, binned into the 10 x 10 (coverage, centres)
lattice, plus opening an explicitly chosen schema (`report_schema`,
`/api/analyze` with `features`).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from vsf.avr import (
    Landscape, _subset_codes, compute_landscape, compute_tau_curves, discover_branches,
    report_schema, tau_grid,
)
from vsf.centers import CenterSpec, coverage_score
from vsf.pmd import discretize_dataset
from vsf.server import _build_server


def _df(seed: int = 0, n: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.choice(["x", "y", "z"], size=n)
    b = rng.choice(["0", "1"], size=n)
    c = rng.choice(["p", "q", "r", "s"], size=n)
    d = rng.choice(["m", "n"], size=n)
    e = rng.choice(["u", "v", "w"], size=n)
    pos = ((a == "z") & (b == "1")) | ((c == "s") & (e == "w")) | (rng.random(n) < 0.04)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "target": np.where(pos, "yes", "no")})


def _reference_records(df, spec):
    """Every candidate scored the slow way: cell codes + coverage_score."""
    X = df[["a", "b", "c", "d", "e"]].values
    z = (df["target"] == "yes").astype(np.int8).values
    X_discrete, bin_counts = discretize_dataset(X)
    n_pos = int(z.sum())
    out = {}
    for d in range(1, 5):
        for combo in itertools.combinations(range(5), d):
            # The search's own partition (capacity coarsening included).
            codes, C = _subset_codes(X_discrete, bin_counts, X.shape[0], combo)
            cov, neg_k, neg_mass, _ = coverage_score(z, codes, C, spec)
            out[combo] = (cov, int(-neg_k), -neg_mass)
    return out, n_pos


# ---------------------------------------------------------------------------
# Recording and binning
# ---------------------------------------------------------------------------

def test_landscape_records_every_candidate_with_the_search_key_numbers():
    df = _df()
    spec = CenterSpec(tau=0.85)
    X = df[["a", "b", "c", "d", "e"]].values
    z = (df["target"] == "yes").astype(int).values
    L = compute_landscape(X, z, feature_names=list("abcde"), positive_class=1, center_spec=spec)
    ref, n_pos = _reference_records(df, spec)
    assert len(L) == len(ref) == 5 + 10 + 10 + 5
    seen = set()
    for i, combo in enumerate(L.features):
        cov, k, mass = ref[combo]
        assert L.k_sel[i] / n_pos == pytest.approx(cov, abs=0)
        assert L.n_centers[i] == k
        assert L.n_sel[i] / L.n_samples == pytest.approx(mass, abs=0)
        seen.add(combo)
    assert seen == set(ref)


@pytest.mark.parametrize(
    "fraction,expected",
    [(0.1, 0), (0.1000001, 1), (0.2, 1), (0.05, 0), (1.0, 9), (0.95, 9), (0.9, 8), (0.3, 2), (7 / 23, 3)],
)
def test_bin_index_is_left_open_right_closed(fraction, expected):
    assert int(Landscape.bin_index(np.array([fraction]), 10)[0]) == expected


def test_bins_exclude_zero_centre_schemas_and_count_them():
    df = _df(1)
    spec = CenterSpec(tau=0.9)
    X = df[["a", "b", "c", "d", "e"]].values
    z = (df["target"] == "yes").astype(int).values
    L = compute_landscape(X, z, feature_names=list("abcde"), positive_class=1, center_spec=spec)
    for d in (None, 1, 2, 3, 4):
        b = L.bins(d)
        family = L.d == d if d is not None else np.ones(len(L), dtype=bool)
        assert b["n_total"] == int(family.sum())
        assert b["n_zero"] == int((family & (L.n_centers == 0)).sum())
        assert sum(map(sum, b["counts"])) == b["n_total"] - b["n_zero"]
        assert b["k_max"] == (int(L.n_centers[family].max()) if family.any() else 0)
        sel = np.nonzero(family & (L.n_centers > 0))[0]
        if sel.size and b["k_max"]:
            ix = Landscape.bin_index(L.k_sel[sel] / L.n_positive, 10)
            iy = Landscape.bin_index(L.n_centers[sel] / b["k_max"], 10)
            recount = np.zeros((10, 10), dtype=int)
            np.add.at(recount, (iy, ix), 1)
            assert recount.tolist() == b["counts"]
            assert (L.k_sel[sel] > 0).all()  # K > 0 implies coverage > 0


def test_cell_lists_exactly_its_schemas_most_concentrated_first():
    df = _df(2)
    spec = CenterSpec(tau=0.85)
    X = df[["a", "b", "c", "d", "e"]].values
    z = (df["target"] == "yes").astype(int).values
    L = compute_landscape(X, z, feature_names=list("abcde"), positive_class=1, center_spec=spec)
    b = L.bins(None)
    total_listed = 0
    for iy in range(10):
        for ix in range(10):
            page = L.cell(None, ix, iy, limit=1000)
            assert page["total"] == b["counts"][iy][ix]
            total_listed += page["total"]
            masses = [s["mass"] for s in page["schemas"]]
            assert masses == sorted(masses)
            for sc in page["schemas"]:
                assert sc["n_centers"] > 0
                assert int(Landscape.bin_index(np.array([sc["coverage"]]), 10)[0]) == ix
                assert int(Landscape.bin_index(np.array([sc["n_centers"] / b["k_max"]]), 10)[0]) == iy
                assert sc["feature_names"] == [list("abcde")[j] for j in sc["features"]]
    assert total_listed == b["n_total"] - b["n_zero"]
    full = L.cell(None, 0, 0, limit=1000)["schemas"]
    if len(full) > 2:
        assert L.cell(None, 0, 0, limit=2, offset=1)["schemas"] == full[1:3]


def test_absence_landscape_uses_mass_on_x():
    df = _df(3)
    X = df[["a", "b", "c", "d", "e"]].values
    z = (df["target"] == "no").astype(int).values  # the common value
    L = compute_landscape(X, z, feature_names=list("abcde"), positive_class=1,
                          center_spec=CenterSpec(tau=0.95), direction="absence")
    assert L.bins(None)["x"] == "mass"
    np.testing.assert_allclose(L.x_values(), L.n_sel / L.n_samples)


# ---------------------------------------------------------------------------
# Opening a schema
# ---------------------------------------------------------------------------

def test_report_schema_equals_the_search_winner_for_the_winning_subset():
    df = _df(4)
    X = df[["a", "b", "c", "d", "e"]].values
    z = (df["target"] == "yes").astype(int).values
    found = discover_branches(X, z, feature_names=list("abcde"), positive_class=1,
                              n_permutations_centers=0, cv_repeats=3)
    for d, br in found.items():
        opened = report_schema(X, z, br.selected_features, feature_names=list("abcde"),
                               positive_class=1, cv_repeats=3)
        assert set(opened) == {d}
        assert dataclasses.asdict(opened[d]) == dataclasses.asdict(br)
    opened = report_schema(X, z, [4, 2], feature_names=list("abcde"), positive_class=1, cv_repeats=0)[2]
    assert opened.selected_features == [4, 2]
    assert opened.centers.coverage_p_value is None  # never uncorrected, by design


@pytest.mark.parametrize("bad", [[], [0, 0], [0, 1, 2, 3, 4], [7], [-1]])
def test_report_schema_rejects_bad_feature_lists(bad):
    df = _df(5)
    X = df[["a", "b", "c", "d", "e"]].values
    z = (df["target"] == "yes").astype(int).values
    with pytest.raises(ValueError):
        report_schema(X, z, bad, positive_class=1, cv_repeats=0)


# ---------------------------------------------------------------------------
# Server
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


def test_landscape_endpoints_and_opening_a_schema():
    df = _df(6)
    srv = _Server(df)
    base = {"target": "target", "criterion": "yes", "tau": 0.85}
    try:
        status, analysis = srv.post("/api/analyze", base)
        assert status == 200
        status, bins = srv.post("/api/landscape", dict(base, d=None))
        assert status == 200, bins
        assert bins["n_candidates"] == 30 and bins["n_bins"] == 10
        assert bins["feature_names"] == ["a", "b", "c", "d", "e"]
        assert sum(map(sum, bins["counts"])) == bins["n_total"] - bins["n_zero"]
        status, bins2 = srv.post("/api/landscape", dict(base, d=2))
        assert status == 200 and bins2["n_total"] == 10
        assert len(srv.httpd.landscape_cache) == 1  # one landscape serves every d
        for d_str, br in analysis["branches"].items():
            cc = br["search_centers"]  # the search partition, which the landscape is made of
            if cc["n_centers"] == 0:
                continue
            status, bd = srv.post("/api/landscape", dict(base, d=int(d_str)))
            ix = int(Landscape.bin_index(np.array([cc["coverage"]]), 10)[0])
            iy = int(Landscape.bin_index(np.array([cc["n_centers"] / bd["k_max"]]), 10)[0])
            assert bd["counts"][iy][ix] >= 1
            status, cell = srv.post("/api/landscape/cell", dict(base, d=int(d_str), ix=ix, iy=iy, limit=1000))
            assert status == 200
            listed = {tuple(sorted(s["features"])) for s in cell["schemas"]}
            assert tuple(sorted(br["selected_feature_indices"])) in listed
        cells = [(iy, ix) for iy in range(10) for ix in range(10) if bins["counts"][iy][ix]]
        iy, ix = cells[0]
        status, cell = srv.post("/api/landscape/cell", dict(base, d=None, ix=ix, iy=iy, limit=5))
        feats = cell["schemas"][0]["features"]
        status, opened = srv.post("/api/analyze", dict(base, features=feats))
        assert status == 200, opened
        assert opened["schema"] == {
            "features": feats, "feature_names": [["a", "b", "c", "d", "e"][j] for j in feats],
            "selected_from_landscape": True,
        }
        assert opened["branch_dims"] == [len(feats)]
        b = opened["branches"][str(len(feats))]
        assert b["search_centers"]["coverage_p_value"] is None
        assert b["search_centers"]["coverage"] == pytest.approx(cell["schemas"][0]["coverage"])
        assert b["selected_feature_indices"] == feats
        assert len(srv.httpd.analyze_cache) == 2
        assert srv.post("/api/analyze", dict(base, features=[0, 0]))[0] == 400
        assert srv.post("/api/landscape/cell", dict(base, ix=10, iy=0))[0] == 400
        assert srv.post("/api/landscape", dict(base, d=9))[0] == 400
        assert srv.post("/api/landscape", dict(base, tau=0.01))[0] == 400  # below base rate
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# Tau-curves: the landscape's envelope over the purity floor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anchor, step, first, last, count", [
    (0.0, 1.0, 0.01, 1.0, 100),
    (0.383, 1.0, 0.39, 1.0, 62),
    (0.39, 1.0, 0.40, 1.0, 61),      # a grid point IS the anchor: strictly above it
    (0.5, 5.0, 0.55, 1.0, 10),
    (1.0, 1.0, None, None, 0),
])
def test_tau_grid_is_whole_steps_strictly_above_the_anchor(anchor, step, first, last, count):
    g = tau_grid(anchor, step)
    assert g.shape == (count,)
    if count:
        assert g[0] == pytest.approx(first) and g[-1] == pytest.approx(last)
        assert np.allclose(np.diff(g), step / 100.0)
        assert (g > anchor).all()


@pytest.mark.parametrize("bad", [(-0.1, 1.0), (1.1, 1.0), (0.5, 0.0), (0.5, 60.0)])
def test_tau_grid_rejects_bad_arguments(bad):
    with pytest.raises(ValueError):
        tau_grid(*bad)


@pytest.mark.parametrize("rule", ["purity", "certified"])
@pytest.mark.parametrize("direction", ["presence", "absence"])
def test_tau_curves_equal_the_search_at_every_grid_point(rule, direction):
    """
    At each grid tau and each d the curve's point is exactly the winner
    `discover_branches` reports for that d at that tau on the search
    partition (coverage, centres, mass, features) - the curve is the search
    evaluated on the grid, not an approximation of it.
    """
    df = _df(3, n=500)
    X = df[["a", "b", "c", "d", "e"]].values
    Z = (df["target"] == "yes").astype(int).values
    names = ["a", "b", "c", "d", "e"]
    out = compute_tau_curves(
        X, Z, names, max_d=4, positive_class=1,
        center_spec=CenterSpec(tau=0.5, rule=rule, alpha=0.05, min_samples=2),
        direction=direction, step_pct=5.0,
    )
    taus = out["taus"]
    assert out["x"] == ("mass" if direction == "absence" else "coverage")
    assert len(taus) >= 5 and all(t > out["anchor"] for t in taus)
    if rule == "certified":
        assert taus[-1] < 1.0
    for ti in [0, len(taus) // 2, len(taus) - 1]:
        tau = taus[ti]
        branches = discover_branches(
            X, Z, names, max_d=4, positive_class=1,
            center_spec=CenterSpec(tau=tau, rule=rule, alpha=0.05, min_samples=2),
            direction=direction, random_state=0, n_permutations_centers=0,
            n_permutations_familywise_coverage=0, cv_splits=2, cv_repeats=1,
        )
        for d, br in branches.items():
            c = out["curves"][str(d)]
            sc = br.centers  # the search partition's statistics (vis derives search_centers from it)
            assert c["coverage"][ti] == pytest.approx(sc.coverage)
            assert c["n_centers"][ti] == sc.n_centers
            assert c["mass"][ti] == pytest.approx(sc.mass)
            if sc.n_centers > 0:
                assert c["features"][ti] == list(br.selected_features)
                assert c["feature_names"][ti] == br.selected_feature_names
    for d in range(1, 5):
        c = out["curves"][str(d)]
        assert c["n_family"] == len(list(itertools.combinations(range(5), d)))
        assert all(a >= b for a, b in zip(c["x"], c["x"][1:]))   # envelope non-increasing in tau
        assert all(0 <= v <= c["n_family"] for v in c["n_certifying"])
        assert all(k == 0 or f for k, f in zip(c["n_centers"], c["features"]))


def test_tau_curves_ignore_the_spec_tau_and_reject_bad_max_d():
    df = _df(4, n=300)
    X = df[["a", "b", "c", "d", "e"]].values
    Z = (df["target"] == "yes").astype(int).values
    a = compute_tau_curves(X, Z, max_d=2, positive_class=1, center_spec=CenterSpec(tau=0.6), step_pct=10.0)
    b = compute_tau_curves(X, Z, max_d=2, positive_class=1, center_spec=CenterSpec(tau=0.95), step_pct=10.0)
    assert a == b
    with pytest.raises(ValueError):
        compute_tau_curves(X, Z, max_d=0, positive_class=1)


def test_by_x_category_lists_the_schemas_at_a_coverage_category():
    df = _df(5)
    X = df[["a", "b", "c", "d", "e"]].values
    Z = (df["target"] == "yes").astype(int).values
    spec = CenterSpec(tau=0.8)
    ls = compute_landscape(X, Z, ["a", "b", "c", "d", "e"], max_d=4, positive_class=1, center_spec=spec)
    ref, n_pos = _reference_records(df, spec)
    for d in (None, 2, 4):
        seen = 0
        for ix in range(10):
            out = ls.by_x_category(d, ix, limit=1000)
            seen += out["total"]
            xs = [s["coverage"] for s in out["schemas"]]
            assert xs == sorted(xs, reverse=True)
            for s in out["schemas"]:
                assert s["n_centers"] > 0
                assert int(Landscape.bin_index(np.array([s["coverage"]]), 10)[0]) == ix
                assert d is None or s["d"] == d
                cov, k, mass = ref[tuple(s["features"])]
                assert s["coverage"] == pytest.approx(cov) and s["n_centers"] == k
            # the union over categories is every certifying schema of the family
        expected = sum(1 for combo, (cov, k, _) in ref.items() if k > 0 and (d is None or len(combo) == d))
        assert seen == expected
    page = ls.by_x_category(None, 9, limit=2, offset=1)
    full = ls.by_x_category(None, 9, limit=1000)
    assert page["schemas"] == full["schemas"][1:3]


def test_tau_curve_endpoints():
    df = _df(7)
    srv = _Server(df)
    base = {"target": "target", "criterion": "yes", "tau": 0.85}
    try:
        status, curves = srv.post("/api/landscape/curves", base)
        assert status == 200, curves
        assert curves["direction"] == "presence" and curves["rule"] == "purity" and curves["x"] == "coverage"
        assert curves["step_pct"] == 1.0 and curves["taus"][-1] == pytest.approx(1.0)
        assert set(curves["curves"]) == {"1", "2", "3", "4"}
        # tau is not part of the key: another tau is the same cached object
        status, curves2 = srv.post("/api/landscape/curves", dict(base, tau=0.95))
        assert status == 200 and curves2["taus"] == curves["taus"] and len(srv.httpd.curves_cache) == 1
        assert srv.post("/api/landscape/curves", dict(base, min_samples=3))[0] == 200
        assert len(srv.httpd.curves_cache) == 2
        # a curve point's click-through: the schemas of d at that tau in the point's category
        c = curves["curves"]["2"]
        ti = next(i for i, v in enumerate(c["x"]) if v > 0)
        tau = curves["taus"][ti]
        ix = int(Landscape.bin_index(np.array([c["x"][ti]]), 10)[0])
        status, at = srv.post("/api/landscape/at", dict(base, tau=tau, d=2, ix=ix, limit=1000))
        assert status == 200, at
        assert at["tau"] == pytest.approx(tau) and at["d"] == 2 and at["ix"] == ix
        top = at["schemas"][0]
        assert top["coverage"] == pytest.approx(c["x"][ti])
        assert top["features"] == c["features"][ti] and top["n_centers"] == c["n_centers"][ti]
        assert all(s["d"] == 2 for s in at["schemas"])
        # the landscape at that tau is what served the listing
        assert any(k[0] == ("alpha", "0.05") for k in srv.httpd.landscape_cache)
        assert srv.post("/api/landscape/at", dict(base, tau=tau, ix=ix))[0] == 400        # d required
        assert srv.post("/api/landscape/at", dict(base, tau=tau, d=2, ix=10))[0] == 400   # bad ix
        assert srv.post("/api/landscape/curves", dict(base, direction="absence", criterion=None))[0] == 400
        status, absent = srv.post("/api/landscape/curves", dict(base, direction="absence", tau=0.95))
        assert status == 200 and absent["x"] == "mass" and absent["anchor"] == pytest.approx(1 - curves["anchor"])
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# Cost-coverage frontier (Section 4.9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cost", ["centers", "conditions"])
@pytest.mark.parametrize("direction", ["presence", "absence"])
def test_frontier_is_the_pareto_staircase_of_the_landscape(cost, direction):
    """
    Every step must be Pareto-optimal for its dimensionality — no scored
    schema of that d reaches at least as much x at no greater cost — the
    staircase must be strictly increasing on both axes, and the last step
    must carry the family's best x, the number the branch list reports.
    """
    df = _df(5)
    X = df.drop(columns=["target"]).values
    z = (df["target"].values == "yes").astype(int)
    names = list(df.drop(columns=["target"]).columns)
    spec = CenterSpec(tau=0.8)
    L = compute_landscape(X, z, names, positive_class=1, center_spec=spec, direction=direction)
    out = L.frontier(cost)
    assert out["cost"] == cost
    assert out["x"] == ("mass" if direction == "absence" else "coverage")
    x = L.x_values()
    c = L.n_centers if cost == "centers" else L.n_centers * L.d
    seen = 0
    for d_str, blk in out["dims"].items():
        d = int(d_str)
        fam = (L.d == d) & (L.n_centers > 0)
        assert blk["n_family"] == int(np.count_nonzero(L.d == d))
        assert blk["n_certifying"] == int(np.count_nonzero(fam))
        steps = blk["steps"]
        seen += len(steps)
        if not np.any(fam):
            assert steps == []
            continue
        assert steps, "a certifying dimensionality must have at least one step"
        costs = [s["cost"] for s in steps]
        covs = [s["coverage"] if direction == "presence" else s["mass"] for s in steps]
        assert costs == sorted(costs) and len(set(costs)) == len(costs)
        assert all(covs[i] < covs[i + 1] for i in range(len(covs) - 1))
        assert max(covs) == pytest.approx(float(x[fam].max()))
        for s in steps:
            i = L.features.index(tuple(s["features"]))
            assert int(c[i]) == s["cost"]
            assert s["conditions"] == s["n_centers"] * d
            # Pareto: no schema of this d dominates the step — none is at
            # least as good on both axes and strictly better on one
            others = fam.copy()
            others[i] = False
            dominates = ((x[others] >= x[i] - 1e-12) & (c[others] < c[i])) | (
                (x[others] > x[i] + 1e-12) & (c[others] <= c[i]))
            assert not np.any(dominates)
            # and the step is the cheapest schema reaching at least this x
            assert int(c[i]) == int(c[fam & (x >= x[i] - 1e-12)].min())
    assert seen > 0
    with pytest.raises(ValueError):
        L.frontier("rows")


def test_frontier_endpoint():
    df = _df(6)
    srv = _Server(df)
    base = {"target": "target", "criterion": "yes", "tau": 0.8}
    try:
        status, out = srv.post("/api/landscape/frontier", base)
        assert status == 200, out
        assert out["cost"] == "centers" and out["x"] == "coverage"
        assert out["n_candidates"] == 30 and out["tau"] == 0.8
        status, cond = srv.post("/api/landscape/frontier", dict(base, cost="conditions"))
        assert status == 200
        # the same schemas, charged d times as much
        for d_str in out["dims"]:
            a = {tuple(s["features"]): s for s in out["dims"][d_str]["steps"]}
            b = {tuple(s["features"]): s for s in cond["dims"][d_str]["steps"]}
            for f, s in b.items():
                assert s["cost"] == s["n_centers"] * int(d_str)
            assert {round(s["coverage"], 12) for s in a.values()} == {round(s["coverage"], 12) for s in b.values()}
        assert len(srv.httpd.landscape_cache) == 1  # served from the cached landscape
        assert srv.post("/api/landscape/frontier", dict(base, cost="rows"))[0] == 400
        assert srv.post("/api/landscape/frontier", dict(base, tau=0.01))[0] == 400
    finally:
        srv.close()


def test_tau_curves_exclude_schemas_from_the_family():
    """
    `exclude` leaves schemas out of the family: the envelope without a
    schema is pointwise <= the envelope with it, equals it wherever that
    schema was not the best, and never names an excluded schema.
    """
    df = _df(8, n=500)
    X = df[["a", "b", "c", "d", "e"]].values
    Z = (df["target"] == "yes").astype(int).values
    names = ["a", "b", "c", "d", "e"]
    full = compute_tau_curves(X, Z, names, max_d=3, positive_class=1, center_spec=CenterSpec(tau=0.5), step_pct=5.0)
    c2 = full["curves"]["2"]
    ti = next(i for i, v in enumerate(c2["x"]) if v > 0)
    schema = c2["features"][ti]
    part = compute_tau_curves(X, Z, names, max_d=3, positive_class=1, center_spec=CenterSpec(tau=0.5), step_pct=5.0,
                              exclude=[schema, [0]])
    assert part["n_excluded"] == 2 and full["n_excluded"] == 0
    assert part["taus"] == full["taus"]
    for d in ("1", "2", "3"):
        for i in range(len(full["taus"])):
            assert part["curves"][d]["x"][i] <= full["curves"][d]["x"][i] + 1e-12
            if sorted(full["curves"][d]["features"][i]) not in (sorted(schema), [0]):
                assert part["curves"][d]["x"][i] == pytest.approx(full["curves"][d]["x"][i])
            assert sorted(part["curves"][d]["features"][i]) not in (sorted(schema), [0])
    assert part["curves"]["2"]["n_family"] == full["curves"]["2"]["n_family"]   # counted, not scored
    # endpoint: exclude is part of the cache key and echoed back
    srv = _Server(df)
    base = {"target": "target", "criterion": "yes", "tau": 0.6}
    try:
        status, a = srv.post("/api/landscape/curves", base)
        status, b = srv.post("/api/landscape/curves", dict(base, exclude=[schema]))
        assert status == 200 and b["exclude"] == [sorted(schema)] and b["n_excluded"] == 1
        assert a["exclude"] == [] and len(srv.httpd.curves_cache) == 2
        assert srv.post("/api/landscape/curves", dict(base, exclude=[[0, 0]]))[0] == 400
        assert srv.post("/api/landscape/curves", dict(base, exclude="x"))[0] == 400
    finally:
        srv.close()
