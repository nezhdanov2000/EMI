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

from vsf.avr import Landscape, _subset_codes, compute_landscape, discover_branches, report_schema
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
