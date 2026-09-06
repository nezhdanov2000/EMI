"""
Regression tests for vsf.vis.

Covers two long-standing bugs from the original code review:
  1. Axis fallback selection (`prepare_visualization_payload`): when the
     branch selects fewer than 3 features (d < 3), the Y/Z scatter axes
     used to fall back to the raw literal column indices 1 and 2 regardless
     of what the primary axis already was -- silently duplicating an axis
     whenever the single selected feature happened to be column 1 or 2.
  2. Multi-class purity (`build_grid`'s `pur` computation): the old formula
     `mean(class_index) / (K-1)` is the cell's average class index
     normalized to [0,1], which only coincides with "share of the positive
     class" when K=2. For K>2 it can report a large "purity" for a cell
     containing ZERO samples of the actual positive/last class.

Also covers the library/example-vocabulary decoupling: `vsf.vis` carries NO
dataset vocabulary of its own -- `humanize_val`/`humanize_col` and
`prepare_visualization_payload` accept an optional `translations` table
(see `vsf.vis.Translations`) and fall back to raw column/value strings with
none supplied.

v2.3 note ("Coverage Only"): `BranchResult.centers` is now a REQUIRED
`vsf.centers.CenterReport`, not `Optional` -- there is no longer a
"branch with no centres" case to construct, because `discover_branches`
cannot produce a `BranchResult` without a positive class and a certificate.
`BranchResult` no longer carries `mi`/`mi_null`/`mi_adj`/`u_adj`/`h_target`/
`p_value`/`p_value_familywise`/`per_class`, and the payload's `metrics` dict
is now just `{"d": branch.d}` -- see `vsf.avr`'s and `vsf.vis`'s module
docstrings. `view_metrics`'s per-collapsed-view breakdown
(`coverage_by_d`/`n_centers_by_d`/`purity_by_d`/`max_purity_lower_by_d`) is
computed directly from the DISPLAYED partition (`cell_stats`), never from
`BranchResult`'s own prefix series -- so it is fully populated even for a
hand-built branch that carries no prefix series at all. The payload no
longer carries a `class_breakdown` key (it read `BranchResult.per_class`,
which is gone).
"""

import numpy as np
import pytest

from vsf.avr import BranchResult, discover_branches
from vsf.centers import CenterSpec, center_report
from vsf.metrics import cell_codes as _cell_codes
from vsf.vis import _axis_fallback_indices, humanize_col, humanize_val, prepare_visualization_payload


def _make_branch(X, Z, selected_features, d=None, spec=None, z_binary=None,
                  cv_repeats=0, n_permutations=0):
    """
    Hand-builds a `BranchResult` the way a test needs to (not via
    `discover_branches`), but with a REAL `vsf.centers.CenterReport` for
    `centers` -- required since v2.3, and cheap to compute directly on the
    raw (undiscretized) `X` columns the payload itself will use.

    `z_binary` defaults to an indicator of `Z`'s highest `np.unique` code,
    matching `prepare_visualization_payload`'s own default `positive_value`
    resolution, so the two stay consistent unless a test overrides it.
    Prefix-series fields (`coverage_by_prefix_d`, ...) are left at their
    empty-list default: this helper is for tests that want a branch NOT
    produced by `discover_branches`, where no per-view breakdown exists.
    """
    combo = list(selected_features)
    codes, n_cells = _cell_codes(np.asarray(X)[:, combo])
    if z_binary is None:
        z_arr = np.asarray(Z).ravel()
        z_binary = (z_arr == np.unique(z_arr)[-1]).astype(np.int8)
    centers = center_report(
        np.asarray(z_binary).astype(np.int8), codes, n_cells,
        spec if spec is not None else CenterSpec(),
        n_repeats=cv_repeats, n_permutations=n_permutations,
    )
    return BranchResult(
        d=d if d is not None else len(combo),
        selected_features=combo,
        selected_feature_names=[f"f{j}" for j in combo],
        centers=centers,
    )


def test_axis_fallback_never_duplicates_when_selected_feature_is_1_or_2():
    # The exact old-bug scenario: selected_idx = [1] used to yield
    # y_col_idx = 1 too (since the old fallback for y was the literal `1`),
    # silently collapsing the Y axis onto the X axis.
    assert _axis_fallback_indices([1], n_features=10, count=3) == [1, 0, 2]
    assert _axis_fallback_indices([2], n_features=10, count=3) == [2, 0, 1]
    # No duplicates for any single selected index in range.
    for sel in range(10):
        idxs = _axis_fallback_indices([sel], n_features=10, count=3)
        assert len(set(idxs)) == 3, f"duplicate axis indices for selected={sel}: {idxs}"


def test_axis_fallback_preserves_selected_order_and_pads_unused():
    assert _axis_fallback_indices([5], n_features=10, count=3) == [5, 0, 1]
    assert _axis_fallback_indices([5, 0], n_features=10, count=3) == [5, 0, 1]
    assert _axis_fallback_indices([2, 7], n_features=10, count=3) == [2, 7, 0]
    # Already >= count selected features: only the first `count` are used.
    assert _axis_fallback_indices([4, 5, 6, 7], n_features=10, count=3) == [4, 5, 6]


def test_axis_fallback_handles_degenerate_low_feature_count():
    # Fewer feature columns than axes requested must not crash; it repeats
    # the last valid index rather than raising.
    idxs = _axis_fallback_indices([0], n_features=1, count=3)
    assert len(idxs) == 3
    assert idxs[0] == 0


def test_prepare_visualization_payload_no_duplicate_axes_when_d_is_1():
    rng = np.random.default_rng(0)
    n = 200
    n_features = 5
    X = rng.integers(0, 4, size=(n, n_features))
    Z = rng.integers(0, 2, size=n)
    feature_names = [f"feat_{i}" for i in range(n_features)]

    # Selected feature is literally column 1 -- the exact case that used to
    # duplicate the Y axis onto X.
    branch = _make_branch(X, Z, [1])
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=feature_names, target_name="class", sort_Z=Z
    )
    names = [payload["axis_names"]["x"], payload["axis_names"]["y"], payload["axis_names"]["z"]]
    assert len(set(names)) == 3, f"expected 3 distinct axes, got {names}"


def test_build_grid_purity_is_share_of_positive_class_for_binary_target():
    # K=2: purity must exactly equal the fraction of the last-index class
    # in each cell (this must be unchanged from the pre-fix formula, since
    # for K=2 the old formula was already correct).
    rng = np.random.default_rng(1)
    n = 300
    X = rng.integers(0, 3, size=(n, 3))
    Z = rng.integers(0, 2, size=n)
    branch = _make_branch(X, Z, [0, 1, 2])
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["a", "b", "c"], target_name="class", sort_Z=Z
    )
    for pur in payload["grid_purity"]:
        assert 0.0 <= pur <= 1.0


def test_build_grid_purity_correct_for_multiclass_middle_class_cell():
    # Regression test for the K>2 purity bug: construct a cell that is 100%
    # the MIDDLE class (index 1 of 3), which the old formula
    # (`mean(class_idx) / (K-1)`) scored as 0.5 ("50% positive") despite
    # containing ZERO samples of the actual positive/last class (index 2).
    # The correct value is 0.0 (share of the LAST class in this cell).
    rng = np.random.default_rng(2)
    n = 600
    X = rng.integers(0, 5, size=(n, 3))
    special_val = 0
    mask = X[:, 0] == special_val
    # Force every sample with X[:,0] == special_val into class-index 1
    # (the middle of 3 classes), everything else random across all 3.
    Z = rng.integers(0, 3, size=n)
    Z[mask] = 1

    # `_make_branch`'s CenterReport is unrelated to this test's assertions
    # (grid purity is computed on the raw Z/positive_value the payload call
    # is given, not on `branch.centers`), so a plain default z_binary
    # (indicator of the highest class code) is fine here.
    branch = _make_branch(X, Z, [0, 1, 2])
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["a", "b", "c"], target_name="class", sort_Z=Z
    )

    g1 = payload["grids"]["1"]
    found = False
    for cd, pur in zip(g1["customdata"], g1["purity"]):
        if cd["coords"]["a"] == special_val:
            found = True
            assert pur == pytest.approx(0.0, abs=1e-9), (
                f"expected 0.0 share-of-positive-class for a cell that is "
                f"100% the middle class, got {pur}"
            )
    assert found, "expected to find the constructed cell in the 1D grid"


def test_humanize_functions_raw_passthrough_with_no_translations():
    """
    With no `translations` supplied (the default), `vsf.vis` must have zero
    embedded dataset vocabulary: every label -- from the bare helpers up
    through the full `prepare_visualization_payload` output -- falls back to
    the raw column/value strings exactly as they appear in the input data.
    """
    assert humanize_val("odor", "f") == "f"
    assert humanize_val("odor", "f", None) == "f"
    assert humanize_val("odor", "f", {}) == "f"
    assert humanize_col("odor") == "odor"
    assert humanize_col("odor", None) == "odor"

    rng = np.random.default_rng(3)
    n = 200
    X = rng.integers(0, 4, size=(n, 3))
    Z = rng.integers(0, 2, size=n)
    branch = _make_branch(X, Z, [0, 1, 2])
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["cap-shape", "odor", "habitat"],
        target_name="class", sort_Z=Z,
    )
    assert payload["axis_names"] == {"x": "cap-shape", "y": "odor", "z": "habitat"}
    assert payload["target_name"] == "class"
    assert payload["all_feature_names"] == ["cap-shape", "odor", "habitat"]
    # Raw integer-coded target values pass through as their str() form.
    assert set(payload["target_labels"]) <= {"0", "1"}
    assert set(payload["unique_target_classes"]) <= {"0", "1"}


def test_humanize_functions_translations_override_end_to_end():
    """
    A caller-supplied `translations` table -- shaped like
    `examples.mushroom_demo.MUSHROOM_TRANSLATIONS` -- must be threaded all
    the way through `prepare_visualization_payload`'s human-readable
    fields. This is the contract `vsf.server`/`vsf.dashboard` rely on to
    keep showing dataset-specific labels after the vocabulary moved out of
    the library.
    """
    translations = {
        "columns": {"odor": "Odor", "cap-shape": "Cap Shape"},
        "values": {"odor": {"f": "foul", "n": "none"}},
    }
    assert humanize_val("odor", "f", translations) == "foul"
    assert humanize_col("odor", translations) == "Odor (odor)"
    # A column/value absent from the table falls back to the raw string --
    # a partial translation table must not raise or blank out the rest.
    assert humanize_val("habitat", "g", translations) == "g"
    assert humanize_col("habitat", translations) == "habitat"

    rng = np.random.default_rng(4)
    n = 200
    X = np.column_stack([
        rng.choice(["f", "n"], size=n),
        rng.integers(0, 4, size=n),
        rng.integers(0, 4, size=n),
    ])
    Z = rng.choice(["f", "n"], size=n)
    branch = _make_branch(X, Z, [0, 1, 2])
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["odor", "cap-shape", "habitat"],
        target_name="odor", sort_Z=Z, translations=translations,
    )
    assert payload["axis_names"]["x"] == "Odor (odor)"
    assert payload["axis_names"]["y"] == "Cap Shape (cap-shape)"
    # habitat has no column-name entry in this (deliberately partial) table.
    assert payload["axis_names"]["z"] == "habitat"
    assert set(payload["target_labels"]) <= {"foul", "none"}
    assert set(payload["unique_target_classes"]) <= {"foul", "none"}


# ---------------------------------------------------------------------------
# v2.3 "Coverage Only" `metrics` shape -- no scenario/vir/history, no MI
# ---------------------------------------------------------------------------

def test_metrics_shape_is_the_v23_coverage_only_set():
    rng = np.random.default_rng(5)
    n = 200
    X = rng.integers(0, 3, size=(n, 3))
    Z = rng.integers(0, 2, size=n)
    branch = _make_branch(X, Z, [0, 1], d=2)
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["a", "b", "c"], target_name="class", sort_Z=Z
    )
    assert payload["metrics"] == {"d": 2}
    # No leftover v1.0 fields, and no MI-era fields (mi/mi_adj/u_adj/p_value/
    # per_class): a consumer must not be able to reach anything v2.1-v2.3
    # removed by reading the payload.
    banned = (
        "scenario", "vir", "l_target", "l_feat", "xai_message", "history",
        "d_star", "nmi", "mi", "mi_null", "mi_adj", "u_adj", "h_target",
        "p_value", "p_value_familywise", "significant", "per_class",
    )
    for name in banned:
        assert name not in payload
        assert name not in payload["metrics"]
    assert "class_breakdown" not in payload
    for key in ("mi_by_d", "mi_adj_by_d", "u_adj_by_d", "nmi_by_d"):
        assert key not in payload["view_metrics"]


# ---------------------------------------------------------------------------
# `view_metrics` -- per-collapsed-view centre statistics, computed from the
# DISPLAYED partition (never from `BranchResult`'s own prefix series).
# ---------------------------------------------------------------------------

def test_view_metrics_is_fully_populated_for_a_hand_built_branch_with_no_prefix_series():
    # A `_make_branch`-style BranchResult (constructed directly, not via
    # `discover_branches`) carries no per-prefix series
    # (`coverage_by_prefix_d` etc. are empty). Unlike the v2.1/v2.2 MI-era
    # payload, v2.3's `view_metrics` does not read those series at all --
    # it is computed fresh from `cell_stats` (the actual displayed
    # partition) every time, so it must be fully populated regardless.
    rng = np.random.default_rng(5)
    n = 200
    X = rng.integers(0, 3, size=(n, 3))
    Z = rng.integers(0, 2, size=n)
    branch = _make_branch(X, Z, [0, 1], d=2)
    assert branch.coverage_by_prefix_d == []  # the precondition this test is about
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["a", "b", "c"], target_name="class", sort_Z=Z
    )
    for key in ("coverage_by_d", "n_centers_by_d", "purity_by_d", "max_purity_lower_by_d"):
        assert set(payload["view_metrics"][key]) == {"1", "2", "3"}
    # `centers` is required since v2.3 -- a hand-built branch's own
    # CenterReport must still surface as `search_centers`, never None.
    assert payload["search_centers"] is not None
    assert payload["search_centers"]["coverage"] == pytest.approx(branch.centers.coverage)


def test_view_metrics_carries_the_full_per_view_breakdown_from_discover_branches():
    # For a branch actually produced by `discover_branches`, `view_metrics`
    # must expose an entry for every view dimensionality 1..3 (the payload
    # always builds 1D/2D/3D grids), and the full-d entry must match the
    # headline `centers` block exactly (both come from the same displayed
    # partition). A collapsed (1D) view's coverage must be a genuinely
    # different, LOWER number than the full 3D branch's whenever the extra
    # axes carry real information -- otherwise this test would not catch a
    # regression back to always showing the full-branch value.
    rng = np.random.default_rng(19)
    n = 3000
    X = rng.integers(0, 3, size=(n, 4))
    # Deterministic AND of three conditions: only the full 3-axis join
    # isolates a pure cell: 1D/2D marginals cannot certify anything at the
    # default tau=0.90.
    Z = ((X[:, 0] == 0) & (X[:, 1] == 1) & (X[:, 2] == 2)).astype(int)
    branches = discover_branches(
        X, Z, feature_names=["a", "b", "c", "d"], max_d=3,
        n_permutations_centers=0, cv_repeats=0, positive_class=1,
    )
    branch = branches[3]
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["a", "b", "c", "d"],
        target_name="t", target_is_indicator=True, positive_value=1,
    )
    vm = payload["view_metrics"]
    assert set(vm["coverage_by_d"].keys()) == {"1", "2", "3"}
    assert set(vm["n_centers_by_d"].keys()) == {"1", "2", "3"}
    assert set(vm["purity_by_d"].keys()) == {"1", "2", "3"}
    assert vm["coverage_by_d"]["3"] == pytest.approx(payload["centers"]["coverage"])
    assert vm["n_centers_by_d"]["3"] == payload["centers"]["n_centers"]
    assert vm["purity_by_d"]["3"] == pytest.approx(payload["centers"]["purity_pooled"])
    # The full join certifies (coverage 1.0); the 1D marginal on its own
    # axis certifies nothing at all -- a real, large difference.
    assert vm["coverage_by_d"]["3"] == pytest.approx(1.0)
    assert vm["coverage_by_d"]["1"] == 0.0
    assert vm["coverage_by_d"]["1"] != pytest.approx(vm["coverage_by_d"]["3"])


def test_prepare_visualization_payload_handles_4d_branch_slice_axis():
    rng = np.random.default_rng(6)
    n = 400
    X = rng.integers(0, 3, size=(n, 4))
    Z = rng.integers(0, 2, size=n)
    branch = _make_branch(X, Z, [0, 1, 2, 3], d=4)
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["a", "b", "c", "w"], target_name="class", sort_Z=Z
    )
    assert payload["slice_axis"] is not None
    assert payload["slice_axis"]["name"] == "w"
    assert payload["metrics"] == {"d": 4}
    # Per-slice grids plus the "all" marginal must be present.
    for sv_idx in range(len(payload["slice_axis"]["ticks"])):
        assert f"4_{sv_idx}" in payload["grids"]
    assert "4_all" in payload["grids"]


def test_generate_interactive_html_no_longer_exists():
    # Confirmed dead/no-callers and deleted outright (see vsf/vis.py's
    # module docstring). Nothing left to test its behavior -- this test
    # only guards against silent reintroduction.
    import vsf.vis as vis_mod
    assert not hasattr(vis_mod, "generate_interactive_html")


# ---------------------------------------------------------------------------
# v2.2: cell statistics are computed on ALL rows, not the render subsample
# ---------------------------------------------------------------------------
def test_cell_statistics_use_every_row_while_only_the_scatter_is_subsampled():
    # REGRESSION WITNESS. Before v2.2, `total_samples` held the RENDERED
    # count while `vsf.avr` computed every metric on all rows, and the
    # per-cell purities drawn on screen were computed on the subsample. The
    # two Ns must be reported separately and the cell counts must sum to N.
    rng = np.random.default_rng(19)
    n = 3000
    X = rng.integers(0, 3, size=(n, 3))
    Z = (X[:, 0] == 0).astype(int)
    branches = discover_branches(
        X, Z, feature_names=["a", "b", "c"],
        n_permutations_centers=0, cv_repeats=0, positive_class=1,
    )
    payload = prepare_visualization_payload(
        branches[2], X, Z, feature_names=["a", "b", "c"],
        max_display_samples=500, target_name="t", target_is_indicator=True,
        positive_value=1,
    )
    assert payload["total_samples"] == n
    assert payload["rendered_samples"] == 500
    grid = payload["grids"]["2"]
    assert sum(grid["sizes"]) == n
    assert sum(grid["positives"]) == int(Z.sum())
    # Every drawn cell carries a bound and a certification decision.
    assert len(grid["purity_lower"]) == len(grid["purity"]) == len(grid["certified"])
    assert all(lo <= p for lo, p in zip(grid["purity_lower"], grid["purity"]))


def test_certificate_block_and_center_summary_are_present_and_consistent():
    rng = np.random.default_rng(21)
    n = 4000
    X = rng.integers(0, 4, size=(n, 3))
    Z = (X[:, 0] == 0).astype(int)
    branches = discover_branches(
        X, Z, feature_names=["a", "b", "c"],
        n_permutations_centers=0, cv_repeats=0, positive_class=1,
    )
    payload = prepare_visualization_payload(
        branches[1], X, Z, feature_names=["a", "b", "c"],
        target_name="t", target_is_indicator=True, positive_value=1,
    )
    cert = payload["certificate"]
    assert cert["tau"] == pytest.approx(0.90)
    assert cert["method"] == "clopper-pearson"
    assert cert["multiplicity"] == "bonferroni"
    assert cert["alpha_effective"] < cert["alpha"]
    centers = payload["centers"]
    # A deterministic 1-D relation: exactly one cell is the target value.
    assert centers["n_centers"] == 1
    assert centers["coverage"] == pytest.approx(1.0)
    assert payload["view_metrics"]["coverage_by_d"]["1"] == pytest.approx(1.0)
    # The displayed partition and the search partition agree for categorical
    # features, and the payload says so rather than leaving it implicit.
    assert cert["partition_matches_search"] is True
    assert payload["search_centers"]["coverage"] == pytest.approx(1.0)


def test_a_singleton_pure_cell_follows_the_active_rule_in_the_payload():
    # Under the default rule the cell IS a centre and its interval shows how
    # little that establishes; under the strict rule it is not, and under
    # min_samples = 2 it is not either. All three must be visible in the
    # payload, because the renderer colours from `certified` and the panel
    # counts from `centers`, and those two must never disagree.
    X = np.array([[0]] * 400 + [[1]], dtype=int)
    Z = np.array([0] * 400 + [1], dtype=int)

    def build(spec):
        branch = _make_branch(X, Z, [0], d=1, spec=spec)
        return prepare_visualization_payload(
            branch, X, Z, feature_names=["a"], target_name="t",
            target_is_indicator=True, positive_value=1, center_spec=spec,
        )

    loose = build(CenterSpec(tau=0.9))
    grid = loose["grids"]["1"]
    singleton = [i for i, size in enumerate(grid["sizes"]) if size == 1]
    assert len(singleton) == 1
    i = singleton[0]
    assert grid["purity"][i] == pytest.approx(1.0)
    assert grid["certified"][i] is True
    assert loose["centers"]["n_centers"] == 1
    # ... and the interval attached to it is [alpha_eff, 1]: the data are
    # equally consistent with a purity of a few percent.
    assert grid["purity_lower"][i] == pytest.approx(loose["certificate"]["alpha_effective"], abs=1e-9)
    assert grid["purity_upper"][i] == pytest.approx(1.0)

    for spec in (CenterSpec(tau=0.9, min_samples=2), CenterSpec(tau=0.9, rule="certified")):
        strict = build(spec)
        assert strict["grids"]["1"]["certified"][i] is False, spec
        assert strict["centers"]["n_centers"] == 0, spec


# ---------------------------------------------------------------------------
# Payload/renderer contract
# ---------------------------------------------------------------------------
def test_every_payload_key_the_renderers_read_is_actually_produced():
    """
    Static contract check between `prepare_visualization_payload` and both
    front-ends. A renderer reading a key that the payload stopped producing
    fails silently in the browser (JavaScript yields `undefined`, the cell
    draws grey, nobody sees an error), so the coupling is pinned here rather
    than discovered visually.
    """
    from pathlib import Path

    rng = np.random.default_rng(29)
    n = 2000
    X = rng.integers(0, 4, size=(n, 4))
    Z = ((X[:, 0] == 0) & (X[:, 1] == 1)).astype(int)
    branches = discover_branches(
        X, Z, feature_names=list("abcd"),
        n_permutations_centers=0, cv_repeats=0, positive_class=1,
    )
    payload = prepare_visualization_payload(
        branches[4], X, Z, feature_names=list("abcd"),
        target_name="t", target_is_indicator=True, positive_value=1,
    )

    required_top = {
        "grids", "metrics", "view_metrics", "certificate", "centers",
        "search_centers", "total_samples", "rendered_samples", "slice_axis",
        "global_max_n", "axis_names", "axis_ticks", "selected_features",
        "target_name",
    }
    assert required_top <= set(payload)
    assert "class_breakdown" not in payload  # read BranchResult.per_class, now gone

    required_grid = {
        "x", "y", "z", "opacity", "purity", "purity_lower", "purity_upper",
        "certified", "positives", "sizes", "hover_text", "customdata", "summary",
    }
    for key, grid in payload["grids"].items():
        assert required_grid <= set(grid), f"grid {key!r} is missing {required_grid - set(grid)}"
        lengths = {len(grid[f]) for f in ("x", "y", "z", "purity", "purity_lower",
                                          "certified", "sizes", "customdata")}
        assert len(lengths) == 1, f"grid {key!r} has ragged arrays: {lengths}"

    required_view = {
        "coverage_by_d", "n_centers_by_d", "purity_by_d", "max_purity_lower_by_d",
    }
    assert required_view <= set(payload["view_metrics"])

    required_cert = {
        "tau", "alpha", "method", "multiplicity", "alpha_effective",
        "positive_value", "positive_label", "partition_matches_search",
    }
    assert required_cert <= set(payload["certificate"])

    # A 4-D branch produces per-slice grids; each must carry the same fields.
    assert payload["slice_axis"] is not None
    assert any(k.startswith("4_") for k in payload["grids"])

    # The keys must also be the ones the shipped front-end actually reads.
    app_js = (Path(__file__).resolve().parent.parent
              / "vsf" / "webapp" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    for key in ("certified", "rendered_samples", "coverage_by_d",
                "n_centers_by_d", "search_centers", "certificate"):
        assert key in app_js, f"app.js no longer reads {key!r}"
    # `purity_lower`/`purity_upper` reach the user through the hover text the
    # payload builds server-side, not through a JS lookup, so they are pinned
    # on the payload instead of on app.js.
    assert "Confidence bound" in payload["grids"]["1"]["hover_text"][0]
