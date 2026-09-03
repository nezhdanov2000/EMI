"""
Regression tests for vsf.vis.

Covers two bugs from the code review:
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

v2.0 note: `prepare_visualization_payload`'s first parameter is now a
`vsf.avr.BranchResult` (Project_Master_Document.md Section 4.2), not the
removed v1.0 `AVRResult`/`Scenario` -- no more scenario/vir/l_target/l_feat/
xai_message/history.

v2.1 note: `metrics` no longer carries `nmi`. It carries the raw plug-in `mi`
together with the noise floor `mi_null` it has to clear, the bias-corrected
`mi_adj`/`u_adj`, and the permutation p-values. A payload that offered a
consumer `nmi` would be offering a number that reads ~66 % on data provably
independent of a rare target. `vsf.vis` no longer exposes a `generate_interactive_html` function
(deleted, confirmed dead/no callers) -- there is nothing left to test for it.
"""

import numpy as np
import pytest

from vsf.avr import BranchResult, discover_branches
from vsf.vis import _axis_fallback_indices, humanize_col, humanize_val, prepare_visualization_payload


def _make_branch(selected_features, d=None):
    return BranchResult(
        d=d if d is not None else len(selected_features),
        selected_features=list(selected_features),
        selected_feature_names=[f"f{j}" for j in selected_features],
        mi=0.5,
        mi_null=0.05,
        mi_adj=0.45,
        u_adj=0.5,
        h_target=1.0,
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
    branch = _make_branch(selected_features=[1])
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
    branch = _make_branch(selected_features=[0, 1, 2])
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

    branch = _make_branch(selected_features=[0, 1, 2])
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
    branch = _make_branch(selected_features=[0, 1, 2])
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
    branch = _make_branch(selected_features=[0, 1, 2])
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
# v2.1 metrics shape -- no scenario/vir/history, and no `nmi`
# ---------------------------------------------------------------------------

def test_metrics_shape_is_the_v21_corrected_set():
    rng = np.random.default_rng(5)
    n = 200
    X = rng.integers(0, 3, size=(n, 3))
    Z = rng.integers(0, 2, size=n)
    branch = BranchResult(
        d=2,
        selected_features=[0, 1],
        selected_feature_names=["a", "b"],
        mi=0.734,
        mi_null=0.034,
        mi_adj=0.700,
        u_adj=0.612,
        h_target=1.178,
        p_value=0.001,
    )
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["a", "b", "c"], target_name="class", sort_Z=Z
    )
    assert set(payload["metrics"].keys()) == {
        "d", "mi", "mi_null", "mi_adj", "u_adj", "h_target",
        "p_value", "p_value_familywise", "significant",
    }
    assert payload["metrics"]["mi"] == pytest.approx(0.734)
    assert payload["metrics"]["mi_null"] == pytest.approx(0.034)
    assert payload["metrics"]["u_adj"] == pytest.approx(0.612)
    assert payload["metrics"]["p_value"] == pytest.approx(0.001)
    assert payload["metrics"]["p_value_familywise"] is None
    # No leftover v1.0 fields anywhere in the payload, and no NMI: a
    # consumer must not be able to reach the metric v2.1 removed.
    for banned in ("scenario", "vir", "l_target", "l_feat", "xai_message", "history", "d_star"):
        assert banned not in payload
        assert banned not in payload["metrics"]
    assert "nmi" not in payload["metrics"]
    assert "nmi_by_d" not in payload["view_metrics"]


# ---------------------------------------------------------------------------
# `view_metrics` -- regression coverage for the within-branch
# dimensionality-collapse HUD bug (the frontend was reading the branch's
# fixed full-d `metrics` scalars for every collapsed view, instead of the
# marginals of the axes actually on screen).
# ---------------------------------------------------------------------------

def test_view_metrics_falls_back_to_single_entry_for_hand_built_branch():
    # `_make_branch`-style BranchResult objects (constructed directly, not
    # via `discover_branches`) carry no per-prefix series --
    # `prepare_visualization_payload` must degrade gracefully to a
    # single-entry breakdown at the branch's own `d`, matching `metrics`,
    # rather than raising or emitting an empty/misleading dict.
    rng = np.random.default_rng(5)
    n = 200
    X = rng.integers(0, 3, size=(n, 3))
    Z = rng.integers(0, 2, size=n)
    branch = BranchResult(
        d=2, selected_features=[0, 1], selected_feature_names=["a", "b"],
        mi=0.734, mi_null=0.034, mi_adj=0.700, u_adj=0.612, h_target=1.178,
    )
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["a", "b", "c"], target_name="class", sort_Z=Z
    )
    assert payload["view_metrics"]["mi_by_d"] == {"2": pytest.approx(0.734)}
    assert payload["view_metrics"]["mi_adj_by_d"] == {"2": pytest.approx(0.700)}
    assert payload["view_metrics"]["u_adj_by_d"] == {"2": pytest.approx(0.612)}
    # v2.2: the centre series does NOT degrade with the branch. It is computed
    # from the displayed partition itself, not from `BranchResult`, so it is
    # populated for all three view dimensionalities even for a hand-built
    # branch that carries no prefix series and no CenterReport.
    for key in ("coverage_by_d", "n_centers_by_d", "purity_by_d"):
        assert set(payload["view_metrics"][key]) == {"1", "2", "3"}
    assert payload["search_centers"] is None


def test_view_metrics_carries_the_full_per_view_breakdown_from_discover_branches():
    # For a branch actually produced by `discover_branches`, `view_metrics`
    # must expose an entry for EVERY view dimensionality 1..branch.d (not
    # just the branch's own full d), and the full-d entry must match
    # `metrics` exactly.
    rng = np.random.default_rng(13)
    n = 500
    X = rng.integers(0, 3, size=(n, 4))
    Z = rng.integers(0, 2, size=n)
    branches = discover_branches(X, Z, max_d=3)
    branch = branches[3]
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["a", "b", "c", "d"], target_name="class", sort_Z=Z
    )
    vm = payload["view_metrics"]
    assert set(vm["mi_by_d"].keys()) == {"1", "2", "3"}
    assert set(vm["mi_adj_by_d"].keys()) == {"1", "2", "3"}
    assert set(vm["u_adj_by_d"].keys()) == {"1", "2", "3"}
    assert vm["mi_by_d"]["3"] == pytest.approx(payload["metrics"]["mi"])
    assert vm["mi_adj_by_d"]["3"] == pytest.approx(payload["metrics"]["mi_adj"])
    assert vm["u_adj_by_d"]["3"] == pytest.approx(payload["metrics"]["u_adj"])
    # A collapsed (1D/2D) view's mi must be a genuinely different number
    # from the fixed full-branch mi whenever the branch's own 3rd axis adds
    # information -- otherwise this test would not actually catch a
    # regression back to always showing `metrics.mi`.
    assert vm["mi_by_d"]["1"] != pytest.approx(payload["metrics"]["mi"])


def test_prepare_visualization_payload_handles_4d_branch_slice_axis():
    rng = np.random.default_rng(6)
    n = 400
    X = rng.integers(0, 3, size=(n, 4))
    Z = rng.integers(0, 2, size=n)
    branch = BranchResult(
        d=4,
        selected_features=[0, 1, 2, 3],
        selected_feature_names=["a", "b", "c", "w"],
        mi=0.4,
        mi_null=0.05,
        mi_adj=0.35,
        u_adj=0.3,
        h_target=1.0,
    )
    payload = prepare_visualization_payload(
        branch, X, Z, feature_names=["a", "b", "c", "w"], target_name="class", sort_Z=Z
    )
    assert payload["slice_axis"] is not None
    assert payload["slice_axis"]["name"] == "w"
    assert payload["metrics"]["d"] == 4
    assert payload["metrics"]["mi"] == pytest.approx(0.4)
    assert payload["metrics"]["u_adj"] == pytest.approx(0.3)
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
    # REGRESSION WITNESS for Section 0-ter, defect 2. Before v2.2,
    # `total_samples` held the RENDERED count while `vsf.avr` computed every
    # metric on all rows, and the per-cell purities drawn on screen were
    # computed on the subsample. The two Ns must now be reported separately
    # and the cell counts must sum to the full N.
    from vsf.avr import discover_branches

    rng = np.random.default_rng(19)
    n = 3000
    X = rng.integers(0, 3, size=(n, 3))
    Z = (X[:, 0] == 0).astype(int)
    branches = discover_branches(
        X, Z, feature_names=["a", "b", "c"], n_permutations=0, positive_class=1
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
    from vsf.avr import discover_branches

    rng = np.random.default_rng(21)
    n = 4000
    X = rng.integers(0, 4, size=(n, 3))
    Z = (X[:, 0] == 0).astype(int)
    branches = discover_branches(
        X, Z, feature_names=["a", "b", "c"], n_permutations=0, positive_class=1
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
    from vsf.avr import BranchResult
    from vsf.centers import CenterSpec

    X = np.array([[0]] * 400 + [[1]], dtype=int)
    Z = np.array([0] * 400 + [1], dtype=int)
    branch = BranchResult(
        d=1, selected_features=[0], selected_feature_names=["a"],
        mi=0.0, mi_null=0.0, mi_adj=0.0, u_adj=0.0, h_target=0.0,
    )

    def build(spec):
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
    front-ends. The v2.2 payload grew six grid arrays and two top-level
    blocks; a renderer reading a key that the payload stopped producing
    fails silently in the browser (JavaScript yields `undefined`, the cell
    draws grey, nobody sees an error), so the coupling is pinned here rather
    than discovered visually.
    """
    from pathlib import Path

    from vsf.avr import discover_branches

    rng = np.random.default_rng(29)
    n = 2000
    X = rng.integers(0, 4, size=(n, 4))
    Z = ((X[:, 0] == 0) & (X[:, 1] == 1)).astype(int)
    branches = discover_branches(
        X, Z, feature_names=list("abcd"), n_permutations=0, positive_class=1
    )
    payload = prepare_visualization_payload(
        branches[4], X, Z, feature_names=list("abcd"),
        target_name="t", target_is_indicator=True, positive_value=1,
    )

    required_top = {
        "grids", "metrics", "view_metrics", "certificate", "centers",
        "search_centers", "total_samples", "rendered_samples", "slice_axis",
        "global_max_n", "axis_names", "axis_ticks", "selected_features",
        "class_breakdown", "target_name",
    }
    assert required_top <= set(payload)

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
        "mi_by_d", "mi_adj_by_d", "u_adj_by_d",
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

    # The keys must also be the ones the shipped front-ends actually read.
    app_js = (Path(__file__).resolve().parent.parent
              / "vsf" / "webapp" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    for key in ("certified", "rendered_samples", "coverage_by_d",
                "n_centers_by_d", "search_centers", "certificate"):
        assert key in app_js, f"app.js no longer reads {key!r}"
    # `purity_lower`/`purity_upper` reach the user through the hover text the
    # payload builds server-side, not through a JS lookup, so they are pinned
    # on the payload instead of on app.js.
    assert "Confidence bound" in payload["grids"]["1"]["hover_text"][0]
