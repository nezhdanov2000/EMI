"""
Regression tests for vsf.vis.

Covers two bugs from the code review:
  1. Axis fallback selection (`prepare_visualization_payload`): when AVR
     selects fewer than 3 features (d* < 3), the Y/Z scatter axes used to
     fall back to the raw literal column indices 1 and 2 regardless of
     what the primary axis already was — silently duplicating an axis
     whenever the single selected feature happened to be column 1 or 2.
  2. Multi-class purity (`build_grid`'s `pur` computation): the old formula
     `mean(class_index) / (K-1)` is the cell's average class index
     normalized to [0,1], which only coincides with "share of the positive
     class" when K=2. For K>2 it can report a large "purity" for a cell
     containing ZERO samples of the actual positive/last class.
"""

import numpy as np
import pytest

from vsf.avr import AVRResult, Scenario
from vsf.vis import _axis_fallback_indices, prepare_visualization_payload


def _make_result(selected_features, d_star=None):
    return AVRResult(
        selected_features=list(selected_features),
        selected_feature_names=[f"f{j}" for j in selected_features],
        d_star=d_star if d_star is not None else len(selected_features),
        scenario=Scenario.SCENARIO_A,
        vir=0.5,
        l_target=0.5,
        l_feat=0.0,
        nmi_full=0.5,
        xai_message="test",
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


def test_prepare_visualization_payload_no_duplicate_axes_when_d_star_1():
    rng = np.random.default_rng(0)
    n = 200
    n_features = 5
    X = rng.integers(0, 4, size=(n, n_features))
    Z = rng.integers(0, 2, size=n)
    feature_names = [f"feat_{i}" for i in range(n_features)]

    # Selected feature is literally column 1 — the exact case that used to
    # duplicate the Y axis onto X.
    res = _make_result(selected_features=[1])
    payload = prepare_visualization_payload(
        res, X, Z, feature_names=feature_names, target_name="class", sort_Z=Z
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
    res = _make_result(selected_features=[0, 1, 2])
    payload = prepare_visualization_payload(
        res, X, Z, feature_names=["a", "b", "c"], target_name="class", sort_Z=Z
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

    res = _make_result(selected_features=[0, 1, 2])
    payload = prepare_visualization_payload(
        res, X, Z, feature_names=["a", "b", "c"], target_name="class", sort_Z=Z
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
