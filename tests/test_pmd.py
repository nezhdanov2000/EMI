"""
Unit tests for `vsf.pmd` - categorical encoding and grid-capacity control.

VSF is a categorical framework: every distinct value of a column is one
category, whatever dtype the column has. These tests pin that reading, and
pin the ABSENCE of the continuous-feature machinery that used to live here
(Freedman-Diaconis bin counts, quantile cut points, `CHANNEL_LIMITS`, and
the mutual-information "distortion" figure), so it cannot come back
unnoticed.
"""

import numpy as np
import pytest

import vsf.pmd as pmd
from vsf.pmd import (
    adaptively_coarsen_bins,
    check_grid_capacity,
    coarsen_column,
    coarsen_to_capacity,
    grid_capacity,
    level_frequency_order,
    occupied_cells,
    discretize_dataset,
    discretize_feature,
)


# ---------------------------------------------------------------------------
# 1. Encoding: one category per distinct value, whatever the dtype
# ---------------------------------------------------------------------------

def test_strings_are_encoded_in_sorted_distinct_value_order():
    x = np.array(["red", "blue", "red", "green", "blue"], dtype=object)
    codes, k = discretize_feature(x)
    assert k == 3
    # blue < green < red
    assert codes.tolist() == [2, 0, 2, 1, 0]


def test_numbers_are_categories_not_intervals():
    # 500 rows over 10 distinct integers: 10 categories, one per value.
    x = np.repeat(np.arange(10), 50)
    codes, k = discretize_feature(x)
    assert k == 10
    assert np.array_equal(codes, x)


def test_a_continuous_column_becomes_one_category_per_distinct_value():
    # The behaviour that replaced binning: no bin count, no cut points, no
    # cap. A column of 500 distinct floats is 500 categories, and the caller
    # sees that rather than a silently quantized axis.
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=500)
    codes, k = discretize_feature(x)
    assert k == 500
    assert sorted(codes.tolist()) == list(range(500))
    # Codes follow the sorted values: the smallest value gets code 0.
    assert codes[np.argmin(x)] == 0
    assert codes[np.argmax(x)] == 499


def test_encoding_is_independent_of_row_order():
    rng = np.random.default_rng(7)
    x = rng.integers(0, 6, 200)
    perm = rng.permutation(200)
    codes, k = discretize_feature(x)
    codes_perm, k_perm = discretize_feature(x[perm])
    assert k == k_perm
    assert np.array_equal(codes[perm], codes_perm)


def test_booleans_and_empty_input():
    codes, k = discretize_feature(np.array([True, False, True]))
    assert k == 2 and codes.tolist() == [1, 0, 1]
    codes, k = discretize_feature(np.array([]))
    assert k == 0 and codes.shape == (0,)


def test_unorderable_object_column_falls_back_to_string_forms():
    # A mixed column cannot be sorted numerically; encoding must still work.
    x = np.array(["a", 1, "b", 1], dtype=object)
    codes, k = discretize_feature(x)
    assert k == 3
    assert codes[1] == codes[3]


def test_discretize_dataset_encodes_every_column_and_reports_counts():
    X = np.array(
        [["red", 1, "x"],
         ["blue", 2, "x"],
         ["red", 2, "y"]],
        dtype=object,
    )
    X_discrete, counts = discretize_dataset(X)
    assert X_discrete.shape == (3, 3)
    assert counts == [2, 2, 2]
    assert X_discrete.dtype.kind in ("i", "u")
    # Same row values -> same codes.
    assert X_discrete[0, 0] == X_discrete[2, 0]


# ---------------------------------------------------------------------------
# 2. Grid capacity
# ---------------------------------------------------------------------------

def test_capacity_is_measured_on_occupied_cells_not_the_nominal_product():
    # 100 samples -> at most 10 occupied cells. Two 7-level columns whose
    # rows only ever combine into 7 joint values occupy 7 of 49 nominal
    # cells: within capacity, and must not be merged.
    assert grid_capacity(100) == 10
    a = np.repeat(np.arange(7), 15)[:100]
    X = np.column_stack([a, a])
    assert occupied_cells(X) == 7
    assert check_grid_capacity(X, 100)
    assert np.array_equal(adaptively_coarsen_bins(X, n_samples=100), X)
    rng = np.random.default_rng(0)
    Y = rng.integers(0, 4, size=(100, 2))  # 16 nominal, ~16 occupied
    assert occupied_cells(Y) > 10
    assert not check_grid_capacity(Y, 100)


def test_adaptively_coarsen_bins_respects_the_ceiling_and_stops_at_it():
    rng = np.random.default_rng(42)
    X = rng.integers(0, 5, size=(100, 4))        # 625 nominal cells
    coarsened, k = adaptively_coarsen_bins(X, n_samples=100, target_max_cells=10), None
    assert occupied_cells(coarsened) <= 10
    # Minimal: undoing the last merge would exceed the ceiling again. The
    # engine reports the level counts it stopped at; re-running it to one
    # level more on the column it reduced last must not fit.
    _, levels = coarsen_to_capacity(X.astype(np.int64), 10)
    assert occupied_cells(coarsened) <= 10
    assert sum(levels) < 4 * 5


def test_adaptively_coarsen_bins_is_a_no_op_below_the_ceiling():
    X = np.repeat(np.arange(2), 50).reshape(-1, 1)
    out = adaptively_coarsen_bins(X, n_samples=100)
    assert np.array_equal(out, X)


def test_coarsen_column_keeps_the_most_frequent_levels_and_merges_the_rest():
    # Level 3 is the most frequent, then 0, then 7; the rest are rare.
    col = np.array([3] * 10 + [0] * 6 + [7] * 4 + [1, 2, 4, 5, 6, 8, 9])
    assert level_frequency_order(col).tolist()[:3] == [3, 0, 7]
    merged = coarsen_column(col, 3)
    assert len(np.unique(merged)) == 3
    # Kept levels are recoded in frequency order; everything else is "other" (2).
    assert set(merged[col == 3]) == {0}
    assert set(merged[col == 0]) == {1}
    assert set(merged[np.isin(col, [7, 1, 2, 4, 5, 6, 8, 9])]) == {2}
    assert np.array_equal(coarsen_column(col, 1), np.zeros_like(col))
    assert np.array_equal(coarsen_column(col, 20), col)
    # Ties in frequency break by ascending code, so the result is deterministic.
    tie = np.array([5, 5, 2, 2, 9, 9, 1])
    assert level_frequency_order(tie).tolist() == [2, 5, 9, 1]


def test_ordered_columns_merge_adjacent_values_into_equal_count_ranges():
    from vsf.pmd import column_is_ordered, ordered_columns
    assert column_is_ordered(np.array([1.5, 2.0, 3.25]))
    assert column_is_ordered(np.array([1, 2, 3]))
    assert not column_is_ordered(np.array(["a", "b"]))
    assert not column_is_ordered(np.array([True, False]))
    assert ordered_columns(np.array([[1, "x"], [2, "y"]], dtype=object)) == [True, False]
    # 100 distinct values, uniform: 4 groups of 25 adjacent values each.
    col = np.arange(100)
    merged = coarsen_column(col, 4, ordered=True)
    assert merged.tolist() == [i // 25 for i in range(100)]
    # Skewed: one value holds half the rows; groups still follow value order
    # and never split a value.
    col = np.array([0] * 50 + list(range(1, 51)))
    merged = coarsen_column(col, 4, ordered=True)
    assert np.all(np.diff(merged) >= 0)
    assert len(np.unique(merged)) <= 4
    assert len(set(merged[col == 0])) == 1
    # The nominal rule on the same column would keep 0 and lump the rest.
    nominal = coarsen_column(col, 4)
    assert set(nominal[col == 0]) == {0} and len(np.unique(nominal)) == 4


def test_greedy_merge_reduces_the_widest_column_first():
    rng = np.random.default_rng(1)
    wide = rng.integers(0, 12, size=200)
    narrow = rng.integers(0, 3, size=200)
    X = np.column_stack([narrow, wide])
    out, levels = coarsen_to_capacity(X.astype(np.int64), 15)
    assert occupied_cells(out) <= 15
    assert levels[0] == 3            # the 3-level column was never touched
    assert levels[1] < 12
    assert np.array_equal(out[:, 0], narrow)


# ---------------------------------------------------------------------------
# 3. Regression witness: the continuous-feature machinery stays gone
# ---------------------------------------------------------------------------

def test_the_continuous_feature_machinery_stays_deleted():
    removed = [
        "CHANNEL_LIMITS",
        "freedman_diaconis_bins",
        "mutual_information_bits",
        "entropy_bits_from_counts",
        "contingency_table",
    ]
    for name in removed:
        assert not hasattr(pmd, name), (
            f"vsf.pmd.{name} was removed: VSF encodes categories and does not "
            "bin continuous features."
        )
    assert "max_bins_per_dimension" not in dir(pmd)  # the uniform per-axis cap is gone too
    assert set(pmd.__all__) == {
        "adaptively_coarsen_bins",
        "check_grid_capacity",
        "coarsen_column",
        "discretize_dataset",
        "discretize_feature",
        "column_is_ordered",
        "grid_capacity",
        "merge_state",
        "occupied_cells",
        "ordered_columns",
    }


def test_discretize_takes_no_binning_parameters():
    import inspect

    assert list(inspect.signature(discretize_feature).parameters) == ["X"]
    assert list(inspect.signature(discretize_dataset).parameters) == ["X_matrix"]
    # Two-tuple returns: no distortion figure to discard.
    assert len(discretize_feature(np.array([1, 2, 2]))) == 2
    assert len(discretize_dataset(np.array([[1], [2]], dtype=object))) == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
