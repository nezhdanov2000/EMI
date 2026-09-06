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
    discretize_dataset,
    discretize_feature,
    max_bins_per_dimension,
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

def test_check_grid_capacity():
    # 100 samples -> max allowed cells = 10
    assert check_grid_capacity([2, 3], 100)      # 6 cells <= 10
    assert not check_grid_capacity([4, 4], 100)  # 16 cells > 10


def test_adaptively_coarsen_bins_respects_the_ceiling():
    rng = np.random.default_rng(42)
    X = rng.integers(0, 5, size=(100, 4))        # 625 nominal cells
    coarsened = adaptively_coarsen_bins(X, n_samples=100, target_max_cells=10)
    assert len(np.unique(coarsened, axis=0)) <= 10


def test_adaptively_coarsen_bins_is_a_no_op_below_the_ceiling():
    X = np.repeat(np.arange(2), 50).reshape(-1, 1)
    out = adaptively_coarsen_bins(X, n_samples=100)
    assert np.array_equal(out, X)


def test_max_bins_per_dimension_and_coarsen_column():
    assert max_bins_per_dimension(2, 1000) ** 2 <= 100
    col = np.arange(10)
    merged = coarsen_column(col, 3)
    assert len(np.unique(merged)) == 3
    # Adjacent codes merge into the same group; the mapping is monotone.
    assert np.all(np.diff(merged) >= 0)
    assert np.array_equal(coarsen_column(col, 1), np.zeros_like(col))
    assert np.array_equal(coarsen_column(col, 20), col)


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
    assert set(pmd.__all__) == {
        "adaptively_coarsen_bins",
        "check_grid_capacity",
        "coarsen_column",
        "discretize_dataset",
        "discretize_feature",
        "max_bins_per_dimension",
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
