"""
Unit tests for `vsf.metrics` - the shared cell-code bookkeeping and the
Benjamini-Hochberg gate, which is all this module still contains.

What it no longer contains, and what these tests therefore pin as ABSENT:
every information-theoretic quantity. The bias-corrected association layer
that once ranked branches went first; `mutual_information_bits` and
`entropy_bits_from_counts` followed it once `vsf.pmd` stopped binning
continuous features and became a plain categorical encoder, and the
`contingency_table` / `contingency_from_codes` helpers went with them, since
feeding those estimators was all they were ever used for. See git history if
an information-theoretic quantity is ever wanted again.

What remains here:
  1. `cell_codes` / `dense_codes_from_flat` produce correct, dense
     joint-cell bookkeeping.
  2. `benjamini_hochberg` controls FDR - load-bearing for the Global
     Pattern Scan.
  3. A regression witness for the removals: the deleted names must stay
     gone, so an accidental reintroduction fails loudly here instead of
     silently resurrecting dead code.
"""

import numpy as np
import pytest

from vsf.metrics import benjamini_hochberg, cell_codes, dense_codes_from_flat


# ---------------------------------------------------------------------------
# 1. Cell-code bookkeeping
# ---------------------------------------------------------------------------

def test_cell_codes_are_dense_and_preserve_the_partition():
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.integers(0, 4, 500), rng.integers(0, 3, 500)])
    codes, n_cells = cell_codes(x)
    assert codes.shape == (500,)
    assert set(np.unique(codes).tolist()) == set(range(n_cells))
    # Two rows share a code iff they are equal in every column.
    for a, b in [(0, 1), (2, 7), (11, 300)]:
        assert (codes[a] == codes[b]) == bool(np.array_equal(x[a], x[b]))


def test_cell_codes_handles_string_columns():
    x = np.array([["red", "s"], ["red", "l"], ["blue", "s"], ["red", "s"]], dtype=object)
    codes, n_cells = cell_codes(x)
    assert n_cells == 3
    assert codes[0] == codes[3]
    assert codes[0] != codes[1] != codes[2]


def test_dense_codes_from_flat_matches_np_unique_on_both_paths():
    rng = np.random.default_rng(23)
    flat = rng.integers(0, 50, 4000)
    expected = np.unique(flat, return_inverse=True)[1]
    # Small nominal range -> the sort-free bincount path.
    assert np.array_equal(dense_codes_from_flat(flat, 50), expected)
    # Nominal range above the bincount limit -> the sort path. Both must
    # number the occupied values in the same ascending order.
    assert np.array_equal(dense_codes_from_flat(flat, (1 << 24) + 1), expected)
    assert dense_codes_from_flat(np.array([], dtype=np.int64), 10).shape == (0,)


# ---------------------------------------------------------------------------
# 2. Multiplicity control (Global Pattern Scan's FDR gate)
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_matches_a_worked_example():
    # m = 5, q = 0.05 -> critical values i*q/m = 0.01, 0.02, 0.03, 0.04, 0.05.
    # Sorted p = 0.005, 0.011, 0.02, 0.04, 0.13; the last index that clears its
    # own critical value is i = 4 (0.04 <= 0.04), so the step-up procedure
    # rejects the first four and only the fifth survives.
    p = [0.005, 0.011, 0.02, 0.04, 0.13]
    assert np.array_equal(benjamini_hochberg(p, 0.05),
                          np.array([True, True, True, True, False]))
    # One notch tighter on the largest rejected p and the cascade stops at 3.
    p2 = [0.005, 0.011, 0.02, 0.041, 0.13]
    assert np.array_equal(benjamini_hochberg(p2, 0.05),
                          np.array([True, True, True, False, False]))


def test_benjamini_hochberg_is_a_step_up_not_a_per_test_threshold():
    # p = 0.03 alone would fail 0.05*1/3, but the step-up procedure rejects
    # every hypothesis up to the LARGEST index that passes, so it is carried.
    p = [0.03, 0.001, 0.002]
    assert np.array_equal(benjamini_hochberg(p, 0.05), np.array([True, True, True]))


def test_benjamini_hochberg_edge_cases():
    assert benjamini_hochberg([], 0.05).shape == (0,)
    assert not benjamini_hochberg([0.9, 0.8], 0.05).any()
    assert benjamini_hochberg([0.0, 0.0], 0.05).all()
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1], q=0.0)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1], q=1.5)


def test_benjamini_hochberg_controls_fdr_under_a_global_null():
    # Every hypothesis true: any rejection is a false discovery, so the
    # expected proportion of runs with at least one rejection is bounded by q.
    rng = np.random.default_rng(22)
    q = 0.05
    n_runs, m = 400, 30
    any_rejected = [benjamini_hochberg(rng.random(m), q).any() for _ in range(n_runs)]
    assert np.mean(any_rejected) <= q * 2.0


# ---------------------------------------------------------------------------
# 3. Regression witness: the v2.3 removal itself
# ---------------------------------------------------------------------------

def test_the_deleted_association_ranking_api_stays_deleted():
    import vsf.metrics as metrics_module

    removed_names = [
        "mutual_information_bits",
        "entropy_bits_from_counts",
        "contingency_table",
        "contingency_from_codes",
        "expected_mutual_information_bits",
        "adjusted_mutual_information_bits",
        "adjusted_uncertainty_coefficient",
        "information_report",
        "ClassInfo",
        "InfoReport",
        "specific_surprise_bits",
        "miller_madow_bias_bits",
        "permutation_pvalue",
        "permutation_null",
        "familywise_max_null",
        "FamilywiseNull",
        "branch_score",
        "NullMethod",
    ]
    for name in removed_names:
        assert not hasattr(metrics_module, name), (
            f"vsf.metrics.{name} was removed and must not silently come back: "
            "VSF computes no information-theoretic quantity anywhere."
        )
    assert set(metrics_module.__all__) == {
        "benjamini_hochberg",
        "cell_codes",
        "dense_codes_from_flat",
    }


def test_no_module_in_the_package_computes_mutual_information():
    """
    Package-wide witness: no module imports or defines an
    information-theoretic quantity. `vsf.math` (entropy / MI / NMI) was
    deleted outright, and nothing may import it back.
    """
    import importlib
    import pkgutil

    import vsf

    with pytest.raises(ImportError):
        importlib.import_module("vsf.math")

    banned = ("mutual_information", "shannon_entropy", "joint_entropy",
              "entropy_bits_from_counts", "normalized_mutual_information")
    for mod_info in pkgutil.iter_modules(vsf.__path__):
        module = importlib.import_module(f"vsf.{mod_info.name}")
        for name in banned:
            assert not hasattr(module, name), (
                f"vsf.{mod_info.name}.{name} exists; VSF is a categorical "
                "framework and computes no information-theoretic quantity."
            )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
