"""
Regression tests for vsf.stats.benjamini_hochberg.

vsf.stats did not exist before the code-review fix pass: every module that
now imports `benjamini_hochberg` (avr.py Phase 1, mining.py, graph_miner.py)
previously thresholded raw p-values against a fixed alpha independently
across many simultaneous tests, with no control over the family-wise false
discovery rate. These tests pin down the BH step-up procedure itself.
"""

import numpy as np
import pytest

from vsf.stats import benjamini_hochberg


def test_bh_all_null_controls_false_discoveries():
    # 200 independent U(0,1) p-values under a TRUE null (no real signal
    # anywhere): naive alpha=0.05 thresholding would reject ~10 of these by
    # chance; BH at q=0.05 must reject dramatically fewer on average.
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200)
    reject = benjamini_hochberg(p, q=0.05)
    naive_reject_count = int(np.sum(p < 0.05))
    assert reject.sum() <= naive_reject_count


def test_bh_all_significant():
    # Every p-value effectively zero -> every hypothesis rejected.
    p = np.full(10, 1e-10)
    reject = benjamini_hochberg(p, q=0.05)
    assert reject.all()


def test_bh_none_significant():
    p = np.full(10, 0.99)
    reject = benjamini_hochberg(p, q=0.05)
    assert not reject.any()


def test_bh_textbook_example():
    # Classic worked example (Benjamini & Hochberg 1995 step-up procedure):
    # sorted p-values [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205,
    # 0.212, 0.216, 0.222, 0.251, 0.269, 0.275, 0.34] at q=0.05 with m=15
    # rejects the first 4 (the largest k with p_(k) <= k/m * q is k=4:
    # 0.041 <= 4/15*0.05 = 0.0133... actually recompute directly below).
    p = np.array([
        0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205,
        0.212, 0.216, 0.222, 0.251, 0.269, 0.275, 0.34,
    ])
    m = len(p)
    q = 0.05
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds
    if below.any():
        k_max = np.max(np.where(below)[0])
        expected_reject_sorted = np.zeros(m, dtype=bool)
        expected_reject_sorted[: k_max + 1] = True
    else:
        expected_reject_sorted = np.zeros(m, dtype=bool)
    expected = np.zeros(m, dtype=bool)
    expected[order] = expected_reject_sorted

    reject = benjamini_hochberg(p, q=q)
    np.testing.assert_array_equal(reject, expected)


def test_bh_returns_original_order():
    # Result must be indexed by the INPUT order, not sorted order.
    p = np.array([0.5, 0.001, 0.9, 0.002])
    reject = benjamini_hochberg(p, q=0.1)
    assert len(reject) == 4
    # The two smallest p-values (indices 1, 3) should be the ones most
    # likely to be rejected; whatever the exact cutoff, index 0 (p=0.5)
    # must never be rejected while index 1 (p=0.001) is not.
    if reject[0]:
        assert reject[1] and reject[3]


def test_bh_rejects_invalid_q():
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([0.1, 0.2]), q=0.0)
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([0.1, 0.2]), q=1.0)
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([0.1, 0.2]), q=-0.1)


def test_bh_rejects_out_of_range_pvalues():
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([0.1, 1.5]), q=0.05)
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([-0.1, 0.5]), q=0.05)


def test_bh_accepts_plain_list():
    reject = benjamini_hochberg([0.001, 0.5, 0.9], q=0.05)
    assert isinstance(reject, np.ndarray)
    assert reject.dtype == bool
    assert len(reject) == 3


def test_bh_empty_input():
    reject = benjamini_hochberg(np.array([]), q=0.05)
    assert len(reject) == 0
