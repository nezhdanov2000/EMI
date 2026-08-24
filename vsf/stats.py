"""
VSF Statistics Utilities: Multiple-Testing Correction

Implements the Benjamini-Hochberg (1995) step-up procedure for controlling
the False Discovery Rate (FDR), as mandated by Project_Master_Document.md
Section 4.5.3 ("Multiple Testing Protection (FDR Control)"):

    p_(i) <= (i / K_tests) * q

Every module in this package that screens more than one hypothesis at a
fixed significance level (AVR's Phase 1 marginal noise filter, the dirty
center pattern miner, the graph-reasoning chain miners, and the server's
/api/top_columns endpoint) must run its raw permutation p-values through
`benjamini_hochberg` before treating any of them as "significant" — a
single, uncorrected per-test alpha does not control the family-wise false
discovery rate once more than a handful of hypotheses are tested at once.
"""

import numpy as np


def benjamini_hochberg(p_values: np.ndarray | list[float], q: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg step-up procedure for FDR control.

    Given m p-values from m simultaneous hypothesis tests, finds the largest
    rank k such that the k-th smallest p-value satisfies p_(k) <= (k/m) * q,
    and rejects the null hypothesis (declares significance) for the k
    smallest p-values.

    Args:
        p_values: Array of raw (uncorrected) p-values, one per hypothesis test.
        q: Target False Discovery Rate (default 0.05, per VSF spec Section 4.5.3).

    Returns:
        Boolean array of the same length/order as `p_values`; True where the
        null hypothesis is rejected (i.e. the test is significant after FDR
        correction).
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")

    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    if np.any((p < 0.0) | (p > 1.0) | np.isnan(p)):
        raise ValueError("p_values must all be finite and in [0, 1]")

    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    thresholds = (ranks / m) * q

    passed = sorted_p <= thresholds
    if not np.any(passed):
        return np.zeros(m, dtype=bool)

    # Largest k (1-indexed) satisfying the BH inequality; reject the k
    # smallest p-values (step-up: once we find the largest qualifying rank,
    # everything below it is rejected too, even if it individually failed
    # the pointwise threshold at its own rank).
    k = int(np.max(np.where(passed)[0])) + 1

    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[:k] = True

    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject
