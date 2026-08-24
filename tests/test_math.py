"""
Unit tests for vsf.math module using standard unittest
"""

import unittest

import numpy as np

from vsf.math import (
    mutual_information,
    normalized_mutual_information,
    shannon_entropy,
)


class TestMathModule(unittest.TestCase):

    def test_shannon_entropy_uniform(self):
        # Uniform binary variable: H = 1.0 bit
        x = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(shannon_entropy(x), 1.0, places=5)

    def test_shannon_entropy_deterministic(self):
        # Constant variable: H = 0.0 bits
        x = np.array([5, 5, 5, 5, 5])
        self.assertAlmostEqual(shannon_entropy(x), 0.0, places=5)

    def test_mutual_information_identical(self):
        # I(X; X) = H(X)
        x = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        mi = mutual_information(x, x)
        h_x = shannon_entropy(x)
        self.assertAlmostEqual(mi, h_x, places=5)

    def test_mutual_information_independent(self):
        # Independent variables: I(X; Y) approx 0
        np.random.seed(42)
        x = np.random.choice([0, 1], size=1000)
        y = np.random.choice([0, 1], size=1000)
        nmi = normalized_mutual_information(x, y)
        self.assertLess(nmi, 0.05)

    def test_normalized_mutual_information_bounds(self):
        x = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        nmi = normalized_mutual_information(x, y)
        self.assertAlmostEqual(nmi, 1.0, places=5)

    def test_mi_and_nmi_stay_consistent_after_entropy_hoisting(self):
        # Regression test for the redundant-entropy-computation fix:
        # mutual_information and normalized_mutual_information used to
        # independently recompute H(Z)/H(X_S)/H(Z,X_S) (5 entropy calls
        # total across both functions for the same pair); they now share a
        # single `_mi_with_entropies` helper. This checks the shared path
        # still produces results consistent with the textbook relation
        # NMI = MI / min(H(Z), H(X_S)) computed independently here.
        rng = np.random.default_rng(0)
        z = rng.integers(0, 5, size=500)
        x = np.where(rng.random(500) < 0.6, z, rng.integers(0, 5, size=500))

        mi = mutual_information(z, x)
        nmi = normalized_mutual_information(z, x)
        h_z = shannon_entropy(z)
        h_x = shannon_entropy(x)

        expected_nmi = mi / min(h_z, h_x)
        self.assertAlmostEqual(nmi, expected_nmi, places=9)


if __name__ == "__main__":
    unittest.main()
