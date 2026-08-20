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


if __name__ == "__main__":
    unittest.main()
