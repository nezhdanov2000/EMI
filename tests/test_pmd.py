"""
Unit tests for vsf.pmd module
"""

import unittest

import numpy as np

from vsf.pmd import (
    CHANNEL_LIMITS,
    adaptively_coarsen_bins,
    check_grid_capacity,
    discretize_feature,
)


class TestPMDModule(unittest.TestCase):

    def test_channel_limits_presence(self):
        self.assertEqual(CHANNEL_LIMITS["color_hue"], 12)
        self.assertEqual(CHANNEL_LIMITS["position_z"], 20)
        self.assertEqual(CHANNEL_LIMITS["color_saturation"], 7)

    def test_discretize_feature_limits(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, size=500)
        disc_x, k_act, dist = discretize_feature(x, channel_name="color_saturation")
        # k_act must not exceed L_v = 7 for color_saturation
        self.assertLessEqual(k_act, 7)
        self.assertEqual(len(disc_x), 500)
        self.assertGreaterEqual(dist, 0.0)

    def test_check_grid_capacity(self):
        # 100 samples -> max allowed cells = 10
        self.assertTrue(check_grid_capacity([2, 3], 100))  # 6 cells <= 10
        self.assertFalse(check_grid_capacity([4, 4], 100)) # 16 cells > 10

    def test_adaptively_coarsen_bins(self):
        # Create 4D array with 5 bins per col = 625 total cells
        np.random.seed(42)
        X = np.random.randint(0, 5, size=(100, 4))
        coarsened = adaptively_coarsen_bins(X, n_samples=100, target_max_cells=10)
        # Check that total unique cell combinations in coarsened array <= 10
        unique_cells = len(np.unique(coarsened, axis=0))
        self.assertLessEqual(unique_cells, 10)


if __name__ == "__main__":
    unittest.main()
