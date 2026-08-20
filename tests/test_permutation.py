"""
Unit tests for vsf.permutation module
"""

import unittest

import numpy as np

from vsf.permutation import conditional_permutation_test, marginal_permutation_test


class TestPermutationModule(unittest.TestCase):

    def test_marginal_permutation_test_correlated(self):
        np.random.seed(42)
        z = np.random.choice([0, 1], size=200)
        # x is strongly correlated with z
        x = z.copy()
        
        i_obs, p_val, is_sig = marginal_permutation_test(
            z, x, n_permutations=200, alpha=0.01, random_state=42
        )
        self.assertTrue(is_sig)
        self.assertLess(p_val, 0.01)

    def test_marginal_permutation_test_uncorrelated(self):
        rng = np.random.default_rng(42)
        z = rng.choice([0, 1], size=200)
        # x is independent noise
        x = rng.choice([0, 1, 2, 3], size=200)
        
        i_obs, p_val, is_sig = marginal_permutation_test(
            z, x, n_permutations=200, alpha=0.01, random_state=42
        )
        self.assertFalse(is_sig)
        self.assertGreaterEqual(p_val, 0.01)

    def test_conditional_permutation_test(self):
        rng = np.random.default_rng(42)
        z = rng.choice([0, 1], size=300)
        x_s = z.copy() # Already captures z
        x_j = rng.choice([0, 1], size=300) # Noise brings no extra info about z
        
        delta_obs, p_val, is_sig = conditional_permutation_test(
            z, x_j, x_s, n_permutations=200, alpha=0.01, random_state=42
        )
        # x_j adds no new info about z given x_s = z
        self.assertFalse(is_sig)


if __name__ == "__main__":
    unittest.main()
