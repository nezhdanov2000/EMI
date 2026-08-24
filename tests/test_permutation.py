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

    def test_get_strata_groups_excludes_singletons_not_pools_them(self):
        # Regression test for the stratified permutation test's sparse-strata
        # bug: an earlier version pooled every singleton/sparse X_S stratum
        # into one shared "fallback" group and permuted Z-values ACROSS it,
        # even though those samples' X_S values are genuinely different from
        # one another. That breaks conditional exchangeability (samples that
        # do not share a conditioning value are not exchangeable under H0)
        # and biases the resulting p-value.
        #
        # Construct X_S with 3 DISTINCT singleton values (each appearing
        # exactly once) plus one stratum with 2 members (permutable). Under
        # the old pooling bug, the 3 singletons would end up in one group of
        # size 3 and get shuffled among themselves; under the fix, each is
        # its own excluded (non-permuted) sample.
        from vsf.permutation import _get_strata_groups

        x_s = np.array([100, 200, 300, 7, 7]).reshape(-1, 1)
        groups, n_excluded = _get_strata_groups(x_s)

        # The three singleton values (100, 200, 300) contribute 0 permutable
        # groups and are all excluded; only the (7, 7) pair forms a
        # permutable group of size 2.
        self.assertEqual(n_excluded, 3)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0].tolist()), [3, 4])

    def test_conditional_permutation_test_warns_on_heavy_exclusion(self):
        # When most of the sample falls into singleton X_S strata (common
        # once |S| grows and the conditioning grid gets sparse), the test
        # must warn rather than silently returning an unqualified p-value.
        rng = np.random.default_rng(0)
        n = 60
        z = rng.choice([0, 1], size=n)
        x_j = rng.choice([0, 1], size=n)
        # X_S with all-unique rows -> every sample is a singleton stratum.
        x_s = np.arange(n).reshape(-1, 1)

        with self.assertWarns(UserWarning):
            conditional_permutation_test(
                z, x_j, x_s, n_permutations=50, alpha=0.01, random_state=42,
                max_excluded_fraction=0.5,
            )

    def test_conditional_permutation_test_no_warning_when_strata_dense(self):
        rng = np.random.default_rng(1)
        n = 400
        z = rng.choice([0, 1], size=n)
        x_j = rng.choice([0, 1], size=n)
        # Only 2 distinct X_S values -> every sample lands in a large,
        # fully-permutable stratum; no exclusion at all.
        x_s = rng.choice([0, 1], size=n).reshape(-1, 1)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            conditional_permutation_test(
                z, x_j, x_s, n_permutations=50, alpha=0.01, random_state=42,
            )


if __name__ == "__main__":
    unittest.main()
