"""
Unit tests for vsf.avr module
"""

import unittest
import warnings

import numpy as np

from vsf.avr import AVREngine, Scenario


class TestAVRModule(unittest.TestCase):

    def test_avr_scenario_d_pure_noise(self):
        # Pure noise dataset should trigger Scenario D (Chaos Block)
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1], size=200)
        X = rng.normal(0, 1, size=(200, 5))
        
        engine = AVREngine(alpha=0.01, n_permutations=200, random_state=42)
        res = engine.fit(X, Z)
        
        self.assertEqual(res.scenario, Scenario.SCENARIO_D)
        self.assertEqual(res.d_star, 0)
        self.assertEqual(len(res.selected_features), 0)

    def test_avr_scenario_a_minimalism(self):
        # 2 correlated features -> Scenario A (Minimalism)
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=300)
        x1 = Z + rng.normal(0, 0.1, size=300)
        x2 = np.sin(Z * np.pi) + rng.normal(0, 0.1, size=300)
        noise = rng.normal(0, 1, size=(300, 5))
        
        X = np.column_stack([x1, x2, noise])
        
        engine = AVREngine(alpha=0.01, n_permutations=200, random_state=42)
        res = engine.fit(X, Z)
        
        self.assertIn(res.scenario, [Scenario.SCENARIO_A, Scenario.SCENARIO_B])
        self.assertLessEqual(res.d_star, 3)
        self.assertIn(0, res.selected_features)

    def test_avr_submodularity_offline_check(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1], size=200)
        X = np.column_stack([Z + rng.normal(0, 0.1, 200), rng.normal(0, 1, (200, 3))])

        engine = AVREngine(alpha=0.01, n_permutations=100, random_state=42)
        res = engine.fit(X, Z, offline_brute_force=True)

        self.assertIsNotNone(res.submodularity_ratio)
        self.assertGreaterEqual(res.submodularity_ratio, 0.39)

    def test_phase3_routing_plateau_not_mislabeled_as_cap_warning(self):
        # Regression test for the Phase 3 scenario-routing bug: a selection
        # that PLATEAUS below max_d (greedy stops because no remaining
        # candidate's conditional gain is significant, not because max_d was
        # reached) must be routed to SCENARIO_B ("Full Load - Incomplete"),
        # never to SCENARIO_C, which is reserved for the case that actually
        # hit the max_d display cap (d_star == max_d).
        #
        # 5 independent informative bits -> 32-class target, each bit
        # revealed by its own (noisy) feature; a low-noise proxy makes all 5
        # features individually significant, so Phase 1 lets all 5 through,
        # but the greedy conditional-significance test is expected to stop
        # before consuming every one of them (small conditional gains from
        # the later bits get lost to entropy-estimation noise on a 32-class
        # target with a few thousand samples). Setting vir_threshold to an
        # unreachable 0.999 forces `vir < vir_threshold` unconditionally, so
        # the ONLY thing distinguishing SCENARIO_B-incomplete from
        # SCENARIO_C is whether d_star actually reached max_d.
        rng = np.random.default_rng(7)
        n = 3200
        d_true = 5
        factors = rng.integers(0, 2, size=(n, d_true))
        Z = (factors * (1 << np.arange(d_true))).sum(axis=1)

        X_cols = []
        for j in range(d_true):
            b = factors[:, j].astype(float)
            reveals = rng.random(n) >= 0.05
            corrupted = rng.integers(0, 2, size=n).astype(float)
            proxy = np.where(reveals, b, corrupted)
            X_cols.append(proxy + rng.normal(0, 0.1, n))
        for _ in range(3):
            X_cols.append(rng.normal(0, 1, n))
        X = np.column_stack(X_cols)

        engine = AVREngine(
            alpha=0.01, fdr_q=0.05, vir_threshold=0.999, max_d=7,
            n_permutations=200, random_state=42,
        )
        with warnings.catch_warnings():
            # Small strata from the high-cardinality (32-class) target are
            # expected here and are irrelevant to what this test checks.
            warnings.simplefilter("ignore", category=UserWarning)
            res = engine.fit(X, Z)

        self.assertLess(res.d_star, engine.max_d)
        self.assertLess(res.vir, engine.vir_threshold)
        self.assertEqual(res.scenario, Scenario.SCENARIO_B)
        self.assertIn("Incomplete", res.xai_message)
        # The message must not claim a >max_d cap situation it didn't reach.
        self.assertNotIn("Warning", res.xai_message)
        self.assertNotIn(f"beyond the {engine.max_d}-axis display cap", res.xai_message)

    def test_phase3_routing_genuine_cap_is_scenario_c(self):
        # Mirror image of the above: when d_star DOES reach max_d and VIR is
        # still below threshold, that genuinely is the >max_d warning case
        # and must be SCENARIO_C.
        rng = np.random.default_rng(3)
        n = 4000
        d_true = 8  # more informative bits than max_d can display
        factors = rng.integers(0, 2, size=(n, d_true))
        Z = (factors * (1 << np.arange(d_true))).sum(axis=1)

        X_cols = []
        for j in range(d_true):
            b = factors[:, j].astype(float)
            reveals = rng.random(n) >= 0.02
            corrupted = rng.integers(0, 2, size=n).astype(float)
            proxy = np.where(reveals, b, corrupted)
            X_cols.append(proxy + rng.normal(0, 0.05, n))
        X = np.column_stack(X_cols)

        engine = AVREngine(
            alpha=0.01, fdr_q=0.05, vir_threshold=0.999, max_d=4,
            n_permutations=200, random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            res = engine.fit(X, Z)

        self.assertEqual(res.d_star, engine.max_d)
        self.assertLess(res.vir, engine.vir_threshold)
        self.assertEqual(res.scenario, Scenario.SCENARIO_C)
        self.assertIn(f"Warning: >{engine.max_d}D", res.xai_message)

    def test_phase1_fdr_controls_noise_features_at_scale(self):
        # Regression test for Phase 1's switch from per-feature alpha
        # thresholding to joint Benjamini-Hochberg correction: with a large
        # number of pure-noise candidate features, a fixed per-test alpha=0.05
        # would be expected to let ~5% of them through by chance (false
        # discoveries); BH at fdr_q=0.05 must not.
        rng = np.random.default_rng(11)
        n = 500
        Z = rng.choice([0, 1], size=n)
        X = rng.normal(0, 1, size=(n, 60))  # 60 pure-noise features

        engine = AVREngine(alpha=0.05, fdr_q=0.05, n_permutations=200, random_state=42)
        res = engine.fit(X, Z)

        # A pure-noise design should trigger Scenario D (nothing survives
        # FDR-controlled Phase 1) far more reliably than it would survive
        # naive per-feature alpha=0.05 thresholding across 60 simultaneous
        # tests.
        self.assertEqual(res.scenario, Scenario.SCENARIO_D)
        self.assertEqual(res.d_star, 0)


if __name__ == "__main__":
    unittest.main()
