"""
Unit tests for vsf.avr module
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
