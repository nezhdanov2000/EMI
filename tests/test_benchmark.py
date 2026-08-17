"""
Unit tests for vsf.benchmark module
"""

import unittest
import numpy as np
from vsf.benchmark import generate_synthetic_dataset, evaluate_vsf_accuracy
from vsf.avr import AVREngine


class TestBenchmarkModule(unittest.TestCase):

    def test_generate_synthetic_dataset(self):
        X, Z, true_idx, names = generate_synthetic_dataset(
            n_samples=200, d_true=3, n_noise_features=4, random_state=42
        )
        self.assertEqual(X.shape, (200, 7))
        self.assertEqual(len(Z), 200)
        self.assertEqual(true_idx, [0, 1, 2])
        self.assertEqual(len(names), 7)

    def test_evaluate_vsf_accuracy(self):
        engine = AVREngine(alpha=0.01, n_permutations=100, random_state=42)
        eval_res = evaluate_vsf_accuracy(
            engine=engine, n_runs=2, d_true_list=[2, 3], n_samples=200, random_state=42
        )
        self.assertIn("overall_d_star_accuracy", eval_res)
        self.assertIn("overall_feature_recall", eval_res)
        self.assertGreaterEqual(eval_res["overall_feature_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
