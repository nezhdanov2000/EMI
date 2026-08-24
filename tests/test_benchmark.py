"""
Unit tests for vsf.benchmark module
"""

import unittest
import warnings

import numpy as np

from vsf.avr import AVREngine
from vsf.benchmark import evaluate_vsf_accuracy, generate_synthetic_dataset
from vsf.math import normalized_mutual_information
from vsf.pmd import discretize_dataset


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

    def test_no_single_feature_saturates_nmi_regardless_of_d_true(self):
        # Regression test for the generator bug: an earlier version's
        # "informative" features were recycled deterministic transforms of
        # the same scalar Z, so a SINGLE feature reached
        # NMI(Z; feature) ~= 0.81 no matter how large d_true was requested —
        # `d_true` did not represent a genuinely necessary dimensionality.
        # With independent factors, no single feature can approach full
        # NMI when d_true > 1: I(Z; feature_j) <= H(factor_j) = 1 bit while
        # H(Z) = d_true bits.
        for d_true in (3, 5):
            X, Z, true_idx, names = generate_synthetic_dataset(
                n_samples=4000, d_true=d_true, n_noise_features=3,
                feature_error_rate=0.15, random_state=1,
            )
            X_discrete, _, _ = discretize_dataset(X)
            for j in true_idx:
                nmi_single = normalized_mutual_information(Z, X_discrete[:, j])
                self.assertLess(
                    nmi_single, 0.6,
                    msg=f"d_true={d_true}, feature {j}: single-feature NMI="
                        f"{nmi_single} should not approach saturation",
                )

    def test_generate_synthetic_dataset_rejects_invalid_feature_error_rate(self):
        with self.assertRaises(ValueError):
            generate_synthetic_dataset(n_samples=100, d_true=2, feature_error_rate=0.5)
        with self.assertRaises(ValueError):
            generate_synthetic_dataset(n_samples=100, d_true=2, feature_error_rate=-0.1)

    def test_evaluate_vsf_accuracy_scales_samples_and_warns_for_large_d_true(self):
        # Regression test for the "effective_n_samples" auto-scaling fix:
        # without it, a large d_true (many target classes) run at a small
        # fixed n_samples would starve most classes of samples and produce
        # unreliable entropy estimates silently.
        engine = AVREngine(alpha=0.01, n_permutations=50, random_state=42)
        with self.assertWarns(UserWarning):
            eval_res = evaluate_vsf_accuracy(
                engine=engine, n_runs=1, d_true_list=[6], n_samples=50,
                random_state=42, min_samples_per_class=20,
            )
        detail = eval_res["detailed_results"]["d_true_6"]
        self.assertIn("effective_n_samples", detail)
        # 2**6 * 20 = 1280, far above the requested 50.
        self.assertGreaterEqual(detail["effective_n_samples"], 20 * (2 ** 6))

    def test_evaluate_vsf_accuracy_no_warning_when_samples_already_sufficient(self):
        engine = AVREngine(alpha=0.01, n_permutations=50, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            evaluate_vsf_accuracy(
                engine=engine, n_runs=1, d_true_list=[2], n_samples=2000,
                random_state=42, min_samples_per_class=20,
            )


if __name__ == "__main__":
    unittest.main()
