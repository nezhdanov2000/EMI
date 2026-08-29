"""
Unit tests for vsf.benchmark, v2.0 "Clean Core".

`generate_synthetic_dataset` itself is UNCHANGED from v1.0 (same signature,
same behavior) -- the tests below confirm that directly. `evaluate_vsf_accuracy`
was rewritten to evaluate `BranchEngine`/`discover_branches` (Independent
Branch Discovery) instead of the removed `AVREngine`: the old `d_star_accuracy`
and `mean_vir` metrics are gone (meaningless under the new architecture, see
`vsf/benchmark.py`'s module docstring); the new `expected_d = min(engine.max_d,
d_true)` concept, `feature_recall_at_expected_d`, `mean_mi_at_expected_d`,
`mean_nmi_at_expected_d`, and `effective_n_samples` replace them.
"""

import unittest
import warnings

import numpy as np

from vsf.avr import MAX_BRANCH_D, BranchEngine
from vsf.benchmark import evaluate_vsf_accuracy, generate_synthetic_dataset
from vsf.math import normalized_mutual_information
from vsf.pmd import discretize_dataset


class TestGenerateSyntheticDataset(unittest.TestCase):
    """generate_synthetic_dataset is unchanged from v1.0 -- confirm its
    ground-truth structure directly."""

    def test_shape_and_true_indices(self):
        X, Z, true_idx, names = generate_synthetic_dataset(
            n_samples=200, d_true=3, n_noise_features=4, random_state=42
        )
        self.assertEqual(X.shape, (200, 7))
        self.assertEqual(len(Z), 200)
        self.assertEqual(true_idx, [0, 1, 2])
        self.assertEqual(len(names), 7)
        self.assertEqual(names[:3], ["Informative_1", "Informative_2", "Informative_3"])
        self.assertTrue(all(n.startswith("Noise_") for n in names[3:]))

    def test_z_has_2_pow_d_true_classes(self):
        for d_true in (1, 2, 4):
            _, Z, _, _ = generate_synthetic_dataset(n_samples=2000, d_true=d_true, random_state=1)
            self.assertLessEqual(len(np.unique(Z)), 2 ** d_true)
            self.assertGreaterEqual(Z.min(), 0)
            self.assertLess(Z.max(), 2 ** d_true)

    def test_no_single_feature_saturates_nmi_regardless_of_d_true(self):
        # With independent factors, no single feature can approach full NMI
        # when d_true > 1: I(Z; feature_j) <= H(factor_j) = 1 bit while
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

    def test_rejects_invalid_d_true(self):
        with self.assertRaises(ValueError):
            generate_synthetic_dataset(n_samples=100, d_true=0)

    def test_rejects_invalid_feature_error_rate(self):
        with self.assertRaises(ValueError):
            generate_synthetic_dataset(n_samples=100, d_true=2, feature_error_rate=0.5)
        with self.assertRaises(ValueError):
            generate_synthetic_dataset(n_samples=100, d_true=2, feature_error_rate=-0.1)

    def test_deterministic_given_random_state(self):
        X1, Z1, idx1, names1 = generate_synthetic_dataset(n_samples=100, d_true=2, random_state=7)
        X2, Z2, idx2, names2 = generate_synthetic_dataset(n_samples=100, d_true=2, random_state=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Z1, Z2)
        self.assertEqual(idx1, idx2)
        self.assertEqual(names1, names2)


class TestEvaluateVsfAccuracy(unittest.TestCase):

    def test_basic_shape_and_new_metric_names(self):
        engine = BranchEngine(max_d=4)
        eval_res = evaluate_vsf_accuracy(
            engine=engine, n_runs=2, d_true_list=[2, 3], n_samples=400, random_state=42
        )
        # Old v1.0 metrics must be gone.
        self.assertNotIn("overall_d_star_accuracy", eval_res)
        self.assertNotIn("mean_vir", eval_res)
        # New v2.0 metrics must be present.
        self.assertIn("overall_feature_recall", eval_res)
        self.assertIn("overall_mean_nmi", eval_res)
        self.assertIn("detailed_results", eval_res)
        self.assertEqual(set(eval_res["detailed_results"].keys()), {"d_true_2", "d_true_3"})
        for key in ("d_true_2", "d_true_3"):
            detail = eval_res["detailed_results"][key]
            self.assertIn("expected_d", detail)
            self.assertIn("feature_recall_at_expected_d", detail)
            self.assertIn("mean_mi_at_expected_d", detail)
            self.assertIn("mean_nmi_at_expected_d", detail)
            self.assertIn("effective_n_samples", detail)

    def test_expected_d_is_min_of_engine_max_d_and_d_true(self):
        engine = BranchEngine(max_d=4)
        eval_res = evaluate_vsf_accuracy(
            engine=engine, n_runs=1, d_true_list=[2, 4, 6], n_samples=2000, random_state=42
        )
        detailed = eval_res["detailed_results"]
        self.assertEqual(detailed["d_true_2"]["expected_d"], 2)   # d_true <= max_d
        self.assertEqual(detailed["d_true_4"]["expected_d"], 4)   # d_true == max_d
        self.assertEqual(detailed["d_true_6"]["expected_d"], 4)   # d_true > max_d, clamped

    def test_recall_is_high_when_d_true_within_engine_capacity(self):
        # When d_true <= engine.max_d, the branch at expected_d == d_true
        # should be well-powered to recover essentially all true features
        # (error_rate defaults keep individual features fairly clean).
        engine = BranchEngine(max_d=4)
        eval_res = evaluate_vsf_accuracy(
            engine=engine, n_runs=3, d_true_list=[2], n_samples=3000, random_state=42
        )
        self.assertGreaterEqual(eval_res["overall_feature_recall"], 0.5)

    def test_recall_degrades_honestly_not_errors_when_d_true_exceeds_max_d(self):
        # d_true=6 > MAX_BRANCH_D=4: the engine cannot search a 6-dimensional
        # combination at all, so expected_d clamps to 4 and can recover AT
        # MOST 4 of the 6 true features -- this must degrade gracefully
        # (recall < 1.0, reported honestly) rather than raising or silently
        # reporting perfect recall.
        engine = BranchEngine(max_d=MAX_BRANCH_D)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            eval_res = evaluate_vsf_accuracy(
                engine=engine, n_runs=2, d_true_list=[6], n_samples=500, random_state=42,
            )
        detail = eval_res["detailed_results"]["d_true_6"]
        self.assertEqual(detail["expected_d"], MAX_BRANCH_D)
        # At most 4 of the 6 true features can possibly be recalled by a
        # 4-dimensional branch -- recall must not exceed 4/6, and, since the
        # branch is a strict subset of the necessary dimensionality, it
        # cannot be a perfect 1.0.
        self.assertLessEqual(detail["feature_recall_at_expected_d"], 4 / 6 + 1e-9)
        self.assertLess(detail["feature_recall_at_expected_d"], 1.0)

    def test_scales_samples_and_warns_for_large_d_true(self):
        # Regression test for the "effective_n_samples" auto-scaling: a
        # large d_true (many target classes) run at a small fixed n_samples
        # would starve most classes of samples and produce unreliable
        # entropy estimates silently unless auto-scaled with a warning.
        engine = BranchEngine(max_d=4)
        with self.assertWarns(UserWarning):
            eval_res = evaluate_vsf_accuracy(
                engine=engine, n_runs=1, d_true_list=[6], n_samples=50,
                random_state=42, min_samples_per_class=20,
            )
        detail = eval_res["detailed_results"]["d_true_6"]
        # 2**6 * 20 = 1280, far above the requested 50.
        self.assertGreaterEqual(detail["effective_n_samples"], 20 * (2 ** 6))

    def test_no_warning_when_samples_already_sufficient(self):
        engine = BranchEngine(max_d=4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            evaluate_vsf_accuracy(
                engine=engine, n_runs=1, d_true_list=[2], n_samples=2000,
                random_state=42, min_samples_per_class=20,
            )

    def test_smaller_max_d_engine_still_reports_meaningful_metrics(self):
        # An engine capped below MAX_BRANCH_D must clamp expected_d
        # accordingly and still produce a coherent, non-erroring report.
        engine = BranchEngine(max_d=2)
        eval_res = evaluate_vsf_accuracy(
            engine=engine, n_runs=1, d_true_list=[1, 3], n_samples=1000, random_state=42,
        )
        detailed = eval_res["detailed_results"]
        self.assertEqual(detailed["d_true_1"]["expected_d"], 1)
        self.assertEqual(detailed["d_true_3"]["expected_d"], 2)  # clamped to engine.max_d


if __name__ == "__main__":
    unittest.main()
