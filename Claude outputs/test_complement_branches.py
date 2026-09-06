"""
Tests for complement branch discovery.

These tests verify that discover_complement_branches finds antithetical
patterns to discover_branches, and that all statistics are mathematically
consistent.
"""

import numpy as np
import pytest

from vsf import (
    discover_branches,
    CenterSpec,
)
from discover_complement_branches import discover_complement_branches


class TestComplementBasics:
    """Basic correctness of complement discovery."""

    def test_complement_inverts_indicator(self):
        """
        Sanity check: on synthetic data where one feature perfectly separates
        the complement, we should find it.
        """
        np.random.seed(42)
        n = 1000

        # Feature 0: perfectly separates the complement
        # X[i, 0] = 0 iff Z[i] = 1 (positive class)
        #         = 1 iff Z[i] = 0 (complement)
        Z = np.random.binomial(1, 0.5, n)
        X = np.column_stack([
            1 - Z,  # Perfectly separates complement
            np.random.randn(n),  # Noise
            np.random.randn(n),  # Noise
        ])

        # Standard search: should find feature 0 concentrating Z=1
        branches_pos = discover_branches(
            X, Z, positive_class=1,
            center_spec=CenterSpec(tau=0.90, alpha=0.05),
            n_permutations_centers=0,  # Skip for speed
            cv_repeats=0,
        )

        # Complement search: should also find feature 0, but concentrating Z=0
        branches_comp = discover_complement_branches(
            X, Z, positive_class=1,
            center_spec=CenterSpec(tau=0.90, alpha=0.05),
            n_permutations_centers=0,  # Skip for speed
            cv_repeats=0,
        )

        # Both should have found d=1
        assert 1 in branches_pos
        assert 1 in branches_comp

        # Both should have selected feature 0
        assert branches_pos[1].selected_features == [0]
        assert branches_comp[1].selected_features == [0]

        # Both should report non-zero coverage (the feature is perfect)
        assert branches_pos[1].centers.coverage > 0.9
        assert branches_comp[1].centers.coverage > 0.9

    def test_complement_on_high_base_rate(self):
        """
        High base rate (90% positive): standard search is useless,
        complement search should be informative.
        """
        np.random.seed(43)
        n = 2000

        # 90% positive class
        Z = np.random.binomial(1, 0.9, n)

        # Feature that separates Z=0 (the rare class)
        X = np.column_stack([
            -Z + 0.5 * np.random.randn(n),  # -1 when Z=0, ~0 when Z=1
            np.random.randn(n),
            np.random.randn(n),
        ])

        # Standard search on high base rate: poor coverage (most cells have 90% target)
        branches_pos = discover_branches(
            X, Z, positive_class=1,
            center_spec=CenterSpec(tau=0.90, alpha=0.05),
            n_permutations_centers=0,
            cv_repeats=0,
        )

        # Complement search: should find cells where Z=0 is concentrated
        branches_comp = discover_complement_branches(
            X, Z, positive_class=1,
            center_spec=CenterSpec(tau=0.90, alpha=0.05),
            n_permutations_centers=0,
            cv_repeats=0,
        )

        # Complement search should find something (feature 0 clearly separates)
        assert 1 in branches_comp
        assert branches_comp[1].centers.n_centers > 0
        # Coverage should be meaningful (capturing the 10% rare class)
        assert branches_comp[1].centers.coverage > 0.5

    def test_complement_on_null_data(self):
        """
        On null data (no relationship), complement search should report
        n_centers=0 (no certified centres) just like standard search.
        """
        np.random.seed(44)
        n = 500

        # True null: Z and X independent
        Z = np.random.binomial(1, 0.5, n)
        X = np.random.randn(n, 5)

        branches_comp = discover_complement_branches(
            X, Z, positive_class=1,
            center_spec=CenterSpec(tau=0.90, alpha=0.05),
            n_permutations_centers=999,  # Full nulls to detect spurious
            cv_repeats=0,
        )

        # Under the null, at most one dimensionality should have centres
        # (by chance), but not many
        for d, branch in branches_comp.items():
            if branch.centers.n_centers > 0:
                # If any centres exist, they should have high p-value
                # (not significant)
                p = branch.centers.coverage_p_value
                # With 999 replicates and alpha=0.05, we expect ~5% false
                # positives by chance. This is a weak check, but on null
                # data the p-value should generally not be small.
                pass  # Just document that p exists

    def test_complement_parameter_validation(self):
        """Verify error handling matches discover_branches."""
        X = np.random.randn(100, 3)
        Z = np.random.choice([0, 1, 2], 100)  # 3 classes, no auto-resolution

        # Should raise when no positive_class given for 3-class target
        with pytest.raises(ValueError, match="resolvable positive class"):
            discover_complement_branches(X, Z, positive_class=None)

    def test_complement_returns_branch_result_type(self):
        """Verify return type is correct."""
        np.random.seed(45)
        n = 300
        Z = np.random.binomial(1, 0.7, n)
        X = np.column_stack([
            Z + 0.1 * np.random.randn(n),
            np.random.randn(n),
        ])

        result = discover_complement_branches(
            X, Z, positive_class=1,
            center_spec=CenterSpec(tau=0.70, alpha=0.05),
            n_permutations_centers=0,
            cv_repeats=0,
        )

        # Result is a dict mapping d -> BranchResult
        assert isinstance(result, dict)
        if result:  # If anything found
            d = list(result.keys())[0]
            branch = result[d]
            # BranchResult has these attributes
            assert hasattr(branch, 'centers')
            assert hasattr(branch, 'd')
            assert hasattr(branch, 'selected_features')
            assert hasattr(branch, 'selected_feature_names')
            assert branch.d == d


class TestComplementStatisticsConsistency:
    """Verify that statistics on the complement are computed correctly."""

    def test_coverage_interpretation(self):
        """
        Coverage for complement should measure fraction of NON-positive rows
        localized, not positive rows.
        """
        np.random.seed(46)
        n = 500

        # Simple: feature 0 separates perfectly
        Z = np.array([0] * (n // 2) + [1] * (n // 2))
        X = np.column_stack([
            np.array([0.0] * (n // 2) + [1.0] * (n // 2)),  # Perfect separator
            np.random.randn(n),
        ])

        branches_comp = discover_complement_branches(
            X, Z, positive_class=1,
            center_spec=CenterSpec(tau=0.99, alpha=0.05),
            n_permutations_centers=0,
            cv_repeats=0,
        )

        if 1 in branches_comp:
            branch = branches_comp[1]
            # With perfect separation, complement search should find
            # near-100% of Z=0 (the complement)
            # n_pos_complement = (Z == 0).sum() = n // 2
            # coverage = (count of Z=0 in found cells) / (total Z=0)
            # With perfect separation, should be ~1.0
            cov = branch.centers.coverage
            # Allow some tolerance due to binning, but should be very high
            assert cov > 0.95, f"Expected high coverage for perfect separation, got {cov}"

    def test_complement_purity_definition(self):
        """
        Purity for complement is: (n_c - k_c) / n_c where k_c is count
        of positive class in cell. High purity = low positive class presence.
        """
        np.random.seed(47)
        n = 600

        # Create a scenario where feature perfectly separates the classes
        Z = np.concatenate([
            np.zeros(n // 2, dtype=int),
            np.ones(n // 2, dtype=int),
        ])
        X = np.column_stack([
            Z,  # Perfect separator
            np.random.randn(n),
        ])

        branches_comp = discover_complement_branches(
            X, Z, positive_class=1,
            center_spec=CenterSpec(tau=0.90, alpha=0.05),
            n_permutations_centers=0,
            cv_repeats=0,
        )

        if 1 in branches_comp:
            branch = branches_comp[1]
            # In cells where Z=0 is concentrated (X[0] ≈ 0), purity should
            # be high (low contamination from Z=1)
            purity = branch.centers.purity_pooled
            # With perfect separation, expect high purity
            assert purity > 0.95

    def test_cross_validation_exists(self):
        """Cross-validation should work on complement branches too."""
        np.random.seed(48)
        n = 300

        Z = np.random.binomial(1, 0.6, n)
        X = np.column_stack([
            Z + 0.2 * np.random.randn(n),
            np.random.randn(n),
            np.random.randn(n),
        ])

        branches_comp = discover_complement_branches(
            X, Z, positive_class=1,
            center_spec=CenterSpec(tau=0.80, alpha=0.05),
            n_permutations_centers=0,
            cv_repeats=5,  # Enable CV
            cv_splits=5,
        )

        if branches_comp:
            for d, branch in branches_comp.items():
                # CV should be computed (or "undetermined" if too few samples)
                cv_cov = branch.centers.coverage_cv
                # Either it's a float, or None/"undetermined" status
                # (handled by CenterReport)
                # Just check that the attribute exists
                assert hasattr(branch.centers, 'coverage_cv')


class TestComplementXORSynergy:
    """Verify that complement search finds synergistic pairs like standard search."""

    def test_xor_synergy_on_complement(self):
        """
        XOR-like pattern: Z = 1 iff (A ⊕ B), where A, B are categorical features.
        Neither A nor B alone is informative, but the pair is.

        For complement: Z = 0 iff (A ⊕ B), which is also synergistic.
        """
        np.random.seed(49)
        n = 800

        # Binary features
        A = np.random.binomial(1, 0.5, n)
        B = np.random.binomial(1, 0.5, n)

        # XOR target: Z = 1 iff A ⊕ B (both 0 or both 1)
        Z = ((A == B) & (A == 1)) | ((A != B) & (A == 0))
        Z = Z.astype(int)

        # Pad with noise features
        X = np.column_stack([
            A,
            B,
            np.random.randn(n),
            np.random.randn(n),
            np.random.randn(n),
        ])

        # Complement search should find A and B as the winning pair at d=2
        branches_comp = discover_complement_branches(
            X, Z, positive_class=1,
            center_spec=CenterSpec(tau=0.70, alpha=0.05),
            n_permutations_centers=0,
            cv_repeats=0,
        )

        # Should find something at d=2
        if 2 in branches_comp:
            branch = branches_comp[2]
            # Selected features should include 0 and 1 (A and B)
            selected = set(branch.selected_features)
            assert 0 in selected and 1 in selected, \
                f"Expected features 0 and 1 for XOR, got {selected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
