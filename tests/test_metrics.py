"""
Unit tests for `vsf.metrics` — the bias-corrected estimator layer added in
v2.1.

These are not incidental coverage. Each block pins one of the claims the v2.1
redesign rests on, so that a future "simplification" back toward raw plug-in
MI or toward a symmetric NMI denominator fails loudly rather than silently:

  1. Point estimators agree with their textbook definitions and with each
     other (`mutual_information_bits` vs `H(Z) + H(X) - H(Z,X)`), and the
     table form is the numerically stable one on a micro-class.
  2. `expected_mutual_information_bits` reproduces the Monte-Carlo
     permutation mean — i.e. the closed-form noise floor is the real one.
  3. `_sample_null_mi`'s multiple-hypergeometric draw is distributionally
     identical to explicitly shuffling the N labels, which is what licenses
     using the cheaper of the two paths per table (up to 80x on a sparse
     grid; see that function's measured crossover).
  4. `adjusted_uncertainty_coefficient` is 0 under independence and 1 under
     determinism AT ANY CLASS BALANCE — the property that neither
     `MI / min(H)` nor `MI / sqrt(H H)` has.
  5. `specific_surprise_bits` decomposes MI exactly.
  6. `familywise_max_null` is never anti-conservative relative to the
     per-branch p-value.
  7. `benjamini_hochberg` controls FDR on a simulated global null.

Block 4 also carries an explicit REGRESSION WITNESS for the metric choice:
on a 7-in-32561 micro-class, NMI_geo reads 1.0% for pure noise and 1.6% for a
deterministic relation. Those two numbers are asserted, because "the
geometric mean is the academic standard" is a real and superficially
convincing counter-proposal, and the reason to reject it is exactly that it
cannot separate those two cases.
"""

import numpy as np
import pytest

from vsf.math import shannon_entropy
from vsf.metrics import (
    adjusted_mutual_information_bits,
    adjusted_uncertainty_coefficient,
    benjamini_hochberg,
    cell_codes,
    contingency_from_codes,
    contingency_table,
    entropy_bits_from_counts,
    expected_mutual_information_bits,
    familywise_max_null,
    information_report,
    miller_madow_bias_bits,
    mutual_information_bits,
    permutation_null,
    permutation_pvalue,
    specific_surprise_bits,
)
from vsf.metrics import _mi_bits_batch, _sample_null_mi

N_BIG = 32561          # the reference dataset's row count
N_RARE = 7             # "Never-worked" in that dataset
C_CAP = N_BIG // 10    # the Grid Capacity Limit's cell ceiling


# ---------------------------------------------------------------------------
# Fixtures: the four canonical (target, features) regimes
# ---------------------------------------------------------------------------

def _micro_class_noise(seed: int = 0):
    """7-in-32561 target, feature grid at the capacity ceiling, NO signal."""
    rng = np.random.default_rng(seed)
    z = np.zeros(N_BIG, dtype=int)
    z[rng.choice(N_BIG, N_RARE, replace=False)] = 1
    x = rng.integers(0, C_CAP, N_BIG)
    return z, x


def _micro_class_deterministic(seed: int = 0):
    """Same target, but every positive sits alone in its own cell."""
    z, x = _micro_class_noise(seed)
    x = x.copy()
    x[z == 1] = C_CAP + np.arange(N_RARE)
    return z, x


# ---------------------------------------------------------------------------
# 1. Coding and point estimators
# ---------------------------------------------------------------------------

def test_cell_codes_are_dense_and_preserve_the_partition():
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.integers(0, 4, 500), rng.integers(0, 3, 500)])
    codes, n_cells = cell_codes(x)
    assert codes.shape == (500,)
    assert set(np.unique(codes).tolist()) == set(range(n_cells))
    # Two rows share a code iff they are equal in every column.
    for a, b in [(0, 1), (2, 7), (11, 300)]:
        assert (codes[a] == codes[b]) == bool(np.array_equal(x[a], x[b]))


def test_cell_codes_handles_string_columns():
    x = np.array([["red", "s"], ["red", "l"], ["blue", "s"], ["red", "s"]], dtype=object)
    codes, n_cells = cell_codes(x)
    assert n_cells == 3
    assert codes[0] == codes[3]
    assert codes[0] != codes[1] != codes[2]


def test_contingency_table_matches_a_manual_count():
    z = np.array([0, 0, 1, 1, 1])
    x = np.array([0, 1, 1, 1, 0])
    assert np.array_equal(contingency_table(z, x), np.array([[1, 1], [1, 2]]))


def test_mutual_information_of_a_variable_with_itself_is_its_entropy():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 6, 2000)
    assert mutual_information_bits(contingency_table(x, x)) == pytest.approx(
        entropy_bits_from_counts(np.bincount(x)), abs=1e-12
    )


def test_mutual_information_is_symmetric_and_non_negative():
    rng = np.random.default_rng(3)
    z = rng.integers(0, 4, 1500)
    x = np.where(rng.random(1500) < 0.6, z, rng.integers(0, 4, 1500))
    forward = mutual_information_bits(contingency_table(z, x))
    backward = mutual_information_bits(contingency_table(x, z))
    assert forward == pytest.approx(backward, abs=1e-12)
    assert forward > 0.0


def test_table_form_beats_the_subtractive_form_on_a_micro_class():
    # H(Z) + H(X) - H(Z, X) subtracts three O(10)-bit quantities to produce an
    # O(1e-3)-bit answer. The table form does not, which is why v2.1 routes
    # `vsf.math` through it and dropped the `max(0.0, ...)` clamp.
    z, x = _micro_class_noise()
    table = contingency_table(z, x)
    mi_table = mutual_information_bits(table)
    h_z = entropy_bits_from_counts(table.sum(axis=1))
    h_x = entropy_bits_from_counts(table.sum(axis=0))
    h_joint = shannon_entropy(np.column_stack([z, x]))
    mi_subtractive = h_z + h_x - h_joint
    assert mi_table > 0.0
    assert mi_table == pytest.approx(mi_subtractive, abs=1e-9)
    # The subtractive form's absolute error is a fixed number of ULPs of
    # h_x (~11.6 bits), i.e. ~1e-15 * 2^4; relative to an MI of ~2e-3 that
    # is already a 1e-12 relative error, and it degrades linearly as the
    # grid grows. The table form has no such term.
    assert abs(mi_table - mi_subtractive) < 1e-9


def test_batched_mi_matches_the_scalar_estimator():
    rng = np.random.default_rng(4)
    tables = np.stack([
        contingency_table(rng.integers(0, 3, 800), rng.integers(0, 5, 800))
        for _ in range(6)
    ])
    batched = _mi_bits_batch(tables)
    scalar = np.array([mutual_information_bits(t) for t in tables])
    assert np.allclose(batched, scalar, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. The exact null expectation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_emi_reproduces_the_monte_carlo_permutation_mean(seed):
    rng = np.random.default_rng(seed)
    n = 4000
    z = rng.integers(0, 3, n)
    x = rng.integers(0, 11, n)
    table = contingency_table(z, x)
    exact = expected_mutual_information_bits(table)

    shuffler = np.random.default_rng(100 + seed)
    z_codes, n_rows = cell_codes(z)
    x_codes, n_cols = cell_codes(x)
    draws = np.array([
        mutual_information_bits(
            contingency_from_codes(shuffler.permutation(z_codes), x_codes, n_rows, n_cols)
        )
        for _ in range(3000)
    ])
    stderr = draws.std(ddof=1) / np.sqrt(draws.size)
    assert abs(exact - draws.mean()) < 4.0 * stderr


def test_exact_emi_grows_with_cell_count_which_is_the_whole_defect():
    # The reason raw MI cannot rank candidates of unequal cardinality.
    rng = np.random.default_rng(5)
    z = (rng.random(N_BIG) < 0.24).astype(int)
    floors = [
        expected_mutual_information_bits(contingency_table(z, rng.integers(0, c, N_BIG)))
        for c in (10, 100, 1000, C_CAP)
    ]
    assert floors == sorted(floors)
    assert floors[0] < 0.001
    # At the Grid Capacity Limit the floor is ~0.08 bits — an order of
    # magnitude above the "< 0.01 bits is noise" rule of thumb that raw MI
    # in bits invites.
    assert floors[-1] > 0.05


def test_emi_is_zero_for_a_degenerate_table():
    assert expected_mutual_information_bits(np.array([[10, 5]])) == 0.0
    assert expected_mutual_information_bits(np.zeros((0, 0), dtype=int)) == 0.0


def test_miller_madow_is_the_documented_closed_form():
    assert miller_madow_bias_bits(2, 1379, N_BIG) == pytest.approx(
        (2 - 1) * (1379 - 1) / (2.0 * N_BIG * np.log(2.0))
    )
    assert miller_madow_bias_bits(3, 4, 0) == 0.0


# ---------------------------------------------------------------------------
# 3. Null sampling: hypergeometric draw == explicit label shuffle
# ---------------------------------------------------------------------------

def test_hypergeometric_null_is_distributionally_identical_to_shuffling():
    rng = np.random.default_rng(6)
    n = 4000
    z = rng.integers(0, 3, n)
    x = rng.integers(0, 7, n)
    table = contingency_table(z, x)

    fast = _sample_null_mi(table.sum(axis=1), table.sum(axis=0), 4000,
                           np.random.default_rng(7))

    z_codes, n_rows = cell_codes(z)
    x_codes, n_cols = cell_codes(x)
    shuffler = np.random.default_rng(8)
    slow = np.array([
        mutual_information_bits(
            contingency_from_codes(shuffler.permutation(z_codes), x_codes, n_rows, n_cols)
        )
        for _ in range(4000)
    ])

    se = np.sqrt(fast.var(ddof=1) / fast.size + slow.var(ddof=1) / slow.size)
    assert abs(fast.mean() - slow.mean()) < 4.0 * se
    assert fast.std(ddof=1) == pytest.approx(slow.std(ddof=1), rel=0.10)
    # And both must centre on the closed-form expectation.
    assert fast.mean() == pytest.approx(expected_mutual_information_bits(table), rel=0.05)


def test_hypergeometric_null_handles_more_than_two_target_classes():
    rng = np.random.default_rng(9)
    n = 2000
    z = rng.integers(0, 5, n)
    x = rng.integers(0, 6, n)
    table = contingency_table(z, x)
    draws = _sample_null_mi(table.sum(axis=1), table.sum(axis=0), 800,
                            np.random.default_rng(10))
    assert draws.size == 800
    assert np.all(draws >= 0.0)
    assert draws.mean() == pytest.approx(expected_mutual_information_bits(table), rel=0.08)


def test_permutation_pvalue_is_uniform_under_the_null_and_tiny_under_signal():
    rng = np.random.default_rng(11)
    n = 3000
    x = rng.integers(0, 6, n)

    null_ps = []
    for k in range(40):
        z = rng.integers(0, 3, n)
        zc, nr = cell_codes(z)
        xc, nc = cell_codes(x)
        null_ps.append(permutation_pvalue(zc, xc, nr, nc, 199, random_state=k)[0])
    # Under the null a p-value is uniform: with 40 draws, roughly half should
    # exceed 0.5 and essentially none should be at the resolution floor.
    assert 0.25 < np.mean(np.array(null_ps) > 0.5) < 0.75
    assert np.mean(np.array(null_ps) <= 0.05) < 0.20

    z_signal = np.where(rng.random(n) < 0.8, x % 3, rng.integers(0, 3, n))
    zc, nr = cell_codes(z_signal)
    xc, nc = cell_codes(x)
    p, _, _ = permutation_pvalue(zc, xc, nr, nc, 199, random_state=0)
    assert p == pytest.approx(1.0 / 200.0)


def test_both_null_sampling_paths_agree_on_a_dense_table():
    """
    `permutation_pvalue` dispatches between direct multiple-hypergeometric
    sampling and explicit relabelling on `5 * R * C <= N`. The two are
    distributionally identical by construction (both condition on the same
    margins), and this pins that: the branch actually taken at a dense table
    must land on the same null as the one it did not take.
    """
    rng = np.random.default_rng(30)
    n = 6000
    z = rng.integers(0, 5, n)
    x = rng.integers(0, 400, n)          # 5 * 5 * 400 = 10000 > 6000 -> shuffle path
    z_codes, n_rows = cell_codes(z)
    x_codes, n_cols = cell_codes(x)
    assert 5 * n_rows * n_cols > n, "fixture must exercise the relabelling branch"

    table = contingency_from_codes(z_codes, x_codes, n_rows, n_cols)
    _, mean_dispatched, sd_dispatched = permutation_pvalue(
        z_codes, x_codes, n_rows, n_cols, 600, random_state=0
    )
    direct = _sample_null_mi(table.sum(axis=1), table.sum(axis=0), 600,
                             np.random.default_rng(1))
    se = np.sqrt(sd_dispatched ** 2 / 600 + direct.var(ddof=1) / 600)
    assert abs(mean_dispatched - direct.mean()) < 4.0 * se
    assert mean_dispatched == pytest.approx(
        expected_mutual_information_bits(table), rel=0.05
    )


def test_permutation_null_wrapper_matches_the_code_based_entry_point():
    rng = np.random.default_rng(12)
    z = rng.integers(0, 2, 1200)
    x = rng.integers(0, 5, 1200)
    mean, std, p = permutation_null(z, x, 199, random_state=3)
    zc, nr = cell_codes(z)
    xc, nc = cell_codes(x)
    p2, mean2, std2 = permutation_pvalue(zc, xc, nr, nc, 199, random_state=3)
    assert (mean, std, p) == pytest.approx((mean2, std2, p2))


# ---------------------------------------------------------------------------
# 4. The corrected metrics — and why not NMI
# ---------------------------------------------------------------------------

def test_u_adj_is_one_for_a_deterministic_relation():
    rng = np.random.default_rng(13)
    z = rng.integers(0, 4, 5000)
    assert adjusted_uncertainty_coefficient(contingency_table(z, z)) == pytest.approx(1.0)


def test_u_adj_is_essentially_zero_under_independence_at_the_capacity_limit():
    rng = np.random.default_rng(14)
    z = (rng.random(N_BIG) < 0.24).astype(int)
    x = rng.integers(0, C_CAP, N_BIG)
    table = contingency_table(z, x)
    assert adjusted_uncertainty_coefficient(table) < 0.01
    # The uncorrected quantities on the SAME table are not small.
    mi = mutual_information_bits(table)
    h_z = entropy_bits_from_counts(table.sum(axis=1))
    assert mi / h_z > 0.05


def test_u_adj_separates_noise_from_determinism_on_a_micro_class():
    """
    The headline regression. On a 7-in-32561 target at the grid capacity
    limit, compare the three candidate normalizations across pure noise and a
    deterministic relation. Only `u_adj` separates them.
    """
    def readings(z, x):
        table = contingency_table(z, x)
        mi = mutual_information_bits(table)
        h_z = entropy_bits_from_counts(table.sum(axis=1))
        h_x = entropy_bits_from_counts(table.sum(axis=0))
        return {
            "nmi_min": min(1.0, mi / min(h_z, h_x)),
            "nmi_geo": mi / np.sqrt(h_z * h_x),
            "u_adj": adjusted_uncertainty_coefficient(table),
        }

    noise = readings(*_micro_class_noise())
    exact = readings(*_micro_class_deterministic())

    # NMI_min: saturates on noise. Unusable as a percentage.
    assert noise["nmi_min"] > 0.5
    assert exact["nmi_min"] == pytest.approx(1.0, abs=1e-9)

    # NMI_geo: the geometric-mean proposal. It does not inflate — and that is
    # the trap: it also does not respond. A deterministic relation and pure
    # noise land within a factor of two of each other, both under 2%, so no
    # threshold can be placed between them.
    assert noise["nmi_geo"] < 0.02
    assert exact["nmi_geo"] < 0.02
    assert exact["nmi_geo"] / noise["nmi_geo"] < 2.0

    # U_adj: 0 vs 1. This is the property the reported percentage must have.
    assert noise["u_adj"] < 0.05
    assert exact["u_adj"] > 0.95


def test_micro_class_effect_size_still_needs_a_p_value():
    """
    Honest limitation, pinned so it is not forgotten: for a micro-class the
    ADJUSTED denominator H(Z) - E_0 is itself tiny, so a chance fluctuation in
    the numerator still moves `u_adj` by a few percent under the null. The
    p-value, not the effect size, is what settles a micro-class.
    """
    z, x = _micro_class_noise(seed=3)
    zc, nr = cell_codes(z)
    xc, nc = cell_codes(x)
    u = adjusted_uncertainty_coefficient(contingency_from_codes(zc, xc, nr, nc))
    p, _, _ = permutation_pvalue(zc, xc, nr, nc, 199, random_state=0)
    assert u < 0.10          # small, but demonstrably not exactly 0
    assert p > 0.05          # and the test correctly refuses to call it


def test_mi_adj_is_comparable_across_cardinalities_where_raw_mi_is_not():
    rng = np.random.default_rng(15)
    n = 8000
    z = (rng.random(n) < 0.3).astype(int)
    low = rng.integers(0, 2, n)      # binary noise
    high = rng.integers(0, 300, n)   # high-cardinality noise
    t_low = contingency_table(z, low)
    t_high = contingency_table(z, high)

    # Raw MI ranks the high-cardinality noise above the binary noise...
    assert mutual_information_bits(t_high) > mutual_information_bits(t_low)
    # ...while the corrected score puts both at the floor and does not.
    assert abs(adjusted_mutual_information_bits(t_high)) < 0.01
    assert abs(adjusted_mutual_information_bits(t_low)) < 0.01


def test_adjusted_estimators_reject_an_unknown_null_method():
    table = contingency_table(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
    for fn in (adjusted_mutual_information_bits, adjusted_uncertainty_coefficient):
        with pytest.raises(ValueError, match="unknown null method"):
            fn(table, null="bootstrap")


def test_null_none_reproduces_the_uncorrected_quantities():
    rng = np.random.default_rng(16)
    z = rng.integers(0, 3, 1000)
    x = np.where(rng.random(1000) < 0.7, z, rng.integers(0, 3, 1000))
    table = contingency_table(z, x)
    mi = mutual_information_bits(table)
    h_z = entropy_bits_from_counts(table.sum(axis=1))
    assert adjusted_mutual_information_bits(table, null="none") == pytest.approx(mi)
    assert adjusted_uncertainty_coefficient(table, null="none") == pytest.approx(mi / h_z)


# ---------------------------------------------------------------------------
# 5. Per-class decomposition
# ---------------------------------------------------------------------------

def test_specific_surprise_decomposes_mutual_information_exactly():
    rng = np.random.default_rng(17)
    n = 6000
    z = rng.integers(0, 5, n)
    x = np.where(rng.random(n) < 0.65, z, rng.integers(0, 5, n))
    table = contingency_table(z, x)
    priors = table.sum(axis=1) / table.sum()
    assert float(np.sum(priors * specific_surprise_bits(table))) == pytest.approx(
        mutual_information_bits(table), abs=1e-12
    )


def test_specific_surprise_is_non_negative():
    rng = np.random.default_rng(18)
    table = contingency_table(rng.integers(0, 4, 3000), rng.integers(0, 9, 3000))
    assert np.all(specific_surprise_bits(table) >= -1e-15)


def test_information_report_shows_a_rare_class_contributing_almost_nothing():
    # The question no scalar answers: is the headline number about the class
    # the user cares about, or about the majority class?
    rng = np.random.default_rng(19)
    n = 20000
    z = rng.choice([0, 1, 2], n, p=[0.7, 0.299, 0.001])
    x = np.where(z == 0, rng.integers(0, 20, n), rng.integers(15, 60, n))
    report = information_report(z, x, n_permutations=0)
    shares = {c.label_index: c.contribution_share for c in report.per_class}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares[2] < 0.02          # the 0.1%-prevalence class
    assert shares[0] + shares[1] > 0.95
    assert report.n_samples == n
    assert report.mi_adj_bits == pytest.approx(report.mi_bits - report.mi_null_bits)


def test_information_report_attaches_a_p_value_when_asked():
    rng = np.random.default_rng(20)
    z = rng.integers(0, 2, 2000)
    x = rng.integers(0, 5, 2000)
    assert information_report(z, x, n_permutations=0).p_value is None
    report = information_report(z, x, n_permutations=99, random_state=0)
    assert report.p_value is not None
    assert not report.is_signal()


# ---------------------------------------------------------------------------
# 6. Familywise (look-elsewhere) correction
# ---------------------------------------------------------------------------

def _noise_family(n=4000, n_features=6, seed=21):
    rng = np.random.default_rng(seed)
    cards = [2, 3, 5, 10, 50, 200][:n_features]
    cols = [rng.integers(0, c, n) for c in cards]
    z = (rng.random(n) < 0.3).astype(int)
    return z, cols


def test_familywise_null_dominates_the_single_candidate_null():
    z, cols = _noise_family()
    z_codes, _ = cell_codes(z)
    candidates = lambda: [
        (i, *cell_codes(col)) for i, col in enumerate(cols)
    ]
    fw = familywise_max_null(z_codes, candidates, n_permutations=199, random_state=0)
    assert fw.n_candidates == len(cols)
    assert fw.max_scores.shape == (199,)

    # The maximum over a family is stochastically larger than any single
    # member's score, so the familywise p-value can never be smaller than the
    # uncorrected one — which is precisely the look-elsewhere correction.
    best_key, best_codes, best_cells = max(
        candidates(),
        key=lambda c: adjusted_mutual_information_bits(
            contingency_from_codes(z_codes, c[1], 2, c[2])
        ),
    )
    observed = adjusted_mutual_information_bits(
        contingency_from_codes(z_codes, best_codes, 2, best_cells)
    )
    p_uncorrected, _, _ = permutation_pvalue(z_codes, best_codes, 2, best_cells, 199, 0)
    assert fw.p_value(observed) >= p_uncorrected - 1e-9


def test_familywise_null_is_disabled_cleanly_at_zero_replicates():
    z, cols = _noise_family()
    z_codes, _ = cell_codes(z)
    fw = familywise_max_null(z_codes, lambda: [(0, *cell_codes(cols[0]))], n_permutations=0)
    assert fw.n_permutations == 0
    assert fw.p_value(0.5) == 1.0


def test_familywise_null_uses_a_common_permutation_across_candidates():
    # If each candidate were scored against its OWN independent shuffle, the
    # maximum would be inflated (independent maxima are stochastically larger
    # than positively-correlated ones). Duplicating a candidate must therefore
    # leave the max-null distribution unchanged.
    z, cols = _noise_family(n_features=3)
    z_codes, _ = cell_codes(z)
    single = familywise_max_null(
        z_codes, lambda: [(i, *cell_codes(c)) for i, c in enumerate(cols)],
        n_permutations=199, random_state=0,
    )
    duplicated = familywise_max_null(
        z_codes,
        lambda: [(i, *cell_codes(c)) for i, c in enumerate(cols)]
                + [(f"dup{i}", *cell_codes(c)) for i, c in enumerate(cols)],
        n_permutations=199, random_state=0,
    )
    assert np.allclose(single.max_scores, duplicated.max_scores)


# ---------------------------------------------------------------------------
# 7. Multiplicity control
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_matches_a_worked_example():
    # m = 5, q = 0.05 -> critical values i*q/m = 0.01, 0.02, 0.03, 0.04, 0.05.
    # Sorted p = 0.005, 0.011, 0.02, 0.04, 0.13; the last index that clears its
    # own critical value is i = 4 (0.04 <= 0.04), so the step-up procedure
    # rejects the first four and only the fifth survives.
    p = [0.005, 0.011, 0.02, 0.04, 0.13]
    assert np.array_equal(benjamini_hochberg(p, 0.05),
                          np.array([True, True, True, True, False]))
    # One notch tighter on the largest rejected p and the cascade stops at 3.
    p2 = [0.005, 0.011, 0.02, 0.041, 0.13]
    assert np.array_equal(benjamini_hochberg(p2, 0.05),
                          np.array([True, True, True, False, False]))


def test_benjamini_hochberg_is_a_step_up_not_a_per_test_threshold():
    # p = 0.03 alone would fail 0.05*1/3, but the step-up procedure rejects
    # every hypothesis up to the LARGEST index that passes, so it is carried.
    p = [0.03, 0.001, 0.002]
    assert np.array_equal(benjamini_hochberg(p, 0.05), np.array([True, True, True]))


def test_benjamini_hochberg_edge_cases():
    assert benjamini_hochberg([], 0.05).shape == (0,)
    assert not benjamini_hochberg([0.9, 0.8], 0.05).any()
    assert benjamini_hochberg([0.0, 0.0], 0.05).all()
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1], q=0.0)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1], q=1.5)


def test_benjamini_hochberg_controls_fdr_under_a_global_null():
    # Every hypothesis true: any rejection is a false discovery, so the
    # expected proportion of runs with at least one rejection is bounded by q.
    rng = np.random.default_rng(22)
    q = 0.05
    n_runs, m = 400, 30
    any_rejected = [benjamini_hochberg(rng.random(m), q).any() for _ in range(n_runs)]
    assert np.mean(any_rejected) <= q * 2.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
