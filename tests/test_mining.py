import numpy as np
import pandas as pd
import pytest
from vsf.math import joint_entropy, shannon_entropy
from vsf.mining import _min_support_count, mine_dirty_center, nmi_miller_madow_corrected

def test_nmi_miller_madow_corrected():
    # Test perfect correlation
    Z = np.array([0, 0, 0, 1, 1, 1])
    phi = np.array([0, 0, 0, 1, 1, 1])
    nmi = nmi_miller_madow_corrected(Z, phi)
    assert 0.0 <= nmi <= 1.0

    # Test small sample
    Z_small = np.array([1])
    phi_small = np.array([1])
    assert nmi_miller_madow_corrected(Z_small, phi_small) == 0.0


def test_nmi_miller_madow_corrected_matches_derived_formula():
    # Regression test for the Miller-Madow correction bug: an earlier
    # version computed `I_plugin - (K_Z-1)(K_phi-1) / (2N ln2)` (a PRODUCT
    # of the two marginal cardinality offsets, no K_joint term, and the
    # wrong sign relative to the per-entropy-term bias correction). The
    # correct formula per Miller (1955) / Panzeri & Treves (1996), summing
    # each of the three plugin entropy estimators' own bias correction, is
    #   I_MM = I_plugin + [(K_Z-1) + (K_phi-1) - (K_joint-1)] / (2N ln2)
    # This test recomputes that formula independently (from shannon_entropy
    # / joint_entropy, not by importing any internals of mining.py) and
    # checks the function matches it exactly, and diverges from the old
    # (wrong) formula on a case chosen so the two disagree well outside
    # floating-point tolerance and the [0, 1] clip doesn't mask it.
    rng = np.random.default_rng(3)
    N = 300
    base = rng.integers(0, 8, size=N)
    noisy = rng.random(N) < 0.3
    phi = np.where(noisy, rng.integers(0, 4, size=N), base % 4)

    h_z = shannon_entropy(base)
    h_phi = shannon_entropy(phi)
    h_zphi = joint_entropy(base, phi)
    i_plugin = max(0.0, h_z + h_phi - h_zphi)

    k_z = len(np.unique(base))
    k_phi = len(np.unique(phi))
    k_joint = len(np.unique(np.column_stack([base, phi]), axis=0))

    correct_correction = ((k_z - 1) + (k_phi - 1) - (k_joint - 1)) / (2 * N * np.log(2))
    denom = min(h_z, h_phi)
    expected = min(1.0, max(0.0, (i_plugin + correct_correction) / denom))

    old_buggy_correction = -((k_z - 1) * (k_phi - 1)) / (2 * N * np.log(2))
    old_buggy = min(1.0, max(0.0, (i_plugin + old_buggy_correction) / denom))

    actual = nmi_miller_madow_corrected(base, phi)

    assert actual == pytest.approx(expected, abs=1e-9)
    # Guard against silently reverting to the old formula: the two must
    # differ meaningfully for this construction (they do: ~0.0025 apart).
    assert abs(actual - old_buggy) > 1e-4


def test_min_support_count_is_or_not_and_of_the_two_floors():
    # Project_Master_Document.md Section 4.5.2: a predicate is admitted if
    # Support(phi) >= theta_supp OR N_pos >= 30 — an OR of a relative and an
    # absolute floor, so the correct minimal admissible N_phi is the SMALLER
    # of the two candidate thresholds. `max(min_absolute, theta_supp * N_c)`
    # would silently implement an AND instead (the larger, more restrictive
    # threshold), which is stricter than the spec for small dirty centers.
    # Small N_c: theta_supp*N_c=3 < min_absolute=30 -> OR admits at 3.
    assert _min_support_count(N_c=100, theta_supp=0.03, min_absolute=30) == 3
    # Large N_c: theta_supp*N_c=300 > min_absolute=30 -> OR admits at 30
    # (the smaller of the two), not 300.
    assert _min_support_count(N_c=10_000, theta_supp=0.03, min_absolute=30) == 30
    # Degenerate N_c=0 still returns a sane floor of at least 1.
    assert _min_support_count(N_c=0, theta_supp=0.03, min_absolute=30) >= 1


def test_mine_dirty_center_synthetic():
    df = pd.DataFrame({
        "f1": ["a", "a", "a", "a", "b", "b", "b", "b"] * 10,
        "f2": ["x", "x", "y", "y", "x", "x", "y", "y"] * 10,
        "f3": ["m", "n", "m", "n", "m", "n", "m", "n"] * 10,
    })
    # Target Z is 1 if f1 == 'a' and f2 == 'x'
    Z = ((df["f1"] == "a") & (df["f2"] == "x")).astype(int).values

    # Dirty center: select all samples
    mask = np.ones(len(df), dtype=bool)

    results = mine_dirty_center(df, Z, mask, I_Z_X_F=1.0, max_depth=2, top_t1=5)

    assert len(results) > 0
    best = results[0]
    assert "conditions" in best
    assert "human_text" in best
    assert "purity_pos" in best
    assert "nmi_local" in best
    assert "p_value" in best
    assert best["nmi_local"] > 0.0


def test_mine_dirty_center_pure_noise_finds_nothing_significant():
    # Regression test for the missing significance gate: an earlier version
    # accepted any candidate filter with nmi_local >= 0.05 and NO permutation
    # test at all, across a greedy search over hundreds of candidate
    # conjunctions — guaranteed to surface spurious "patterns" from a purely
    # random target given enough candidates. With BH-corrected permutation
    # testing, a genuinely unrelated target should survive with (close to)
    # nothing reported.
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame({
        f"f{i}": rng.choice(["a", "b", "c"], size=n) for i in range(6)
    })
    Z = rng.integers(0, 2, size=n)  # independent of every column in df
    mask = np.ones(n, dtype=bool)

    results = mine_dirty_center(
        df, Z, mask, I_Z_X_F=1.0, max_depth=3, top_t1=10,
        n_permutations=200, fdr_q=0.05, random_state=1,
    )
    assert len(results) == 0
