"""
Regression tests for vsf.graph_miner.

Covers two bugs from the code review:
  1. `mine_strong_links` scored a mediator chain by the PRODUCT of its
     edges' (differently-normalized) Uncertainty Coefficients, mislabeled
     in code comments as "DPI compliant" — the Data Processing Inequality
     says nothing about multiplying two normalized coefficients from what
     may not even be a Markov chain in the data. The score is now the
     MIN (bottleneck) of the chain's edges.
  2. Neither `mine_strong_links` nor `compute_predictiveness_matrix`-based
     chain search ran any significance test — edges/chains were accepted
     purely on point-estimate NMI magnitude against a fixed threshold, with
     no control over the false discovery rate across the large number of
     candidate edges implicitly screened. `compute_significant_edges` now
     gates every edge with a BH-corrected marginal permutation test.
"""

import numpy as np
import pandas as pd
import pytest

from vsf.graph_miner import (
    compute_predictiveness_matrix,
    compute_significant_edges,
    mine_strong_links,
)


def _chain_dataset(n=800, seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.choice(["a", "b"], size=n)
    z1 = np.where(rng.random(n) < 0.85, inp, rng.choice(["a", "b"], size=n))
    target = np.where(rng.random(n) < 0.8, z1, rng.choice(["x", "y"], size=n))
    noise = rng.choice(["p", "q", "r"], size=n)
    return pd.DataFrame({"inp": inp, "z1": z1, "noise": noise, "target": target})


def test_chain_score_is_bottleneck_min_not_product():
    df = _chain_dataset()
    u = compute_predictiveness_matrix(df)
    nmi_inp_z1 = u["inp"]["z1"]
    nmi_z1_tgt = u["z1"]["target"]
    # Sanity: the two edge strengths actually differ, so min != product
    # (otherwise this construction wouldn't distinguish the two formulas).
    assert abs(nmi_inp_z1 - nmi_z1_tgt) > 0.05

    res = mine_strong_links(
        df, target="target", min_nmi=0.05, max_depth=2,
        n_permutations=200, random_state=42, only_winning=False,
    )
    chain = next(c for c in res["top_chains"] if c["path"] == ["inp", "z1", "target"])

    assert chain["chain_score"] == pytest.approx(min(nmi_inp_z1, nmi_z1_tgt), abs=1e-9)
    # It must NOT be (or be close to) the product — the old, wrong formula.
    assert chain["chain_score"] != pytest.approx(nmi_inp_z1 * nmi_z1_tgt, abs=1e-6)


def test_compute_predictiveness_matrix_symmetric_mi_reused():
    # U(Y|X) and U(X|Y) must both be derived from the SAME mutual
    # information value (MI is symmetric) — an earlier version called
    # mutual_information once per ORDERED pair, which is wasteful but also
    # a correctness trap if the two calls could ever disagree.
    df = _chain_dataset()
    u = compute_predictiveness_matrix(df)
    from vsf.math import mutual_information, shannon_entropy

    mi = mutual_information(df["inp"].values, df["z1"].values)
    h_inp = shannon_entropy(df["inp"].values)
    h_z1 = shannon_entropy(df["z1"].values)

    assert u["inp"]["z1"] == pytest.approx(float(mi / h_z1), abs=1e-9)
    assert u["z1"]["inp"] == pytest.approx(float(mi / h_inp), abs=1e-9)


def test_compute_significant_edges_rejects_pure_noise_pairs():
    # With many mutually-independent columns, a fixed-alpha per-pair test
    # would let some pairs through purely by chance; BH-corrected joint
    # testing across all M*(M-1)/2 pairs should leave (close to) none
    # significant.
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({
        f"col{i}": rng.choice(["a", "b", "c"], size=n) for i in range(8)
    })
    p_values, significant = compute_significant_edges(
        df, fdr_q=0.05, n_permutations=200, random_state=1
    )
    n_pairs = len(p_values)
    assert n_pairs == 8 * 7 // 2
    n_significant = sum(significant.values())
    # Not a strict zero guarantee (it's a random test), but should be a
    # small minority, not anywhere near the naive-alpha false-positive rate
    # across 28 simultaneous tests.
    assert n_significant <= 2


def test_mine_strong_links_significant_field_requires_all_edges_significant():
    df = _chain_dataset()
    res = mine_strong_links(
        df, target="target", min_nmi=0.0, max_depth=2,
        n_permutations=200, random_state=42, only_winning=False,
    )
    for chain in res["top_chains"]:
        if chain["significant"]:
            assert chain["edges_significant"] is True
            assert chain["direct_significant"] is True
        # significant must never be True unless BOTH components are.
        assert chain["significant"] == (chain["edges_significant"] and chain["direct_significant"])


def test_mine_strong_links_only_winning_filters_on_significance_not_magnitude():
    # `only_winning=True` used to mean "chain magnitude >= direct edge
    # magnitude"; it now means "every edge in the chain (including the
    # direct-edge comparison) is statistically significant". Every chain
    # returned with only_winning=True must therefore be `significant`.
    df = _chain_dataset()
    res = mine_strong_links(
        df, target="target", min_nmi=0.05, max_depth=2, only_winning=True,
        n_permutations=200, random_state=42,
    )
    for chain in res["top_chains"]:
        assert chain["significant"] is True
