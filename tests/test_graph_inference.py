"""
Regression tests for vsf.graph_inference.

`run_graph_inference` shares the same two bugs `mine_strong_links` had
(see tests/test_graph_miner.py): chain scoring must be the bottleneck MIN
across a path's edges, not a product, and every edge used to build a path
must clear a BH-corrected permutation-test significance bar
(`compute_significant_edges`), not just a raw NMI magnitude threshold.
"""

import numpy as np
import pandas as pd
import pytest

from vsf.graph_inference import compute_nmi_matrix, run_graph_inference
from vsf.graph_miner import compute_significant_edges


def _chain_dataset(n=800, seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.choice(["a", "b"], size=n)
    z1 = np.where(rng.random(n) < 0.85, inp, rng.choice(["a", "b"], size=n))
    target = np.where(rng.random(n) < 0.8, z1, rng.choice(["x", "y"], size=n))
    noise = rng.choice(["p", "q", "r"], size=n)
    return pd.DataFrame({"inp": inp, "z1": z1, "noise": noise, "target": target})


def test_path_score_is_bottleneck_min_of_edge_nmis():
    df = _chain_dataset()
    nmi = compute_nmi_matrix(df)
    res = run_graph_inference(
        df, inputs={"inp": "a"}, target="target", nmi_threshold=0.05,
        n_permutations=200, random_state=42,
    )
    assert res["best_paths"], "expected at least one discovered path"
    top = res["best_paths"][0]
    assert top["path"] == ["inp", "z1", "target"]

    expected_score = min(top["nmis"])
    assert top["score"] == pytest.approx(expected_score, abs=1e-9)
    assert res["chain_score"] == pytest.approx(expected_score, abs=1e-9)
    # It must not equal the product of the edge NMIs (the old, wrong score).
    product = 1.0
    for v in top["nmis"]:
        product *= v
    assert top["score"] != pytest.approx(product, abs=1e-6)


def test_paths_only_use_significant_edges():
    df = _chain_dataset()
    _, significant = compute_significant_edges(
        df, fdr_q=0.05, n_permutations=200, random_state=42
    )
    res = run_graph_inference(
        df, inputs={"inp": "a"}, target="target", nmi_threshold=0.0,
        n_permutations=200, random_state=42,
    )
    for path_info in res["best_paths"]:
        path = path_info["path"]
        for a, b in zip(path, path[1:]):
            assert significant.get(frozenset((a, b)), False), (
                f"path {path} used edge ({a},{b}) which did not clear the "
                "BH-corrected significance test"
            )


def test_no_significant_structure_yields_no_paths():
    # Every column independent of every other -> no edge should clear the
    # significance bar, so no mediator path should be reported (the
    # fallback BFS path may still populate nodes, but `best_paths` — which
    # feeds `chain_score`/`direct_nmi` display and the reasoning narrative —
    # must stay empty rather than stringing together spurious edges).
    rng = np.random.default_rng(9)
    n = 400
    df = pd.DataFrame({
        "inp": rng.choice(["a", "b"], size=n),
        "z1": rng.choice(["a", "b", "c"], size=n),
        "z2": rng.choice(["x", "y"], size=n),
        "target": rng.choice(["p", "q"], size=n),
    })
    res = run_graph_inference(
        df, inputs={"inp": "a"}, target="target", nmi_threshold=0.0,
        n_permutations=200, random_state=42,
    )
    assert res["best_paths"] == []
