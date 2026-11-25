"""Exact Wilcoxon signed-rank, Holm and Friedman of experiments/table_level_tests.py."""
from __future__ import annotations

import importlib.util
import itertools
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Sequence, Tuple

import pytest
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vsf_experiments_table_level_tests", _ROOT / "experiments" / "table_level_tests.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tl = _load()


def _brute(differences: Sequence[float]) -> Tuple[float, float]:
    """(W+, two-sided p) by enumerating every sign assignment of the non-zero differences (Pratt ranks)."""
    ranks = tl.midranks([abs(d) for d in differences])
    kept = [(r, d) for r, d in zip(ranks, differences) if d != 0]
    total = sum(r for r, _ in kept)
    observed = sum(r for r, d in kept if d > 0)
    distance = abs(observed - total / 2)
    hits = 0
    for signs in itertools.product((0, 1), repeat=len(kept)):
        s = sum(r for (r, _), keep in zip(kept, signs) if keep)
        hits += abs(s - total / 2) >= distance - 1e-12
    return observed, hits / 2 ** len(kept)


def test_midranks_share_ties() -> None:
    assert tl.midranks([3.0, 1.0, 3.0, 2.0]) == [3.5, 1.0, 3.5, 2.0]


@pytest.mark.parametrize("seed", range(30))
def test_exact_wilcoxon_equals_enumeration_with_ties_and_zeros(seed: int) -> None:
    rng = random.Random(seed)
    differences: List[float] = [rng.choice((-3, -2, -1, 0, 0, 1, 2, 3, 4)) / 4 for _ in range(rng.randint(1, 11))]
    w, p, positive, zero, negative = tl.wilcoxon_exact(differences)
    w_ref, p_ref = _brute(differences)
    assert (positive, zero, negative) == (sum(d > 0 for d in differences), differences.count(0), sum(d < 0 for d in differences))
    assert w == pytest.approx(w_ref)
    assert p == pytest.approx(p_ref)


@pytest.mark.parametrize("seed", range(10))
def test_exact_wilcoxon_equals_scipy_without_ties(seed: int) -> None:
    rng = random.Random(100 + seed)
    differences = rng.sample([x / 7 for x in range(-30, 31) if x != 0], 12)
    while len({abs(d) for d in differences}) < len(differences):
        differences = rng.sample([x / 7 for x in range(-30, 31) if x != 0], 12)
    _, p, _, _, _ = tl.wilcoxon_exact(differences)
    assert p == pytest.approx(stats.wilcoxon(differences, alternative="two-sided", method="exact").pvalue)


def test_wilcoxon_edge_cases() -> None:
    assert tl.wilcoxon_exact([0.0, 0.0]) == (0.0, 1.0, 0, 2, 0)
    # ten tables all in favour of the first method: 2 / 2**10
    assert tl.wilcoxon_exact([0.1 * i for i in range(1, 11)])[1] == pytest.approx(2 / 1024)
    # one non-zero difference can never be significant
    assert tl.wilcoxon_exact([0.5, 0.0, 0.0])[1] == 1.0


def test_holm_is_monotone_and_bounded() -> None:
    assert tl.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert tl.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("seed", range(10))
def test_friedman_equals_scipy(seed: int) -> None:
    rng = random.Random(200 + seed)
    blocks = [[rng.choice((0.1, 0.2, 0.3, 0.4, 0.5)) for _ in range(4)] for _ in range(rng.randint(6, 15))]
    statistic, p, mean_ranks = tl.friedman(blocks)
    reference = stats.friedmanchisquare(*[[b[j] for b in blocks] for j in range(4)])
    assert statistic == pytest.approx(reference.statistic)
    assert p == pytest.approx(reference.pvalue, rel=1e-9, abs=1e-12)
    assert sum(mean_ranks) == pytest.approx(4 * 5 / 2)


def test_per_table_means_pool_configurations_and_drop_incomplete_methods() -> None:
    rows = [
        {"dataset": "a", "selection": "purity", "budget": "32", "cov_vsf": "0.2", "cov_partial": "0.4", "cov_ssdpp": "0.1"},
        {"dataset": "a", "selection": "purity", "budget": "32", "cov_vsf": "0.4", "cov_partial": "0.6", "cov_ssdpp": "nan"},
        {"dataset": "a", "selection": "purity", "budget": "16", "cov_vsf": "0.9", "cov_partial": "0.9", "cov_ssdpp": "0.9"},
        {"dataset": "b", "selection": "certified", "budget": "32", "cov_vsf": "0.5", "cov_partial": "0.5", "cov_ssdpp": "0.5"},
    ]
    means = tl.per_table_means(rows, "purity", 32, "cov")
    assert set(means) == {"a"}
    assert means["a"] == {"vsf": pytest.approx(0.3), "vsf_partial": pytest.approx(0.5)}
