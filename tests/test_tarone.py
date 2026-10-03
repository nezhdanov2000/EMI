"""Family cell sizes (vsf.avr) and Tarone's reduction of the family (experiments/n_min_table.py)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from vsf.avr import _CandidateFactory, family_cell_count
from vsf.pmd import discretize_dataset

_ROOT = Path(__file__).resolve().parents[1]


def _load_n_min_table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vsf_experiments_n_min_table", _ROOT / "experiments" / "n_min_table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _factory(seed: int = 0, n: int = 600, m: int = 5) -> _CandidateFactory:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, m)).astype(str)
    X_discrete, bin_counts = discretize_dataset(X)
    return _CandidateFactory(X_discrete, bin_counts, n, ordered=[False] * m)


def test_family_cell_sizes_match_the_count_and_are_row_counts() -> None:
    fit = _factory()
    sizes = fit.family_cell_sizes(4)
    assert sizes.shape[0] == family_cell_count(fit, 4)
    assert sizes.dtype == np.int64
    assert np.all(np.diff(sizes) >= 0)
    assert sizes.min() >= 1 and sizes.max() <= fit.n_samples
    # the 1-D cells of one column partition the rows: their sizes sum to n
    codes, n_cells = fit.codes((0,))
    one_d = np.bincount(np.asarray(codes), minlength=n_cells)
    assert set(one_d[one_d > 0].tolist()) <= set(sizes.tolist())


def test_tarone_never_enlarges_the_family_and_never_raises_n_min() -> None:
    nm = _load_n_min_table()
    fit = _factory(n=800)
    sizes = fit.family_cell_sizes(4)
    T = sizes.shape[0]
    for tau in (0.7, 0.9):
        K, pure_tarone = nm.tarone_family(sizes, tau)
        pure_family, _ = nm.n_min(tau, 1.0, nm.ALPHA / T)
        assert 1 <= K <= T
        assert pure_tarone <= pure_family
        # K* is the smallest K at which the number of sets that can reach
        # the level alpha / K is at most K (m is non-increasing in K)
        assert int(np.count_nonzero(sizes >= pure_tarone)) <= K
        if K > 1:
            smaller, _ = nm.n_min(tau, 1.0, nm.ALPHA / (K - 1))
            assert int(np.count_nonzero(sizes >= smaller)) > K - 1
