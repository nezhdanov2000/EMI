"""The grid count of a description: the smallest number of <= d-column schemas that hold every group."""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType
from typing import FrozenSet, List, Sequence

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vsf_experiments_compactness_splits", _ROOT / "experiments" / "compactness_splits.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cs = _load()


def _brute(sets: Sequence[FrozenSet[int]], max_d: int) -> int:
    """Smallest number of blocks over all partitions of the sets whose block unions have <= max_d columns."""
    sets = list(dict.fromkeys(sets))
    if not sets:
        return 0
    best = len(sets)

    def place(index: int, blocks: List[FrozenSet[int]]) -> None:
        nonlocal best
        if len(blocks) >= best and index < len(sets):
            return
        if index == len(sets):
            best = min(best, len(blocks))
            return
        for i, block in enumerate(blocks):
            merged = block | sets[index]
            if len(merged) <= max_d:
                place(index + 1, blocks[:i] + [merged] + blocks[i + 1:])
        place(index + 1, blocks + [sets[index]])

    place(0, [])
    return best


def test_labels_of_cells_and_rules_name_their_columns() -> None:
    assert cs.group_columns("X_1+X_2#0") == frozenset({1, 2})
    assert cs.group_columns("X_3=3 & X_8=0 & X_12=1") == frozenset({3, 8, 12})
    with pytest.raises(ValueError):
        cs.group_columns("entropy-None-depth3-leaf3")


def test_known_grid_counts() -> None:
    f = frozenset
    assert cs.min_grids([], 4) == (0, True)
    assert cs.min_grids([f({1, 2}), f({1, 2}), f({2})], 4) == (1, True)
    # three pairs over three columns share one 3-column schema
    assert cs.min_grids([f({1, 2}), f({1, 3}), f({2, 3})], 4) == (1, True)
    # five single columns need two schemas of four
    assert cs.min_grids([f({i}) for i in range(5)], 4) == (2, True)
    # two disjoint 4-column rules cannot share a schema
    assert cs.min_grids([f({1, 2, 3, 4}), f({5, 6, 7, 8})], 4) == (2, True)
    # the column bound (6 columns -> 2) is not attainable: the three pairs of a triangle plus a far pair
    assert cs.min_grids([f({1, 2, 3}), f({3, 4, 5}), f({5, 6, 1})], 4) == (3, True)
    with pytest.raises(ValueError):
        cs.min_grids([f({1, 2, 3, 4, 5})], 4)


@pytest.mark.parametrize("seed", range(40))
def test_grid_count_equals_brute_force(seed: int) -> None:
    import random
    rng = random.Random(seed)
    n_columns = rng.randint(3, 9)
    sets = [frozenset(rng.sample(range(n_columns), rng.randint(1, min(4, n_columns)))) for _ in range(rng.randint(1, 7))]
    for max_d in (4, 3 if all(len(s) <= 3 for s in sets) else 4):
        count, exact = cs.min_grids(sets, max_d)
        assert exact
        assert count == _brute(sets, max_d)


def test_subsets_ride_along() -> None:
    sets = [frozenset(c) for r in (1, 2, 3) for c in itertools.combinations(range(4), r)]
    assert cs.min_grids(sets, 4) == (1, True)
