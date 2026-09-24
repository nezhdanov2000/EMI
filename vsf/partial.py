"""
Partial-axis centres (Project_Master_Document.md Section 4.16).

A branch of the standard search shows one schema S of d <= 4 columns and
its centres are cells of the FULL grid on S: every axis fixed, d conditions
per centre. That is the alignment constraint, and phase 3 measured its
price against free conjunctive rules: with full-grid centres one schema
covers less than free rules and never more.

This module relaxes the constraint by one step without giving up the shared
picture. A *partial centre* of schema S is a cell of the grid on a
sub-schema S' subset of S: it fixes |S'| of the d axes and leaves the rest
free. On the display of S it is not a point but a bar (one free axis), a
plate (two) or a layer (three) of the same cube. It costs |S'| conditions.

Three facts make this cheap and safe:

* The candidate partial centres of S are exactly the centres of the
  proper sub-schemas of S, which the exhaustive search over d <= 4 already
  enumerates (`_CandidateFactory.iter_candidates` visits every S' before
  S). Nothing new is computed; the cells are re-used.
* The certificate is unchanged. A partial centre is a cell of the
  partition of S', and every such cell is already a member of the family T
  that `CenterSpec(multiplicity="family")` counts (`family_cell_count`).
  The post-selection guarantee of Section 4.14 therefore covers partial
  centres with the same alpha / T.
* Coverage stays a sum over disjoint groups. Partial centres of different
  sub-schemas can overlap; `pack_disjoint` chooses a set that does not, by
  a deterministic greedy (most positives first, ties: fewer conditions,
  fewer rows, enumeration order). The union of disjoint centres each of
  purity >= tau has purity >= tau, so the displayed purity claim holds for
  the whole selection.

The empirical reason to do this is in `experiments/partial_centres.py` and
`claude/partial-centres.md`: under the family certificate, one schema with
partial centres matches free disjoint rules in held-out coverage (2 wins,
0 losses, 77 ties over 79 configurations), where the full grid loses in 18.

The product's interactive search (`vsf.avr.discover_branches`) is NOT
changed by this module: it keeps full-grid centres. `search_partial` is a
separate entry point with the same factory, the same centre rule and the
same enumeration order, so its numbers are comparable to the branches'.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .avr import (
    MAX_BRANCH_D,
    _CandidateFactory,
    _prepare_search,
    apply_fitted_partition,
    resolve_center_spec,
)
from .centers import (
    CenterSpec,
    min_successes_to_select,
    stratified_repeated_kfold,
)

__all__ = [
    "PartialCentre",
    "PartialSelection",
    "pack_disjoint",
    "partial_centres_for_schema",
    "search_partial",
    "crossvalidate_partial",
]


@dataclass(frozen=True)
class PartialCentre:
    """One chosen centre: a cell of the partition of `sub_schema` (a subset of the schema)."""

    sub_schema: Tuple[int, ...]
    cell: int                 # dense cell code within the partition of `sub_schema`
    n: int
    k: int

    @property
    def cost(self) -> int:
        return len(self.sub_schema)

    @property
    def purity(self) -> float:
        return self.k / self.n if self.n else 0.0


@dataclass
class PartialSelection:
    """The disjoint partial centres chosen for one schema and their totals."""

    schema: Tuple[int, ...]
    centres: List[PartialCentre] = field(default_factory=list)
    k_sel: int = 0
    n_sel: int = 0
    n_positive: int = 0
    n_samples: int = 0

    @property
    def coverage(self) -> float:
        return self.k_sel / self.n_positive if self.n_positive else 0.0

    @property
    def mass(self) -> float:
        return self.n_sel / self.n_samples if self.n_samples else 0.0

    @property
    def purity(self) -> float:
        return self.k_sel / self.n_sel if self.n_sel else 0.0

    @property
    def conditions(self) -> int:
        return sum(c.cost for c in self.centres)

    @property
    def key(self) -> Tuple[float, float, float]:
        """Ranking key, the same shape as `coverage_score`'s first three components."""
        return (self.coverage, -float(len(self.centres)), -self.mass)


# --------------------------------------------------------------------------
# One schema
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _Cand:
    sub_schema: Tuple[int, ...]
    cell: int
    n: int
    k: int
    order: int               # enumeration order of the sub-schema (tie-break)
    rows: np.ndarray         # bool over the scored rows


def _qualifying_cells(
    codes: np.ndarray, n_cells: int, z: np.ndarray, spec: CenterSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cell ids, n, k) of the cells that are centres under `spec` - the search's own rule."""
    if n_cells == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e, e
    table = np.bincount(codes * 2 + z, minlength=2 * n_cells).reshape(n_cells, 2)
    n_cell = table.sum(axis=1)
    k_cell = table[:, 1]
    occupied = int(np.count_nonzero(n_cell > 0))
    if occupied == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    k_min = min_successes_to_select(n_cell, spec, spec.effective_alpha(occupied))
    ids = np.flatnonzero((n_cell > 0) & (k_cell >= k_min))
    return ids, n_cell[ids], k_cell[ids]


def pack_disjoint(cands: Sequence[_Cand]) -> List[_Cand]:
    """
    Deterministic disjoint packing: candidates in the order (more positives,
    fewer conditions, fewer rows, earlier sub-schema, lower cell id); each is
    taken if it shares no row with those already taken. Among candidates with
    identical row sets the cheapest description wins (it comes first).
    """
    order = sorted(
        range(len(cands)),
        key=lambda i: (-cands[i].k, len(cands[i].sub_schema), cands[i].n, cands[i].order, cands[i].cell),
    )
    taken: Optional[np.ndarray] = None
    chosen: List[_Cand] = []
    for i in order:
        c = cands[i]
        if taken is None:
            taken = c.rows.copy()
            chosen.append(c)
            continue
        if np.any(taken & c.rows):
            continue
        taken |= c.rows
        chosen.append(c)
    return chosen


class _SubCellStore:
    """Qualifying cells of every enumerated sub-schema, for reuse by its supersets."""

    def __init__(self, factory: _CandidateFactory, z: np.ndarray, spec: CenterSpec,
                 rows: Optional[np.ndarray]) -> None:
        self.factory = factory
        self.z = np.asarray(z, dtype=np.int64).ravel()
        self.spec = spec
        self.rows = rows
        self._cells: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}
        self._order: Dict[Tuple[int, ...], int] = {}

    def add(self, combo: Tuple[int, ...], codes: np.ndarray, n_cells: int) -> None:
        if self.rows is not None:
            codes = codes[self.rows]
        ids, n, k = _qualifying_cells(codes, n_cells, self.z, self.spec)
        self._order[combo] = len(self._order)
        if ids.size:
            self._cells[combo] = (ids, n, k, codes, n_cells)

    def get(self, combo: Tuple[int, ...]) -> List[_Cand]:
        hit = self._cells.get(combo)
        if hit is None:
            if combo not in self._order:
                codes, n_cells = self.factory.codes(combo)
                self.add(combo, codes, n_cells)
                hit = self._cells.get(combo)
            if hit is None:
                return []
        ids, n, k, codes, _ = hit
        order = self._order[combo]
        return [
            _Cand(combo, int(c), int(nn), int(kk), order, codes == c)
            for c, nn, kk in zip(ids.tolist(), n.tolist(), k.tolist())
        ]

    def has_cells(self, combo: Tuple[int, ...]) -> bool:
        if combo not in self._order:
            codes, n_cells = self.factory.codes(combo)
            self.add(combo, codes, n_cells)
        return combo in self._cells


def _sub_schemas(schema: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    for r in range(1, len(schema) + 1):
        yield from itertools.combinations(schema, r)


def _select(store: _SubCellStore, schema: Tuple[int, ...]) -> PartialSelection:
    pool: List[_Cand] = []
    for sub in _sub_schemas(schema):
        pool.extend(store.get(sub))
    z = store.z
    sel = PartialSelection(schema=schema, n_positive=int(z.sum()), n_samples=int(z.shape[0]))
    if not pool:
        return sel
    chosen = pack_disjoint(pool)
    sel.centres = [PartialCentre(c.sub_schema, c.cell, c.n, c.k) for c in chosen]
    sel.k_sel = int(sum(c.k for c in chosen))
    sel.n_sel = int(sum(c.n for c in chosen))
    return sel


def partial_centres_for_schema(
    X: np.ndarray,
    Z: np.ndarray,
    schema: Sequence[int],
    spec: Optional[CenterSpec] = None,
    positive_class: Optional[object] = None,
    max_d: int = MAX_BRANCH_D,
) -> PartialSelection:
    """
    The disjoint partial centres of ONE schema on all rows, under the same
    partitioning (capacity rule, level merging) and the same centre rule as
    `vsf.discover_branches`. `max_d` only sizes the family T of a
    `multiplicity="family"` spec.
    """
    spec = spec if spec is not None else CenterSpec()
    prepared = _prepare_search(X, Z, None, positive_class, spec, "presence")
    if prepared is None:
        raise ValueError("no feature columns")
    factory, z_binary, _ = prepared
    spec = resolve_center_spec(factory, spec, min(max_d, factory.n_features))
    store = _SubCellStore(factory, z_binary, spec, None)
    return _select(store, tuple(int(j) for j in schema))


# --------------------------------------------------------------------------
# Search over schemas
# --------------------------------------------------------------------------
def _search(
    factory: _CandidateFactory, z: np.ndarray, spec: CenterSpec, max_d: int,
    rows: Optional[np.ndarray] = None,
) -> Dict[int, PartialSelection]:
    """
    argmax over |S| = d of the disjoint-partial-centre coverage, for every
    d <= max_d, in the enumeration order of `_CandidateFactory.iter_candidates`
    (first candidate wins ties on the key (coverage, -n_centres, -mass)).
    `rows` restricts the scoring to those rows; partitions still come from
    all rows, as in `vsf.avr._exhaustive_search`.
    """
    z_scored = z if rows is None else z[rows]
    store = _SubCellStore(factory, z_scored, spec, rows)
    best: Dict[int, PartialSelection] = {}
    for combo, codes, n_cells in factory.iter_candidates(max_d):
        combo = tuple(combo)
        store.add(combo, codes, n_cells)
        d = len(combo)
        # Skip schemas none of whose sub-schemas has a centre (score 0).
        if not any(sub in store._cells for sub in _sub_schemas(combo)):
            if d not in best:
                best[d] = PartialSelection(schema=combo, n_positive=int(z_scored.sum()),
                                           n_samples=int(z_scored.shape[0]))
            continue
        sel = _select(store, combo)
        inc = best.get(d)
        if inc is None or sel.key > inc.key:
            best[d] = sel
    return best


def search_partial(
    X: np.ndarray,
    Z: np.ndarray,
    spec: Optional[CenterSpec] = None,
    positive_class: Optional[object] = None,
    max_d: int = MAX_BRANCH_D,
    prune_dependent: bool = False,
) -> Dict[int, PartialSelection]:
    """
    Independent branch discovery with partial centres: for each d <= max_d
    the schema whose disjoint partial centres cover the most positives.
    Same preparation, factory, family and centre rule as
    `vsf.discover_branches`; only the centres of a schema differ.

    For every d the result is at least as good as the full-grid branch on
    the same schema (the full cells are among the candidates), and the
    d-schema of the standard search scores at least its standard coverage.
    """
    spec = spec if spec is not None else CenterSpec()
    if max_d < 1 or max_d > MAX_BRANCH_D:
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
    prepared = _prepare_search(X, Z, None, positive_class, spec, "presence", prune_dependent)
    if prepared is None:
        return {}
    factory, z_binary, _ = prepared
    max_d = min(max_d, factory.n_features)
    spec = resolve_center_spec(factory, spec, max_d)
    return _search(factory, z_binary.astype(np.int64), spec, max_d)


# --------------------------------------------------------------------------
# Cross-validation (nested: the schema is searched inside each fold)
# --------------------------------------------------------------------------
@dataclass
class PartialCV:
    """Held-out coverage and purity of partial-centre branches, per d."""

    d: int
    coverage: np.ndarray          # per split
    purity: np.ndarray            # per split (nan when nothing selected)
    conditions: np.ndarray        # per split
    winners: List[Tuple[int, ...]]

    @property
    def mean_coverage(self) -> float:
        return float(self.coverage.mean()) if self.coverage.size else 0.0


def crossvalidate_partial(
    X: np.ndarray,
    Z: np.ndarray,
    spec: Optional[CenterSpec] = None,
    positive_class: Optional[object] = None,
    max_d: int = MAX_BRANCH_D,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: Optional[int] = 0,
) -> Dict[int, PartialCV]:
    """
    Nested cross-validation of `search_partial` with training-only encoding:
    on each training fold the partitions (capacity, level merging, family T)
    are fitted on the training rows, the search runs there, and the chosen
    centres are carried to the held-out rows with
    `vsf.avr.apply_fitted_partition`. Same folds as
    `vsf.selective.nested_crossvalidation(encoding="train")`, so the two are
    paired.
    """
    spec = spec if spec is not None else CenterSpec()
    prepared = _prepare_search(X, Z, None, positive_class, spec, "presence")
    if prepared is None:
        return {}
    factory, z_binary, _ = prepared
    z = z_binary.astype(np.int64)
    X_all = factory._raw.astype(np.int64, copy=False)
    max_d = min(max_d, factory.n_features)
    splits = stratified_repeated_kfold(z, n_splits, n_repeats, random_state)
    cov: Dict[int, List[float]] = {d: [] for d in range(1, max_d + 1)}
    pur: Dict[int, List[float]] = {d: [] for d in range(1, max_d + 1)}
    cond: Dict[int, List[int]] = {d: [] for d in range(1, max_d + 1)}
    win: Dict[int, List[Tuple[int, ...]]] = {d: [] for d in range(1, max_d + 1)}
    for train, test in splits:
        fit = _CandidateFactory(X_all[train], factory.bin_counts, int(train.size), ordered=factory.ordered)
        fold_spec = resolve_center_spec(fit, spec, max_d)
        best = _search(fit, z[train], fold_spec, max_d)
        z_test = z[test]
        n_pos = max(1, int(z_test.sum()))
        for d in range(1, max_d + 1):
            sel = best.get(d)
            inside = np.zeros(test.size, dtype=bool)
            if sel is not None and sel.centres:
                for sub in sorted({c.sub_schema for c in sel.centres}):
                    codes_fit, _ = fit.codes(sub)
                    codes_all, _ = apply_fitted_partition(fit, X_all, sub)
                    codes_test = codes_all[test]
                    for c in sel.centres:
                        if c.sub_schema != sub:
                            continue
                        first = int(np.flatnonzero(codes_fit == c.cell)[0])
                        image = int(codes_all[train[first]])
                        inside |= codes_test == image
            k_in = int(z_test[inside].sum())
            n_in = int(inside.sum())
            cov[d].append(k_in / n_pos)
            pur[d].append(k_in / n_in if n_in else float("nan"))
            cond[d].append(sel.conditions if sel is not None else 0)
            win[d].append(sel.schema if sel is not None else ())
    return {
        d: PartialCV(d, np.asarray(cov[d]), np.asarray(pur[d]), np.asarray(cond[d]), win[d])
        for d in range(1, max_d + 1)
    }
