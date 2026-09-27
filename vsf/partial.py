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

The product's full-grid search (`vsf.avr.discover_branches`) is NOT
changed by this module. `search_partial` is a separate entry point with the
same factory, the same centre rule and the same enumeration order, so its
numbers are comparable to the branches'. `discover_partial_branches` is what
the interactive application runs in its "partial centres" mode: the partial
search, the full-grid report of each winning schema (the lattice's cells),
the partial selection of every prefix view of the schema (the collapsed
views of Section 5.6) and a cross-validated coverage on the SAME folds as
the full-grid report, so the two are paired (`PartialBranch`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .avr import (
    MAX_BRANCH_D,
    BranchResult,
    Direction,
    _CandidateFactory,
    _prepare_search,
    _report_branches,
    apply_fitted_partition,
    resolve_center_spec,
)
from .centers import (
    MIN_POSITIVES_FOR_CV,
    CenterSpec,
    CVCoverage,
    PairedGain,
    coverage_score,
    min_successes_to_select,
    paired_gain,
    purity_bounds,
    select_dimensionality,
    stratified_repeated_kfold,
    summarize_cv,
    wilson_lower,
)

__all__ = [
    "PartialBranch",
    "PartialCentre",
    "PartialCentreStats",
    "PartialSelection",
    "PartialView",
    "crossvalidate_partial",
    "discover_partial_branches",
    "pack_disjoint",
    "partial_centres_for_schema",
    "search_partial",
    "select_partial_dimensionality",
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
    # Highest Wilson lower bound (at each sub-schema's own per-cell level)
    # over every occupied cell of every sub-schema. Set by the search ONLY
    # while no schema of this size selects anything; it then breaks the
    # all-zero tie the way `vsf.centers.coverage_score`'s fourth component
    # breaks it for the full grid - the schema that came closest to a
    # centre - instead of returning the first schema enumerated. It never
    # decides between schemas with a positive coverage, so no reported
    # coverage depends on it.
    closest_lower: float = 0.0

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
    level: float = 0.0       # per-cell level the centre rule was applied at


def _cell_table(codes: np.ndarray, n_cells: int, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(n, k) per cell of a dense partition."""
    table = np.bincount(codes * 2 + z, minlength=2 * n_cells).reshape(n_cells, 2)
    return table.sum(axis=1), table[:, 1]


def _qualifying_cells(
    codes: np.ndarray, n_cells: int, z: np.ndarray, spec: CenterSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    (cell ids, n, k, per-cell level) of the cells that are centres under
    `spec` - the search's own rule, at the level `spec` assigns to this
    partition (alpha / T under a family certificate, alpha / C_S per schema).
    """
    e = np.zeros(0, dtype=np.int64)
    if n_cells == 0:
        return e, e, e, float(spec.alpha)
    n_cell, k_cell = _cell_table(codes, n_cells, z)
    occupied = int(np.count_nonzero(n_cell > 0))
    if occupied == 0:
        return e, e, e, float(spec.alpha)
    level = float(spec.effective_alpha(occupied))
    k_min = min_successes_to_select(n_cell, spec, level)
    ids = np.flatnonzero((n_cell > 0) & (k_cell >= k_min))
    return ids, n_cell[ids], k_cell[ids], level


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
        self._cells: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]] = {}
        self._order: Dict[Tuple[int, ...], int] = {}
        self._lower: Dict[Tuple[int, ...], float] = {}

    def add(self, combo: Tuple[int, ...], codes: np.ndarray, n_cells: int) -> None:
        if self.rows is not None:
            codes = codes[self.rows]
        ids, n, k, level = _qualifying_cells(codes, n_cells, self.z, self.spec)
        self._order[combo] = len(self._order)
        if ids.size:
            self._cells[combo] = (ids, n, k, codes, n_cells, level)

    def closest_lower(self, combo: Tuple[int, ...]) -> float:
        """
        Highest Wilson lower bound over the occupied cells of `combo`'s
        partition, at the level `spec` assigns to it - `coverage_score`'s
        fourth component for one partition. Cached; computed only on demand
        (the search asks only while nothing of a size has been selected).
        """
        hit = self._lower.get(combo)
        if hit is not None:
            return hit
        codes, n_cells = self.factory.codes(combo)
        if self.rows is not None:
            codes = codes[self.rows]
        value = 0.0
        if n_cells > 0 and codes.size:
            n_cell, k_cell = _cell_table(codes, n_cells, self.z)
            occ = n_cell > 0
            if np.any(occ):
                level = float(self.spec.effective_alpha(int(np.count_nonzero(occ))))
                value = float(np.max(wilson_lower(k_cell[occ], n_cell[occ], level)))
        self._lower[combo] = value
        return value

    def get(self, combo: Tuple[int, ...]) -> List[_Cand]:
        hit = self._cells.get(combo)
        if hit is None:
            if combo not in self._order:
                codes, n_cells = self.factory.codes(combo)
                self.add(combo, codes, n_cells)
                hit = self._cells.get(combo)
            if hit is None:
                return []
        ids, n, k, codes, _, level = hit
        order = self._order[combo]
        return [
            _Cand(combo, int(c), int(nn), int(kk), order, codes == c, level)
            for c, nn, kk in zip(ids.tolist(), n.tolist(), k.tolist())
        ]

    def has_cells(self, combo: Tuple[int, ...]) -> bool:
        if combo not in self._order:
            codes, n_cells = self.factory.codes(combo)
            self.add(combo, codes, n_cells)
        return combo in self._cells


def _sub_schemas(schema: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    """
    Every non-empty sub-schema of `schema`, in canonical (ascending) column
    order - the order `_CandidateFactory.iter_candidates` enumerates, and
    hence the form in which the family T of Section 4.14 counts them (the
    capacity rule's merge tie-break depends on column order, so a sub-schema
    must be partitioned in this order to be the family's member).
    """
    canonical = tuple(sorted(schema))
    for r in range(1, len(canonical) + 1):
        yield from itertools.combinations(canonical, r)


def _select_candidates(store: _SubCellStore, schema: Tuple[int, ...]) -> Tuple[PartialSelection, List[_Cand]]:
    """The packed selection of `schema` and the chosen candidates (with their rows)."""
    pool: List[_Cand] = []
    for sub in _sub_schemas(schema):
        pool.extend(store.get(sub))
    z = store.z
    sel = PartialSelection(schema=tuple(schema), n_positive=int(z.sum()), n_samples=int(z.shape[0]))
    if not pool:
        return sel, []
    chosen = pack_disjoint(pool)
    sel.centres = [PartialCentre(c.sub_schema, c.cell, c.n, c.k) for c in chosen]
    sel.k_sel = int(sum(c.k for c in chosen))
    sel.n_sel = int(sum(c.n for c in chosen))
    return sel, chosen


def _select(store: _SubCellStore, schema: Tuple[int, ...]) -> PartialSelection:
    return _select_candidates(store, schema)[0]


def partial_centres_for_schema(
    X: np.ndarray,
    Z: np.ndarray,
    schema: Sequence[int],
    spec: Optional[CenterSpec] = None,
    positive_class: Optional[object] = None,
    max_d: int = MAX_BRANCH_D,
    direction: Direction = "presence",
    prune_dependent: bool = False,
) -> PartialSelection:
    """
    The disjoint partial centres of ONE schema on all rows, under the same
    partitioning (capacity rule, level merging) and the same centre rule as
    `vsf.discover_branches`. `max_d` only sizes the family T of a
    `multiplicity="family"` spec; `direction` and `prune_dependent` are those
    of the search the schema belongs to.
    """
    spec = spec if spec is not None else CenterSpec()
    prepared = _prepare_search(X, Z, None, positive_class, spec, direction, prune_dependent)
    if prepared is None:
        raise ValueError("no feature columns")
    factory, z_binary, _ = prepared
    spec = resolve_center_spec(factory, spec, min(max_d, factory.n_features))
    store = _SubCellStore(factory, z_binary, spec, None)
    return _select(store, tuple(int(j) for j in schema))


# --------------------------------------------------------------------------
# Search over schemas
# --------------------------------------------------------------------------
def _closest_lower(store: _SubCellStore, schema: Tuple[int, ...]) -> float:
    return max((store.closest_lower(sub) for sub in _sub_schemas(schema)), default=0.0)


def _search(
    factory: _CandidateFactory, z: np.ndarray, spec: CenterSpec, max_d: int,
    rows: Optional[np.ndarray] = None,
) -> Dict[int, PartialSelection]:
    """
    argmax over |S| = d of the disjoint-partial-centre coverage, for every
    d <= max_d, in the enumeration order of `_CandidateFactory.iter_candidates`
    (first candidate wins ties on the key (coverage, -n_centres, -mass)).
    While no schema of size d selects anything, the schema whose sub-schema
    cells come closest to the centre rule is kept instead of the first one
    enumerated (`PartialSelection.closest_lower`; the fourth component of
    `vsf.centers.coverage_score`). `rows` restricts the scoring to those
    rows; partitions still come from all rows, as in
    `vsf.avr._exhaustive_search`.
    """
    z_scored = z if rows is None else z[rows]
    store = _SubCellStore(factory, z_scored, spec, rows)
    best: Dict[int, PartialSelection] = {}
    for combo, codes, n_cells in factory.iter_candidates(max_d):
        combo = tuple(combo)
        store.add(combo, codes, n_cells)
        d = len(combo)
        inc = best.get(d)
        # Skip the packing for schemas none of whose sub-schemas has a centre (score 0).
        if not any(sub in store._cells for sub in _sub_schemas(combo)):
            if inc is None or not inc.centres:
                lower = _closest_lower(store, combo)
                if inc is None or lower > inc.closest_lower:
                    best[d] = PartialSelection(
                        schema=combo, n_positive=int(z_scored.sum()),
                        n_samples=int(z_scored.shape[0]), closest_lower=lower,
                    )
            continue
        sel = _select(store, combo)
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
    direction: Direction = "presence",
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
    prepared = _prepare_search(X, Z, None, positive_class, spec, direction, prune_dependent)
    if prepared is None:
        return {}
    factory, z_binary, _ = prepared
    max_d = min(max_d, factory.n_features)
    spec = resolve_center_spec(factory, spec, max_d)
    return _search(factory, z_binary.astype(np.int64), spec, max_d)


# --------------------------------------------------------------------------
# Reporting: what the interactive application shows for a partial branch
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PartialCentreStats:
    """
    One selected partial centre with everything the display says about it.

    `purity_lower` / `purity_upper` are the Clopper-Pearson (or the spec's
    method) bounds at `level`, the per-cell level at which the centre rule
    was applied to its sub-schema's partition - alpha / T under a family
    certificate, alpha / C_{S'} under the per-schema correction. `rows` is
    the centre's row mask over ALL rows (not serialised; the renderer reads
    the fixed values off it).
    """

    sub_schema: Tuple[int, ...]
    cell: int
    n: int
    k: int
    level: float
    purity_lower: float
    purity_upper: float
    rows: np.ndarray = field(compare=False, repr=False)

    @property
    def cost(self) -> int:
        return len(self.sub_schema)

    @property
    def purity(self) -> float:
        return self.k / self.n if self.n else 0.0


@dataclass
class PartialView:
    """
    The partial selection of one view of a branch: the branch's first
    `len(schema)` axes (the prefix a collapsed view of Section 5.6 shows),
    with the full-grid selection of the same axes under the same rule and
    the same partition for comparison.
    """

    schema: Tuple[int, ...]
    selection: PartialSelection
    centres: List[PartialCentreStats]
    full_grid_k: int
    full_grid_n: int
    full_grid_n_centers: int

    @property
    def full_grid_coverage(self) -> float:
        n_pos = self.selection.n_positive
        return self.full_grid_k / n_pos if n_pos else 0.0

    @property
    def full_grid_conditions(self) -> int:
        return self.full_grid_n_centers * len(self.schema)

    @property
    def axes_used(self) -> Tuple[int, ...]:
        """The schema's columns that at least one selected centre fixes, in schema order."""
        used = {j for c in self.centres for j in c.sub_schema}
        return tuple(j for j in self.schema if j in used)


@dataclass
class PartialBranch:
    """
    One partial-centre branch: the schema the partial search chose for `d`,
    its selection in every prefix view (`views[p]`, p = 1..d; `views[d]` is
    the branch's own), and its out-of-sample coverage.

    `coverage_cv` is the partial selection made on each training fold and
    measured on the held-out fold, schema fixed, partition from all rows -
    the protocol of `vsf.centers.crossvalidated_coverage`, on the SAME folds
    (same indicator, splits, repeats and seed) as the full-grid report of
    the same schema (`full_grid_cv`, i.e. `BranchResult.centers.coverage_cv`),
    so `gain_vs_full` is a paired Nadeau-Bengio comparison. Like the
    full-grid figure it is not a nested estimate: the schema was chosen on
    all rows. `crossvalidate_partial` is the nested one.
    """

    d: int
    schema: Tuple[int, ...]
    views: Dict[int, PartialView]
    coverage_cv: Optional[CVCoverage]
    full_grid_cv: Optional[CVCoverage]
    gain_vs_full: Optional[PairedGain]
    cv_undetermined_reason: Optional[str] = None

    @property
    def own(self) -> PartialView:
        return self.views[self.d]


def _centre_stats(chosen: Sequence[_Cand], spec: CenterSpec, n_rows: int) -> List[PartialCentreStats]:
    out: List[PartialCentreStats] = []
    for c in chosen:
        lower, upper = purity_bounds(np.array([c.k]), np.array([c.n]), c.level, spec.method)
        rows = c.rows
        if rows.shape[0] != n_rows:  # pragma: no cover - the store scores all rows here
            raise ValueError("centre rows are not over all rows")
        out.append(PartialCentreStats(
            sub_schema=tuple(int(j) for j in c.sub_schema), cell=int(c.cell), n=int(c.n), k=int(c.k),
            level=float(c.level), purity_lower=float(lower[0]), purity_upper=float(upper[0]), rows=rows.copy(),
        ))
    return out


def _view(store: _SubCellStore, schema: Tuple[int, ...]) -> PartialView:
    sel, chosen = _select_candidates(store, schema)
    canonical = tuple(sorted(schema))
    full = store._cells.get(canonical)
    if full is None and canonical not in store._order:
        store.has_cells(canonical)
        full = store._cells.get(canonical)
    fk = int(full[2].sum()) if full is not None else 0
    fn = int(full[1].sum()) if full is not None else 0
    fc = int(full[0].size) if full is not None else 0
    return PartialView(
        schema=tuple(schema), selection=sel,
        centres=_centre_stats(chosen, store.spec, int(store.z.shape[0])),
        full_grid_k=fk, full_grid_n=fn, full_grid_n_centers=fc,
    )


def _fixed_schema_cv(
    factory: _CandidateFactory,
    z: np.ndarray,
    spec: CenterSpec,
    schema: Tuple[int, ...],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_splits: int,
    n_repeats: int,
) -> CVCoverage:
    """
    Held-out coverage of the partial selection of a FIXED schema: on each
    split the candidates are qualified and packed on the training rows, and
    the chosen cells (codes from all rows, as in
    `vsf.centers.crossvalidated_coverage`) are counted on the held-out rows.
    """
    codes_of: Dict[Tuple[int, ...], np.ndarray] = {}
    cov = np.zeros(len(splits), dtype=np.float64)
    pur = np.full(len(splits), np.nan, dtype=np.float64)
    for i, (train, test) in enumerate(splits):
        store = _SubCellStore(factory, z[train], spec, train)
        sel = _select(store, schema)
        inside = np.zeros(test.size, dtype=bool)
        for c in sel.centres:
            codes = codes_of.get(c.sub_schema)
            if codes is None:
                codes = factory.codes(c.sub_schema)[0]
                codes_of[c.sub_schema] = codes
            inside |= codes[test] == c.cell
        z_test = z[test]
        pos = int(z_test.sum())
        k_in = int(z_test[inside].sum())
        n_in = int(inside.sum())
        cov[i] = (k_in / pos) if pos > 0 else 0.0
        if n_in > 0:
            pur[i] = k_in / n_in
    return summarize_cv(cov, pur, n_splits, n_repeats)


def discover_partial_branches(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    center_spec: Optional[CenterSpec] = None,
    positive_class: Optional[object] = None,
    max_d: int = MAX_BRANCH_D,
    direction: Direction = "presence",
    prune_dependent: bool = False,
    features: Optional[Sequence[int]] = None,
    family_max_d: int = MAX_BRANCH_D,
    random_state: Optional[int] = 0,
    cv_splits: int = 5,
    cv_repeats: int = 5,
) -> Tuple[Dict[int, BranchResult], Dict[int, PartialBranch]]:
    """
    Branch discovery with partial centres, as the interactive application
    needs it: `(branches, partial)`, both keyed by d.

    * The schema of each d is the partial search's winner (`_search`), or
      `features` alone when an explicit schema is given (opened from the
      landscape; `family_max_d` then sizes the family, as in
      `vsf.avr.report_schema`).
    * `branches[d]` is the ordinary full-grid report of that schema
      (`vsf.avr._report_branches` with no permutations): the lattice's cells,
      their certificate and the full-grid cross-validated coverage. No
      permutation p-value is computed: it would test the full-grid statistic
      of a schema chosen by another statistic.
    * `partial[d]` is the partial selection of every prefix view of the
      schema and its paired cross-validated coverage (`PartialBranch`).

    One factory, one resolved spec and one indicator serve all of it, so the
    lattice, the partial centres and both CV estimates refer to the same
    partition, the same family T and the same folds.
    """
    spec = center_spec if center_spec is not None else CenterSpec()
    if cv_splits < 2 or cv_repeats < 0:
        raise ValueError("cv_splits must be >= 2 and cv_repeats >= 0")
    prepared = _prepare_search(X, Z, feature_names, positive_class, spec, direction, prune_dependent)
    if prepared is None:
        return {}, {}
    factory, z_binary, names = prepared
    z = z_binary.astype(np.int64)

    if features is None:
        if max_d < 1 or max_d > MAX_BRANCH_D:
            raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
        eff_max_d = min(max_d, factory.n_features)
        spec = resolve_center_spec(factory, spec, eff_max_d)
        winners = {d: tuple(sel.schema) for d, sel in _search(factory, z, spec, eff_max_d).items()}
    else:
        combo = tuple(int(j) for j in features)
        if (not combo or len(set(combo)) != len(combo) or len(combo) > MAX_BRANCH_D
                or any(j < 0 or j >= factory.n_features for j in combo)):
            raise ValueError(
                f"features must be 1 to {MAX_BRANCH_D} distinct column indices in "
                f"[0, {factory.n_features}), got {list(features)}"
            )
        if not (1 <= family_max_d <= MAX_BRANCH_D):
            raise ValueError(f"family_max_d must be in [1, {MAX_BRANCH_D}], got {family_max_d}")
        eff_max_d = len(combo)
        spec = resolve_center_spec(factory, spec, min(family_max_d, factory.n_features))
        winners = {len(combo): combo}

    ranked = {}
    for d, combo in winners.items():
        codes, n_cells = factory.codes(combo)
        ranked[d] = (coverage_score(z_binary, codes, n_cells, spec), combo)
    branches = _report_branches(
        factory, z_binary, ranked, names, spec, max(eff_max_d, 1),
        random_state=random_state, progress=None,
        n_permutations_centers=0, n_permutations_familywise_coverage=0,
        cv_splits=cv_splits, cv_repeats=cv_repeats,
    )

    store = _SubCellStore(factory, z, spec, None)
    n_positive = int(z.sum())
    reason: Optional[str] = None
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    if n_positive < MIN_POSITIVES_FOR_CV:
        reason = (
            f"only {n_positive} samples carry the target value; {MIN_POSITIVES_FOR_CV} "
            "are required before an out-of-sample coverage can be estimated"
        )
    elif cv_repeats > 0:
        # `vsf.centers.center_report` draws its folds from the same call.
        splits = stratified_repeated_kfold(z_binary.astype(np.int8), cv_splits, cv_repeats, random_state)

    partial: Dict[int, PartialBranch] = {}
    for d, br in branches.items():
        schema = tuple(int(j) for j in br.selected_features)
        views = {p: _view(store, schema[:p]) for p in range(1, d + 1)}
        cv: Optional[CVCoverage] = None
        full_cv = br.centers.coverage_cv
        gain: Optional[PairedGain] = None
        if splits is not None:
            cv = _fixed_schema_cv(factory, z, spec, schema, splits, cv_splits, cv_repeats)
            if full_cv is not None and len(full_cv.per_split) == len(cv.per_split):
                gain = paired_gain(cv, full_cv)
        partial[d] = PartialBranch(
            d=d, schema=schema, views=views, coverage_cv=cv, full_grid_cv=full_cv,
            gain_vs_full=gain, cv_undetermined_reason=reason,
        )
    return branches, partial


def select_partial_dimensionality(partial: Dict[int, PartialBranch], t_threshold: float = 2.0) -> Optional[int]:
    """
    `vsf.avr.select_branch_dimensionality` for partial branches: the smallest
    d whose held-out partial coverage beats every lower d by more than
    `t_threshold` paired standard errors (`vsf.centers.select_dimensionality`).
    None when no d has positive held-out coverage or no CV was possible.
    """
    cv_by_d = {d: pb.coverage_cv for d, pb in partial.items() if pb.coverage_cv is not None}
    return select_dimensionality(cv_by_d, t_threshold=t_threshold)


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
    direction: Direction = "presence",
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
    prepared = _prepare_search(X, Z, None, positive_class, spec, direction)
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
