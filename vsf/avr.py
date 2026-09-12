"""
VSF Independent Branch Discovery (IBD).

For a chosen target column and a resolved positive value, independently
find, for every dimensionality d in {1, 2, 3, 4}, the single feature subset
of exactly that size whose discrete-centre partition concentrates the MOST
of that value inside certified (green) centres - by honest, exhaustive
enumeration of every C(M, d) combination. See
Project_Master_Document.md Sections 4.1-4.6 for the formal specification
this module implements.

The four resulting "branches" are NOT required to be nested: the winning
pair for d=2 need not contain the winning singleton at d=1. That is
intentional. Coverage is not submodular in the feature set (Krause &
Guestrin, 2005), so a greedy nested chain can provably miss a synergistic
combination that an independent per-d search finds - `tests/test_avr.py`'s
XOR fixture is a running instance.

Ranking
-------
There is exactly one ranking key: `vsf.centers.coverage_score`, the
lexicographic tuple (coverage, -n_centers, -mass, best per-cell lower
bound). `discover_branches` REQUIRES a resolvable positive class -
explicitly via `positive_class`, or automatically for a two-valued raw
target - and raises otherwise rather than substituting a different
statistic for a different question.

What is still not claimed
-------------------------
* Every estimate here is IN-SAMPLE except `centers.coverage_cv`
  (`select_branch_dimensionality`'s basis), which is the one genuinely
  out-of-sample number this module reports.
* `coverage_p_value_familywise` is the look-elsewhere-corrected number for a
  DISCOVERED branch (`centers.coverage_p_value` alone is not, for exactly
  the reason a branch selected as an argmax over the whole candidate family
  needs a family-aware null). It requires
  `n_permutations_familywise_coverage > 0` and costs B times a full search,
  so it is off by default and any published claim must set it.
* No FDR control is applied WITHIN a single search. It is applied across
  targets by the Global Pattern Scan (`vsf.server`), which is the family in
  which multiplicity actually accumulates for that feature.
* Features are CATEGORIES. Every column is encoded by its distinct values
  (`vsf.pmd`); nothing here bins a continuous variable, and no
  information-theoretic quantity is computed anywhere on this path.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .centers import (
    CenterReport,
    CenterSpec,
    CVCoverage,
    center_report,
    center_summary,
    coverage_score,
    familywise_max_coverage_null,
    min_successes_to_select,
    wilson_lower,
)
from .centers import select_dimensionality as _select_dimensionality
from .metrics import cell_codes, dense_codes_from_flat
from .screen import exact_dependencies
from .pmd import (
    coarsen_column,
    discretize_dataset,
    grid_capacity,
    level_frequency_order,
    merge_state,
    ordered_columns,
)

# Hard ceiling on branch dimensionality. Matches the display's actual spatial
# encoding (3 coordinate axes + 1 time/frame axis, see vsf.vis's module
# docstring and Project_Master_Document.md Section 1.3) - there is no 5th/6th
# channel to search for, unlike v1.0's d_max=7 (which searched further than
# the display could ever show).
MAX_BRANCH_D = 4

#: Largest number of combinations whose pre-coarsening occupancy is kept
#: while pruning dependent candidates (`_CandidateFactory._is_redundant`).
#: Beyond it the rule prunes less, never more.
_OCCUPANCY_CACHE_MAX: int = 1_000_000

#: Default number of permutation replicates for a branch's familywise
#: coverage p-value (`coverage_p_value_familywise`). 999 gives a resolution
#: of 1e-3, the coarsest that can still express alpha = 0.01 with a margin.
#: Off by default (`n_permutations_familywise_coverage=0` in
#: `discover_branches`) because it costs B times the whole candidate family.
DEFAULT_N_PERMUTATIONS = 999

#: Which side of the target a search localises. `"presence"` (the default,
#: and everything the framework did before 2026-09) certifies cells that are
#: at least `tau` pure in the positive class and reports how much of that
#: class they capture. `"absence"` certifies cells that are at least `tau`
#: pure in ITS COMPLEMENT - cells where the positive class is almost absent
#: - and reports how much of the complement they capture, i.e. which part of
#: the data can be certified free of the value. Same search, same
#: certificate, same cross-validation and nulls, applied to the inverted
#: indicator `1 - z`: absence(X) is presence(not X) by construction (pinned
#: in `tests/test_absence.py`). It is a separate QUESTION with its own
#: headline (the mass of certified X-free cells), its own base rate
#: (1 - p_0) and its own legend, not a separate algorithm.
Direction = Literal["presence", "absence"]


def base_rate_reason(spec: CenterSpec, prevalence: float, direction: Direction) -> Optional[str]:
    """
    Why a search at `spec.tau` is void for an indicator with this
    `prevalence`, or None if it is well-posed.

    A purity threshold is only meaningful above the indicator's base rate:
    at tau <= p_0 the trivial partition (one cell) is already a certified
    centre with coverage 1, every refinement inherits that, and the display
    turns uniformly green without saying anything. For a value that occupies
    90 % of the rows a presence search therefore needs tau > 0.90, while an
    absence search of the same value works against a base rate of 0.10 and
    is well-posed at the usual tau. The check is the same inequality in both
    directions, on the base rate of the indicator actually searched.
    """
    if not (0.0 <= prevalence <= 1.0):
        raise ValueError(f"prevalence must be in [0, 1], got {prevalence}")
    if spec.tau > prevalence:
        return None
    what = "the positive class" if direction == "presence" else "the complement of the positive class"
    return (
        f"tau = {spec.tau:.4g} is not above the base rate of {what} "
        f"({prevalence:.4g}): every cell of the trivial partition already "
        "reaches the purity floor, so certification cannot distinguish "
        "signal from the base rate. Raise tau above the base rate"
        + (" or search for the value's absence instead." if direction == "presence" else ".")
    )


def _oriented_indicator(z_binary: np.ndarray, direction: Direction) -> np.ndarray:
    """The 0/1 indicator the search runs on: `z` for presence, `1 - z` for absence."""
    if direction == "presence":
        return z_binary.astype(np.int8)
    if direction == "absence":
        return (1 - z_binary).astype(np.int8)
    raise ValueError(f"direction must be 'presence' or 'absence', got {direction!r}")


@dataclass
class BranchResult:
    """
    One independently-discovered feature subset for a single dimensionality
    `d`, ranked by how much of the resolved positive class it concentrates
    inside certified discrete centres.

    Field semantics
    ---------------
    `centers`
        The v2.2 reporting layer (`vsf.centers.CenterReport`) for THIS
        branch's full cell partition: how many cells are certified to be at
        least `tau` pure in the target value, what share of all target-value
        samples they contain (`coverage`), how clean they are, and what
        survives cross-validation and the permutation null. This is the
        statistic the search itself optimises (v2.3: never `None` - a
        positive class is required to call `discover_branches` at all).
    `coverage_by_prefix_d` / `n_centers_by_prefix_d` / `purity_by_prefix_d`
        The centre statistics under this branch's own first-k axes -
        `coverage_by_prefix_d[k-1]` is what THIS branch's own coverage would
        read if only its first k selected axes were displayed. In-sample
        only (no cross-validation per prefix - that is 25 refits per prefix
        per branch and does not belong on an interactive path); use
        `select_branch_dimensionality` for the out-of-sample answer. NOT a
        second independent search: `discover_branches` finds, for every d,
        the BEST-scoring subset of that size, independently (branches need
        not nest - see the module docstring); these lists instead fix THIS
        branch's own `selected_features` ordering and report what a
        within-branch dimensionality collapse to k < d axes would show
        (Project_Master_Document.md Section 5.6, case 2) - showing the
        full-branch value instead while collapsed silently overstates the
        concentration carried by those k axes alone. The final entry always
        equals `centers.coverage` / `centers.n_centers` / `centers.purity_pooled`
        exactly. Empty for a `BranchResult` built by hand rather than through
        `discover_branches` (e.g. in tests); callers must treat that as "no
        per-view breakdown available" and fall back to the scalars.
    `coverage_p_value_familywise`
        Look-elsewhere-corrected p-value of `centers.coverage` against
        `vsf.centers.familywise_max_coverage_null`. This, not
        `centers.coverage_p_value`, is the number a paper quotes for a
        DISCOVERED branch. `None` unless
        `n_permutations_familywise_coverage > 0` was passed to
        `discover_branches`.
    """

    d: int
    selected_features: List[int]
    selected_feature_names: List[str]
    centers: CenterReport
    coverage_by_prefix_d: List[float] = field(default_factory=list)
    n_centers_by_prefix_d: List[int] = field(default_factory=list)
    purity_by_prefix_d: List[float] = field(default_factory=list)
    coverage_p_value_familywise: Optional[float] = None

    def has_certified_centers(self, alpha: float = 0.01) -> bool:
        """
        Gate for "this branch produces the deliverable the display promises".

        A branch can win the coverage search and still certify no centre at
        all when nothing in it clears `tau` - `centers.n_centers == 0`, so
        `centers.coverage == 0.0` by construction. Reporting such a branch as
        a finding is the failure mode v2.2's certificate layer exists to
        stop.
        """
        if self.centers.n_centers == 0:
            return False
        p = (
            self.coverage_p_value_familywise
            if self.coverage_p_value_familywise is not None
            else self.centers.coverage_p_value
        )
        return p is None or p <= alpha


def _discretize_target(Z_arr: np.ndarray) -> np.ndarray:
    """
    Encodes the target vector as integer category codes, one per distinct
    value, in sorted distinct-value order - the same rule
    `vsf.pmd.discretize_feature` applies to a feature column, and for the
    same reason: the target of a categorical framework is a set of
    categories, whatever dtype it arrives in.

    A numeric target used to take a separate branch here that binned it
    when it had more than 20 distinct values. That branch is gone with the
    rest of the continuous-feature machinery (see `vsf.pmd`), so a numeric
    target now contributes exactly the values it holds. Note that this
    function only produces the CODES; which of them counts as positive is
    `_resolve_positive_indicator`'s job, and a target with more than two
    distinct values still requires an explicit `positive_class`.
    """
    _, Z_discrete = np.unique(Z_arr, return_inverse=True)
    return np.asarray(Z_discrete).ravel().astype(int)


class _CandidateFactory:
    """
    Single source of truth for how a feature subset becomes a cell partition,
    shared by the exhaustive search, the per-branch prefix breakdown, the
    landscape and the familywise null - all of them must score the IDENTICAL
    partition or their numbers are not comparable.

    The partition of a subset S is the dense joint category code of its
    columns, after the Grid Capacity rule (Project_Master_Document.md
    Section 2.3): if the joint partition occupies more than
    `grid_capacity(N)` cells, levels are merged exactly as
    `vsf.pmd.adaptively_coarsen_bins` merges them - the widest column loses
    its rarest kept level into an "other" group, one level at a time, until
    the occupancy fits. `codes(combo)` returns exactly what
    `cell_codes(adaptively_coarsen_bins(X[:, combo], N))` returns (pinned in
    `tests/test_fastpaths.py`).

    What this class adds is the bookkeeping that makes producing that
    partition cheap enough to do C(M, 1) + ... + C(M, 4) times:

    * the mixed-radix joint code of S is `code(S \\ {j}) * r_j + x_j`, so an
      enumeration that fixes a prefix and varies the last column reuses the
      prefix's code (`iter_candidates`);
    * dense relabelling goes through `vsf.metrics.dense_codes_from_flat`
      (an occupancy count, not a sort), and the same count is the capacity
      check, so a combination that fits costs nothing beyond its own code;
    * each column's frequency order and each (column, level count)
      coarsening are computed once and cached.
    """

    def __init__(
        self,
        X_discrete: np.ndarray,
        bin_counts: Sequence[int],
        n_samples: int,
        ordered: Optional[Sequence[bool]] = None,
        determined_by: Optional[Dict[int, Sequence[int]]] = None,
    ) -> None:
        X = np.asarray(X_discrete)
        if X.ndim != 2:
            raise ValueError(f"X_discrete must be 2-D, got ndim={X.ndim}")
        self.n_samples = int(n_samples)
        self.n_features = int(X.shape[1])
        self.bin_counts = [int(b) for b in bin_counts]
        self.capacity = grid_capacity(self.n_samples)
        # Which columns carry an order (numeric): merged by adjacency when
        # coarsened; the rest by frequency (`vsf.pmd.coarsen_column`).
        self.ordered = [bool(v) for v in ordered] if ordered is not None else [False] * self.n_features
        if len(self.ordered) != self.n_features:
            raise ValueError("ordered must have one flag per feature column")
        self._raw = X
        # Shifted int64 columns and their radices, exactly as
        # `vsf.metrics._as_cell_codes` forms the mixed-radix code.
        self._raw_shifted: List[np.ndarray] = []
        self._raw_radix: List[int] = []
        self._orders: List[np.ndarray] = []
        self._levels: List[int] = []
        for j in range(self.n_features):
            col = X[:, j].astype(np.int64, copy=False)
            lo = int(col.min()) if col.size else 0
            hi = int(col.max()) if col.size else 0
            self._raw_shifted.append(col - lo)
            self._raw_radix.append(hi - lo + 1)
            order = level_frequency_order(col) if col.size else np.zeros(0, dtype=np.int64)
            self._orders.append(order)
            self._levels.append(int(order.shape[0]))
        self._coarse: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}
        # Exact functional dependencies between columns (`vsf.screen.
        # exact_dependencies`): `determined_by[y]` are the columns that fix
        # y's value. When given, `iter_candidates` skips the candidates they
        # make redundant - see that method.
        self.determined_by: Dict[int, Tuple[int, ...]] = {
            int(j): tuple(int(i) for i in v) for j, v in (determined_by or {}).items()
        }
        # Occupancy of already-enumerated combinations BEFORE any capacity
        # coarsening, the quantity the pruning rule needs. Bounded; beyond
        # the bound the rule simply prunes less (never more).
        self._occupancy: Dict[Tuple[int, ...], int] = {}
        self.n_pruned = 0

    # -- coarsening ---------------------------------------------------------
    def _coarse_column(self, j: int, k: int) -> Tuple[np.ndarray, int]:
        """Column j reduced to k levels (`vsf.pmd.coarsen_column`), shifted, with its radix."""
        key = (j, k)
        hit = self._coarse.get(key)
        if hit is None:
            col = coarsen_column(
                self._raw[:, j], k, self._orders[j], ordered=self.ordered[j]
            ).astype(np.int64, copy=False)
            lo = int(col.min()) if col.size else 0
            hi = int(col.max()) if col.size else 0
            hit = (col - lo, hi - lo + 1)
            self._coarse[key] = hit
        return hit

    @staticmethod
    def _flatten(columns: Sequence[Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, int]:
        flat, total = None, 1
        for col, radix in columns:
            flat = col if flat is None else flat * radix + col
            total *= radix
        assert flat is not None
        return flat, total

    def _dense(self, flat: np.ndarray, total: int) -> Tuple[np.ndarray, int]:
        if total >= (1 << 62):  # pragma: no cover - >2^62 nominal cells
            _, codes = np.unique(flat, return_inverse=True)
            codes = codes.astype(np.int64).ravel()
            return codes, int(codes.max()) + 1
        dense = dense_codes_from_flat(flat, total)
        return dense, int(dense.max()) + 1

    def _coarsened_codes(self, combo: Tuple[int, ...]) -> Tuple[np.ndarray, int]:
        """
        The partition of an over-capacity combination: the first state of
        the greedy merge sequence (`vsf.pmd.merge_state` - widest column
        first, ties lowest index, one level at a time) whose occupancy fits
        the capacity, found by bisection over the step count exactly as
        `vsf.pmd.coarsen_to_capacity` finds it.
        """
        k0 = [self._levels[j] for j in combo]

        def build(k: List[int]) -> Tuple[np.ndarray, int]:
            cols = [
                self._coarse_column(j, k[i]) if k[i] < k0[i]
                else (self._raw_shifted[j], self._raw_radix[j])
                for i, j in enumerate(combo)
            ]
            return self._dense(*self._flatten(cols))

        total_steps = sum(max(0, v - 1) for v in k0)
        last = build(merge_state(k0, total_steps))
        if last[1] > self.capacity:
            return last
        lo, hi = 0, total_steps
        best = last
        while hi - lo > 1:
            mid = (lo + hi) // 2
            dense, n_cells = build(merge_state(k0, mid))
            if n_cells <= self.capacity:
                hi, best = mid, (dense, n_cells)
            else:
                lo = mid
        return best

    # -- partitions ---------------------------------------------------------
    def codes(self, combo: Tuple[int, ...]) -> Tuple[np.ndarray, int]:
        """Dense joint cell codes and cell count for one feature subset."""
        if self.n_samples == 0:
            return np.zeros(0, dtype=np.int64), 0
        flat, total = self._flatten(
            [(self._raw_shifted[j], self._raw_radix[j]) for j in combo]
        )
        dense, n_cells = self._dense(flat, total)
        if n_cells <= self.capacity:
            return dense, n_cells
        return self._coarsened_codes(tuple(combo))

    def _is_redundant(self, combo: Tuple[int, ...], max_d: int) -> bool:
        """
        True when this combination is a RENAMING of a smaller one, provably:
        it holds a column y and a column x that determines y exactly, and the
        combination without y was enumerated with an occupancy (before
        coarsening) within the grid capacity.

        Why that second condition is needed. Because y = f(x), refining by y
        splits no cell, so the joint partition of S and of S u {y} are the
        same SET of cells - but only before the capacity rule runs. Both have
        the same occupancy, so if it fits the capacity neither is coarsened
        and the two partitions are identical; if it does not, the two are
        coarsened from different nominal level counts and can differ, so the
        candidate is kept and scored.

        When the smaller combination's occupancy is not in the cache (it was
        itself pruned, or the cache is full), the rule declines to prune. The
        error is therefore one-sided: a pruned candidate is always an exact
        duplicate of one that is scored.
        """
        if not self.determined_by or len(combo) < 2:
            return False
        members = set(combo)
        for y in combo:
            for x in self.determined_by.get(y, ()):  # x -> y exactly
                if x == y or x not in members:
                    continue
                base = tuple(q for q in combo if q != y)
                occ = self._occupancy.get(base)
                if occ is not None and occ <= self.capacity:
                    return True
        return False

    def iter_candidates(
        self, max_d: int
    ) -> Iterator[Tuple[Tuple[int, ...], np.ndarray, int]]:
        """
        Every (combo, cell codes, n_cells) of the exhaustive family, in
        `itertools.combinations` order for d = 1, ..., max_d - the order the
        search's first-wins tie rule is defined against.

        With `determined_by` given, combinations that `_is_redundant` proves
        to be renamings of a smaller one are skipped (counted in
        `n_pruned`): they carry the identical partition, hence the identical
        coverage, K and mass, at a higher dimensionality.
        """
        m = self.n_features
        if self.n_samples == 0:
            for d in range(1, max_d + 1):
                for combo in itertools.combinations(range(m), d):
                    yield combo, np.zeros(0, dtype=np.int64), 0
            return
        cache = bool(self.determined_by)
        for d in range(1, max_d + 1):
            for prefix in itertools.combinations(range(m), d - 1):
                base: Optional[Tuple[np.ndarray, int]] = (
                    self._flatten([(self._raw_shifted[q], self._raw_radix[q]) for q in prefix])
                    if prefix else None
                )
                start = prefix[-1] + 1 if prefix else 0
                for j in range(start, m):
                    combo = prefix + (j,)
                    if self._is_redundant(combo, max_d):
                        self.n_pruned += 1
                        continue
                    col, radix = self._raw_shifted[j], self._raw_radix[j]
                    if base is None:
                        flat, total = col, radix
                    else:
                        flat, total = base[0] * radix + col, base[1] * radix
                    dense, n_cells = self._dense(flat, total)
                    if cache and d < max_d and len(self._occupancy) < _OCCUPANCY_CACHE_MAX:
                        self._occupancy[combo] = n_cells
                    if n_cells <= self.capacity:
                        yield combo, dense, n_cells
                    else:
                        yield (combo,) + self._coarsened_codes(combo)


def _subset_codes(
    X_discrete: np.ndarray,
    bin_counts: Sequence[int],
    n_samples: int,
    combo: Tuple[int, ...],
    ordered: Optional[Sequence[bool]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Dense joint cell codes for one feature subset, applying the Grid Capacity
    rule exactly as the search does. Thin wrapper over
    `_CandidateFactory.codes` for callers holding no factory.
    """
    return _CandidateFactory(X_discrete, bin_counts, n_samples, ordered=ordered).codes(tuple(combo))


def _iter_candidates(
    X_discrete: np.ndarray,
    bin_counts: Sequence[int],
    n_samples: int,
    max_d: int,
    ordered: Optional[Sequence[bool]] = None,
) -> Iterator[Tuple[Tuple[int, ...], np.ndarray, int]]:
    """Every (combo, cell codes, n_cells) the exhaustive search will score."""
    return _CandidateFactory(X_discrete, bin_counts, n_samples, ordered=ordered).iter_candidates(max_d)


#: The lexicographic ranking key of `vsf.centers.coverage_score`, paired
#: with the combination it belongs to.
_Ranked = Tuple[Tuple[float, float, float, float], Tuple[int, ...]]


class Landscape:
    """
    Every candidate the exhaustive search scored, with the three numbers
    its ranking key is made of: the "solution landscape" of one search
    (Project_Master_Document.md Section 4.9). The winner is one point of
    it; the rest is what the display never showed before - how many other
    schemas reach a given coverage, at how many centres, and how many
    certify nothing at all.

    Records `(d, features, k_sel, n_centers, n_sel)` per candidate, appended
    by `_exhaustive_search` through `record`. `bins` aggregates them into
    the 10 x 10 lattice the frontend draws; `cell` lists the schemas of one
    lattice cell. Both are pure functions of the recorded arrays, so they
    can be evaluated lazily and repeatedly on a cached instance.
    """

    N_BINS: int = 10

    def __init__(
        self, n_samples: int, n_positive: int, direction: Direction,
        feature_names: Sequence[str],
    ) -> None:
        self.n_samples = int(n_samples)
        self.n_positive = int(n_positive)
        self.direction: Direction = direction
        self.feature_names = [str(f) for f in feature_names]
        self._d: List[int] = []
        self._features: List[Tuple[int, ...]] = []
        self._k_sel: List[int] = []
        self._n_centers: List[int] = []
        self._n_sel: List[int] = []
        self._frozen = False

    # -- recording ------------------------------------------------------------
    def record(self, combo: Tuple[int, ...], k_sel: int, n_centers: int, n_sel: int) -> None:
        if self._frozen:
            raise RuntimeError("landscape is frozen")
        self._d.append(len(combo))
        self._features.append(tuple(int(j) for j in combo))
        self._k_sel.append(int(k_sel))
        self._n_centers.append(int(n_centers))
        self._n_sel.append(int(n_sel))

    def freeze(self) -> "Landscape":
        self.d = np.asarray(self._d, dtype=np.int64)
        self.k_sel = np.asarray(self._k_sel, dtype=np.int64)
        self.n_centers = np.asarray(self._n_centers, dtype=np.int64)
        self.n_sel = np.asarray(self._n_sel, dtype=np.int64)
        self.features = self._features
        self._frozen = True
        return self

    def __len__(self) -> int:
        return len(self._features)

    # -- derived quantities ---------------------------------------------------
    def x_values(self) -> np.ndarray:
        """
        The horizontal coordinate of every candidate: coverage of the class
        searched (presence), or the mass of certified cells (absence - the
        headline of that search), both as fractions in [0, 1].
        """
        if self.direction == "absence":
            return self.n_sel / float(self.n_samples) if self.n_samples else np.zeros_like(self.n_sel, dtype=float)
        return self.k_sel / float(self.n_positive) if self.n_positive else np.zeros_like(self.k_sel, dtype=float)

    @staticmethod
    def bin_index(fraction: np.ndarray, n_bins: int) -> np.ndarray:
        """
        Category index of a fraction in (0, 1] under the left-open,
        right-closed intervals (0, 1/n], (1/n, 2/n], ..., ((n-1)/n, 1]:
        `ceil(f * n) - 1`, evaluated on the exact rational so that f = 0.1
        lands in the first interval and f = 1 in the last. Callers exclude
        f = 0 (no certified centre) before binning.
        """
        f = np.asarray(fraction, dtype=np.float64)
        # ceil on a value that is exactly k/n must not be pushed over the
        # edge by floating-point representation: k/n * n is evaluated with
        # a relative tolerance below any gap between distinct fractions
        # with the denominators this package sees (n_positive, K_max <= N).
        idx = np.ceil(f * n_bins - 1e-9).astype(np.int64) - 1
        return np.clip(idx, 0, n_bins - 1)

    def _selection(self, d: Optional[int]) -> np.ndarray:
        mask = self.n_centers > 0
        if d is not None:
            mask &= self.d == int(d)
        return mask

    def k_max(self, d: Optional[int] = None) -> int:
        sel = self.d == int(d) if d is not None else np.ones(self.d.shape, dtype=bool)
        return int(self.n_centers[sel].max()) if np.any(sel) else 0

    def bins(self, d: Optional[int] = None) -> Dict[str, object]:
        """
        The 10 x 10 count lattice for one dimensionality (`d`) or for the
        whole family (`None`): `counts[iy][ix]` is the number of candidates
        whose x-fraction falls in column ix and whose n_centers / k_max
        falls in row iy, both under `bin_index`. Candidates with no
        certified centre are excluded from the lattice and counted in
        `n_zero`; `k_max` is the largest centre count among the family's
        candidates (per d when `d` is given), the fixed top of the y axis.
        """
        n = self.N_BINS
        family = self.d == int(d) if d is not None else np.ones(self.d.shape, dtype=bool)
        n_total = int(family.sum())
        sel = family & (self.n_centers > 0)
        k_max = self.k_max(d)
        counts = np.zeros((n, n), dtype=np.int64)
        if np.any(sel) and k_max > 0:
            ix = self.bin_index(self.x_values()[sel], n)
            iy = self.bin_index(self.n_centers[sel] / float(k_max), n)
            np.add.at(counts, (iy, ix), 1)
        return {
            "d": d,
            "n_bins": n,
            "k_max": k_max,
            "n_total": n_total,
            "n_zero": int(n_total - int(sel.sum())),
            "counts": counts.tolist(),
            "max_count": int(counts.max()) if counts.size else 0,
            "x": "mass" if self.direction == "absence" else "coverage",
        }

    def by_x_category(
        self, d: Optional[int], ix: int, limit: int = 100, offset: int = 0
    ) -> Dict[str, object]:
        """
        The candidates whose x-fraction (coverage, or mass under absence)
        falls in category `ix` of the ten `bin_index` intervals, at ANY
        centre count above zero: the schemas "at this coverage" for the
        tau-curve's click-through. Highest x first, then fewest centres,
        then lowest mass.
        """
        n = self.N_BINS
        family = self.d == int(d) if d is not None else np.ones(self.d.shape, dtype=bool)
        idx = np.nonzero(family & (self.n_centers > 0))[0]
        if idx.size == 0:
            return {"total": 0, "offset": int(offset), "limit": int(limit), "schemas": []}
        x = self.x_values()[idx]
        hit = idx[self.bin_index(x, n) == int(ix)]
        xs = self.x_values()[hit]
        mass = self.n_sel[hit] / float(self.n_samples) if self.n_samples else np.zeros(hit.shape)
        order = np.lexsort((mass, self.n_centers[hit], -xs))
        hit = hit[order]
        total = int(hit.shape[0])
        page = hit[int(offset): int(offset) + int(limit)]
        return {
            "total": total, "offset": int(offset), "limit": int(limit),
            "schemas": [self._schema_dict(i) for i in page.tolist()],
        }

    def _schema_dict(self, i: int) -> Dict[str, object]:
        feats = self.features[i]
        return {
            "d": int(self.d[i]),
            "features": [int(j) for j in feats],
            "feature_names": [self.feature_names[j] for j in feats],
            "coverage": (self.k_sel[i] / self.n_positive) if self.n_positive else 0.0,
            "n_centers": int(self.n_centers[i]),
            "mass": (self.n_sel[i] / self.n_samples) if self.n_samples else 0.0,
        }

    def cell(
        self, d: Optional[int], ix: int, iy: int, limit: int = 100, offset: int = 0
    ) -> Dict[str, object]:
        """
        The candidates of one lattice cell, most concentrated first (lowest
        mass, then fewest centres, then highest x), as dicts with the
        feature indices and names and the exact coverage / n_centers / mass.
        """
        n = self.N_BINS
        family = self.d == int(d) if d is not None else np.ones(self.d.shape, dtype=bool)
        sel = family & (self.n_centers > 0)
        k_max = self.k_max(d)
        idx = np.nonzero(sel)[0]
        if idx.size == 0 or k_max == 0:
            return {"total": 0, "offset": offset, "limit": limit, "schemas": []}
        x = self.x_values()[idx]
        ixs = self.bin_index(x, n)
        iys = self.bin_index(self.n_centers[idx] / float(k_max), n)
        hit = idx[(ixs == int(ix)) & (iys == int(iy))]
        mass = self.n_sel[hit] / float(self.n_samples) if self.n_samples else np.zeros(hit.shape)
        order = np.lexsort((-self.x_values()[hit], self.n_centers[hit], mass))
        hit = hit[order]
        total = int(hit.shape[0])
        page = hit[int(offset): int(offset) + int(limit)]
        schemas = []
        for i in page.tolist():
            feats = self.features[i]
            schemas.append({
                "d": int(self.d[i]),
                "features": [int(j) for j in feats],
                "feature_names": [self.feature_names[j] for j in feats],
                "coverage": (self.k_sel[i] / self.n_positive) if self.n_positive else 0.0,
                "n_centers": int(self.n_centers[i]),
                "mass": (self.n_sel[i] / self.n_samples) if self.n_samples else 0.0,
            })
        return {"total": total, "offset": int(offset), "limit": int(limit), "schemas": schemas}


def _exhaustive_search(
    factory: _CandidateFactory,
    z_codes: np.ndarray,
    n_values: int,
    value_indices: Sequence[int],
    spec: CenterSpec,
    max_d: int,
    complement: bool = False,
    landscape: Optional[Landscape] = None,
) -> List[Dict[int, _Ranked]]:
    """
    argmax_{|S| = d} coverage_score(Z_v; X_S) for every d <= max_d and every
    target value v in `value_indices`, in ONE pass over the candidate family.

    `landscape`, when given, receives every candidate's (features, k_sel,
    n_centers, n_sel) for the FIRST value of `value_indices` (a single-
    indicator search); see `Landscape`.

    `complement=True` scores the indicator `z_codes != v` instead of
    `z_codes == v` for every v - the absence search (`Direction`). The same
    (cells x values) table serves: the complement's per-cell count is
    `n_cell - table[:, v]` and its total is `N - n_positive[v]`, so one
    enumeration still covers every value of the column.

    `z_codes` are dense codes 0..n_values-1 of the target; value v's 0/1
    indicator is `z_codes == v`. One `bincount` of `codes * n_values +
    z_codes` yields the whole (cells x values) contingency table of a
    candidate, from which every value's (k_cell, n_cell) - and hence its
    `coverage_score` - is read off. The Global Pattern Scan uses this to
    score all values of a target column in a single enumeration; a binary
    search passes `n_values = 2, value_indices = [1]`.

    Equivalence with `coverage_score`, component by component: coverage =
    k_sel / n_positive and mass = n_sel / n_samples are the same integer
    ratios; the centre count is the same mask (`min_successes_to_select` on
    the same cell sizes at the same Bonferroni level over the same occupied
    count, which for dense codes is the cell count); the Wilson tie-break is
    evaluated only when the first three components tie the incumbent, and is
    the same `wilson_lower(k, n, alpha_eff).max()`. Ties resolve to the first
    candidate in enumeration order, as `key > incumbent` did.
    """
    z = np.asarray(z_codes, dtype=np.int64).ravel()
    n_samples = int(z.shape[0])
    values = [int(v) for v in value_indices]
    n_positive = np.bincount(z, minlength=n_values).astype(np.int64)
    if complement:
        n_positive = n_samples - n_positive
    best: List[Dict[int, _Ranked]] = [{} for _ in values]
    # Incumbent state per (value, d): first three key components, Wilson
    # tie-break (None until needed), the cell counts to compute it from.
    inc3: List[Dict[int, Tuple[float, float, float]]] = [{} for _ in values]
    inc_w: List[Dict[int, Optional[float]]] = [{} for _ in values]
    inc_kn: List[Dict[int, Tuple[np.ndarray, np.ndarray, float]]] = [{} for _ in values]
    inc_combo: List[Dict[int, Tuple[int, ...]]] = [{} for _ in values]

    for combo, codes, n_cells in factory.iter_candidates(max_d):
        d = len(combo)
        if n_cells == 0:
            table = np.zeros((0, n_values), dtype=np.int64)
        else:
            table = np.bincount(
                codes * n_values + z, minlength=n_cells * n_values
            ).reshape(n_cells, n_values)
        n_cell = table.sum(axis=1)
        occupied = int(np.count_nonzero(n_cell > 0))
        alpha_eff = spec.effective_alpha(occupied)
        if occupied == 0:
            k_min = np.ones(n_cell.shape, dtype=np.int64)
            eligible = np.zeros(n_cell.shape, dtype=bool)
        else:
            k_min = min_successes_to_select(n_cell, spec, alpha_eff)
            eligible = n_cell > 0
        for vi, v in enumerate(values):
            k_cell = (n_cell - table[:, v]) if complement else table[:, v]
            n_pos = int(n_positive[v])
            mask = (k_cell >= k_min) & eligible
            if n_pos > 0 and n_samples > 0:
                k_sel = int(k_cell[mask].sum())
                n_sel = int(n_cell[mask].sum())
                key3 = (k_sel / n_pos, -float(mask.sum()), -(n_sel / n_samples))
            else:
                # `_coverage_from_counts` reports (0, 0, 0) here; the centre
                # count still comes from the real mask, as in `coverage_score`.
                k_sel, n_sel = 0, 0
                key3 = (0.0, -float(mask.sum()), -0.0)
            if landscape is not None and vi == 0:
                landscape.record(combo, k_sel, int(mask.sum()), n_sel)
            current = inc3[vi].get(d)
            if current is None or key3 > current:
                inc3[vi][d] = key3
                inc_w[vi][d] = None
                inc_kn[vi][d] = (k_cell, n_cell, alpha_eff)
                inc_combo[vi][d] = combo
            elif key3 == current:
                w_inc = inc_w[vi][d]
                if w_inc is None:
                    kk, nn, aa = inc_kn[vi][d]
                    w_inc = _best_wilson_lower(kk, nn, aa)
                    inc_w[vi][d] = w_inc
                w_new = _best_wilson_lower(k_cell, n_cell, alpha_eff)
                if w_new > w_inc:
                    inc_w[vi][d] = w_new
                    inc_kn[vi][d] = (k_cell, n_cell, alpha_eff)
                    inc_combo[vi][d] = combo

    for vi in range(len(values)):
        for d, key3 in inc3[vi].items():
            w = inc_w[vi][d]
            if w is None:
                kk, nn, aa = inc_kn[vi][d]
                w = _best_wilson_lower(kk, nn, aa)
            best[vi][d] = (key3 + (w,), inc_combo[vi][d])
    return best


def _best_wilson_lower(k_cell: np.ndarray, n_cell: np.ndarray, alpha_eff: float) -> float:
    """Fourth component of `coverage_score`: the best per-cell Wilson lower bound."""
    return float(wilson_lower(k_cell, n_cell, alpha_eff).max()) if n_cell.size else 0.0


def _resolve_positive_indicator(
    Z_arr: np.ndarray,
    z_codes: np.ndarray,
    n_rows: int,
    positive_class: Optional[object],
) -> Optional[np.ndarray]:
    """
    The 0/1 indicator the centre layer - and hence the search - is defined
    against, or None when the target admits no unambiguous positive value.

    Resolution order, deliberately explicit:

    1.  `positive_class` given -> it must match exactly one distinct value of
        the RAW target, and the raw values must correspond 1:1 to the integer
        codes (i.e. the target was not PMD-binned). Anything else raises,
        rather than silently reporting centres of a different class.
    2.  No `positive_class` and a two-valued target -> the higher code, which
        is `np.unique(Z)[-1]`. This is the value `vsf.vis` labels as the
        positive class and the literal 1 of `/api/analyze`'s One-vs-Rest
        `criterion` path, so the three agree by construction.
    3.  Otherwise -> None. A K-class target has no single purity; the caller
        must say which value it means. `discover_branches` turns this into a
        `ValueError` - v2.3 no longer has a ranking objective defined for
        this case (see module docstring).
    """
    if positive_class is not None:
        uniq = np.unique(Z_arr)
        if uniq.shape[0] != n_rows:
            raise ValueError(
                "positive_class cannot be resolved: the target was discretised "
                f"into {n_rows} bins from {uniq.shape[0]} distinct raw values, so "
                "raw values no longer correspond 1:1 to class codes"
            )
        matches = np.nonzero(uniq == positive_class)[0]
        if matches.shape[0] != 1:
            raise ValueError(
                f"positive_class={positive_class!r} matches {matches.shape[0]} "
                "distinct target values; exactly one is required"
            )
        return (z_codes == int(matches[0])).astype(np.int8)
    if n_rows == 2:
        return (z_codes == 1).astype(np.int8)
    return None


def _center_prefix_series(
    z_binary: np.ndarray,
    X_discrete: np.ndarray,
    bin_counts: Sequence[int],
    ordered_features: Sequence[int],
    spec: CenterSpec,
    factory: Optional[_CandidateFactory] = None,
) -> Tuple[List[float], List[int], List[float]]:
    """
    In-sample (coverage, K, pooled purity) for every prefix of one branch's
    own axis ordering - projections of THIS branch's partition onto its own
    first k axes, not `discover_branches`'s independently optimal
    k-dimensional branch (see `BranchResult.coverage_by_prefix_d`).
    """
    n_samples = int(z_binary.shape[0])
    if factory is None:
        factory = _CandidateFactory(X_discrete, bin_counts, n_samples)
    coverage: List[float] = []
    n_centers: List[int] = []
    purity: List[float] = []
    for k in range(1, len(ordered_features) + 1):
        codes, n_cells = factory.codes(tuple(ordered_features[:k]))
        cov, k_centers, pur = center_summary(z_binary, codes, n_cells, spec)
        coverage.append(cov)
        n_centers.append(k_centers)
        purity.append(pur)
    return coverage, n_centers, purity


def select_branch_dimensionality(
    branches: Dict[int, BranchResult], t_threshold: float = 2.0
) -> Optional[int]:
    """
    "How many characteristics does it take to describe the target value?" -
    the smallest branch dimensionality whose OUT-OF-SAMPLE certified coverage
    is positive and beats every lower dimensionality by more than
    `t_threshold` corrected standard errors of the paired difference.

    Returns None when no dimensionality achieves positive cross-validated
    coverage, which is the correct answer for a target that has no certifiable
    centres at all, and must be surfaced as "none" rather than collapsed to
    d = 1. Also None when the branches carry no cross-validated estimate
    (fewer than `vsf.centers.MIN_POSITIVES_FOR_CV` positives).

    The comparison is valid only because every branch's `CenterReport` was
    cross-validated through `vsf.centers.stratified_repeated_kfold` with the
    same target vector and seed, so the per-fold coverages are paired; that
    is guaranteed by `discover_branches`, which passes one `random_state` to
    all of them.
    """
    cv_by_d: Dict[int, CVCoverage] = {
        d: br.centers.coverage_cv
        for d, br in branches.items()
        if br.centers.coverage_cv is not None
    }
    return _select_dimensionality(cv_by_d, t_threshold=t_threshold)


def discover_branches(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_d: int = MAX_BRANCH_D,
    random_state: Optional[int] = 0,
    progress: Optional[Callable[[int, int], None]] = None,
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    n_permutations_centers: int = DEFAULT_N_PERMUTATIONS,
    n_permutations_familywise_coverage: int = 0,
    cv_splits: int = 5,
    cv_repeats: int = 5,
    direction: Direction = "presence",
    landscape: Optional[Landscape] = None,
    prune_dependent: bool = False,
) -> Dict[int, BranchResult]:
    """
    Independent Branch Discovery (Project_Master_Document.md Section 4.3).

    For every d in {1, ..., min(max_d, M)}, independently computes

        S*_d = argmax_{S subset of {1,...,M}, |S| = d}  coverage_score(Z_pos; X_S)

    by exhaustively enumerating every C(M, d) combination (minus the ones
    `prune_dependent` proves to be renamings of a smaller one, Section 4.12)
    - never a greedy
    approximation, never a prefiltered candidate pool. `coverage_score`
    (`vsf.centers.coverage_score`) ranks lexicographically by (coverage,
    -n_centres, -mass, best per-cell lower bound); see its docstring for the
    tie-break rationale. Returns up to `max_d` branches, keyed by
    dimensionality; a key is missing only when `M < d`.

    Why exhaustive, not greedy: coverage is not submodular in the feature
    set (Krause & Guestrin, 2005). A greedy chain that
    commits to the single best feature at d=1 and only ever ADDS to it can
    never discover a pair {a, b} that is jointly informative while neither a
    nor b is individually strong - exactly the synergy case
    Project_Master_Document.md Section 4.4 requires each branch to be capable
    of finding.

    A positive class is REQUIRED: there is no other ranking to fall back to
    when one cannot be resolved.
    See `_resolve_positive_indicator`'s docstring for exactly how it is
    resolved and `positive_class` below for the parameter itself.

    Parameters
    ----------
    positive_class
        The raw target value that centres are certified against, and hence
        that the search maximises coverage of. Optional for a two-valued
        target (the higher `np.unique` code is used, matching the renderer
        and the One-vs-Rest `criterion` path); REQUIRED for a K-class target
        - `ValueError` otherwise, since no ranking is defined without it.
    center_spec
        `vsf.centers.CenterSpec` - the purity floor `tau`, the simultaneous
        error rate `alpha`, and the multiplicity policy. Defaults to
        tau = 0.90, alpha = 0.05, Bonferroni over the branch's occupied
        cells.
    n_permutations_centers
        Replicates for each branch's uncorrected coverage p-value. Cheap: the
        null is drawn directly from the multivariate hypergeometric law of
        the cell counts, O(C) per replicate rather than O(N).
    n_permutations_familywise_coverage
        Replicates for `coverage_p_value_familywise`. Costs B x the whole
        candidate family, so it is 0 by default; a published coverage claim
        must set it.
    random_state
        Seeds the coverage permutation nulls (per-branch and familywise), so
        the function stays reproducible; with both permutation counts at 0
        it is fully deterministic and consumes no randomness.
    progress
        Optional `(done, total)` callback, invoked during the familywise
        coverage null only - the phase whose runtime is worth reporting to a
        user.
    cv_splits / cv_repeats
        Stratified repeated K-fold parameters behind
        `CenterReport.coverage_cv`. Set `cv_repeats = 0` to skip the
        out-of-sample estimate entirely (and with it
        `select_branch_dimensionality`).
    direction
        `"presence"` (default) or `"absence"` - see `Direction`. Under
        `"absence"` every reported number refers to the complement of the
        positive class: `centers.coverage` is the share of NON-positive rows
        inside cells certified at least `tau` pure in non-positives,
        `centers.mass` the share of ALL rows in those cells (the headline of
        an absence search: how much of the data is certified free of the
        value), `centers.prevalence` is `1 - p_0`.

    landscape
        An unfrozen `Landscape` to record every scored candidate into (its
        `n_samples` / `n_positive` / `direction` must describe this search;
        `compute_landscape` builds one correctly). Frozen on return.

    Raises `ValueError` when `tau` is not above the base rate of the
    indicator searched (`base_rate_reason`): such a search is void, not
    merely weak, and the caller must be told rather than shown a uniformly
    green display.

    Cost: see Project_Master_Document.md Section 4.6. This function makes no
    attempt to bound the number of combinations evaluated - by explicit
    product decision, "always honest full enumeration", regardless of M.
    Callers with very wide datasets (M >~ 50-100) should expect this to take
    minutes; that is the accepted cost of never approximating. The
    enumeration itself is `_CandidateFactory.iter_candidates` and the
    scoring `_exhaustive_search` (2026-09): per-column coarsening cache,
    prefix-shared joint codes, sort-free dense relabelling, one fused
    `bincount` per candidate and a lazily evaluated tie-break - each pinned
    against its straightforward form in `tests/test_fastpaths.py`, none of
    them changing which subset wins or any number reported for it.
    """
    if max_d < 1 or max_d > MAX_BRANCH_D:
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
    if n_permutations_centers < 0 or n_permutations_familywise_coverage < 0:
        raise ValueError("permutation counts must be non-negative")
    spec = center_spec if center_spec is not None else CenterSpec()

    prepared = _prepare_search(X, Z, feature_names, positive_class, spec, direction, prune_dependent)
    branches: Dict[int, BranchResult] = {}
    if prepared is None:
        return branches
    factory, z_binary, feature_names = prepared
    effective_max_d = min(max_d, factory.n_features)
    # ---- exhaustive search ------------------------------------------------
    # The 0/1 indicator is its own two-valued code vector; value 1 is the
    # class searched for (the positive class, or under `direction="absence"`
    # its complement). Ranking key is the lexicographic tuple of
    # `coverage_score`, first candidate in enumeration order wins ties.
    best_by_d = _exhaustive_search(
        factory, z_binary.astype(np.int64), 2, [1], spec, effective_max_d,
        landscape=landscape,
    )[0]
    if landscape is not None:
        landscape.freeze()
    if not best_by_d:
        return branches

    return _report_branches(
        factory,
        z_binary,
        best_by_d,
        feature_names,
        spec,
        effective_max_d,
        random_state=random_state,
        progress=progress,
        n_permutations_centers=n_permutations_centers,
        n_permutations_familywise_coverage=n_permutations_familywise_coverage,
        cv_splits=cv_splits,
        cv_repeats=cv_repeats,
    )


def _prepare_search(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[Sequence[str]],
    positive_class: Optional[object],
    spec: CenterSpec,
    direction: Direction,
    prune_dependent: bool = False,
) -> Optional[Tuple[_CandidateFactory, np.ndarray, List[str]]]:
    """
    Everything a search needs before enumerating: the discretised feature
    matrix wrapped in a `_CandidateFactory`, the oriented 0/1 indicator
    (`_resolve_positive_indicator` then `_oriented_indicator`), and the
    feature names. None when there are no feature columns. Raises the same
    `ValueError`s as `discover_branches` for an unresolvable positive class
    or a threshold not above the base rate.
    """
    X_arr = np.asarray(X)
    Z_arr = np.asarray(Z).ravel()
    n_samples, n_features = X_arr.shape
    names = (
        [f"X_{j + 1}" for j in range(n_features)] if feature_names is None
        else [str(f) for f in feature_names]
    )
    if n_features == 0:
        return None

    X_discrete, bin_counts = discretize_dataset(X_arr)
    Z_discrete = _discretize_target(Z_arr)
    z_codes, n_rows = cell_codes(Z_discrete)
    z_binary = _resolve_positive_indicator(Z_arr, z_codes, n_rows, positive_class)
    if z_binary is None:
        raise ValueError(
            "discover_branches needs a resolvable positive class: the target "
            f"has {n_rows} distinct values and no positive_class was given. "
            "A branch search always ranks by coverage of one named value "
            "(v2.3); pass positive_class explicitly for a target with more "
            "than two values."
        )
    z_binary = _oriented_indicator(z_binary, direction)
    prevalence = float(z_binary.mean()) if n_samples > 0 else 0.0
    reason = base_rate_reason(spec, prevalence, direction)
    if reason is not None:
        raise ValueError(reason)
    determined_by = exact_dependencies(X_arr, names) if prune_dependent else None
    return (
        _CandidateFactory(
            X_discrete, bin_counts, n_samples, ordered=ordered_columns(X_arr),
            determined_by=determined_by,
        ),
        z_binary,
        names,
    )


def compute_landscape(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_d: int = MAX_BRANCH_D,
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    direction: Direction = "presence",
    prune_dependent: bool = False,
) -> Landscape:
    """
    The `Landscape` of the search `discover_branches` would run with these
    arguments - every candidate's (features, k_sel, n_centers, n_sel) - with
    no reporting stage. Costs the ranking pass alone.
    """
    if max_d < 1 or max_d > MAX_BRANCH_D:
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
    spec = center_spec if center_spec is not None else CenterSpec()
    prepared = _prepare_search(X, Z, feature_names, positive_class, spec, direction, prune_dependent)
    if prepared is None:
        return Landscape(int(np.asarray(X).shape[0]), 0, direction, []).freeze()
    factory, z_binary, names = prepared
    landscape = Landscape(factory.n_samples, int(z_binary.sum()), direction, names)
    _exhaustive_search(
        factory, z_binary.astype(np.int64), 2, [1], spec, min(max_d, factory.n_features),
        landscape=landscape,
    )
    return landscape.freeze()


def tau_grid(anchor: float, step_pct: float = 1.0) -> np.ndarray:
    """
    The purity floors a tau-curve is evaluated at: every `step_pct` percent
    from the first grid point strictly above `anchor` (the base rate of the
    indicator searched) up to and including 100 %. Grid points are exact
    multiples of the step, so that the curve's x values coincide with the
    values a user can set on the certificate boundary.
    """
    if not (0.0 <= anchor <= 1.0):
        raise ValueError(f"anchor must be in [0, 1], got {anchor}")
    if not (0.0 < step_pct <= 50.0):
        raise ValueError(f"step_pct must be in (0, 50], got {step_pct}")
    first = int(np.floor(anchor * 100.0 / step_pct)) + 1
    values = np.arange(first, int(np.floor(100.0 / step_pct)) + 1, dtype=np.float64) * step_pct
    values = values[values <= 100.0 + 1e-9]
    return np.round(values / 100.0, 10)


def compute_tau_curves(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_d: int = MAX_BRANCH_D,
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    direction: Direction = "presence",
    step_pct: float = 1.0,
    prune_dependent: bool = False,
) -> Dict[str, object]:
    """
    The per-dimensionality ENVELOPE of the solution landscape over the
    purity floor: for every d <= max_d and every tau on `tau_grid`
    (base rate of the searched indicator up to 100 %, in `step_pct`
    steps), the best coverage any d-subset reaches at that tau, with the
    subset that reaches it, its centre count and mass, and how many of the
    d-subsets certify anything at all. `center_spec.tau` is ignored - the
    curve IS the dependence on tau; `rule`, `min_samples` and `alpha` are
    honoured.

    For a fixed partition, coverage(tau) is a right-continuous step function
    of tau, non-increasing, with steps at the cells' purities: under
    Rule P a cell of size n >= min_samples is a centre at tau iff
    k / n >= tau, so sorting the cells by purity once and taking cumulative
    sums gives the whole curve, and the grid is read off by one
    `searchsorted` per candidate. Under Rule C the threshold depends on n
    and tau jointly (`min_successes_to_certify`), so each grid point is a
    separate selection; the memo in `vsf.centers` keeps that affordable.
    The per-d envelope is the pointwise maximum over the family, so it is
    non-increasing in tau as well; where two envelopes cross, the extra
    axis has stopped paying for itself at that purity floor.

    Ranking at each grid point follows `coverage_score`'s first three
    components (coverage, fewer centres, less mass) with the first
    candidate in enumeration order winning exact ties. The Wilson
    tie-break is not evaluated because it cannot matter: two candidates
    tied on coverage and mass have the same (k_sel, n_sel), hence the
    same Wilson bound, so the ranking here is exactly `coverage_score`'s.
    """
    if max_d < 1 or max_d > MAX_BRANCH_D:
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
    spec = center_spec if center_spec is not None else CenterSpec()
    # The base-rate invariant is what the grid starts from, not a reason to
    # refuse: build the search with a floor that is always admissible.
    prepared = _prepare_search(
        X, Z, feature_names, positive_class,
        CenterSpec(tau=1.0 if spec.rule == "purity" else 0.999, alpha=spec.alpha,
                   rule=spec.rule, min_samples=spec.min_samples, method=spec.method,
                   multiplicity=spec.multiplicity),
        direction,
        prune_dependent,
    )
    if prepared is None:
        return {"taus": [], "curves": {}, "anchor": 0.0, "step_pct": step_pct,
                "x": "mass" if direction == "absence" else "coverage", "n_positive": 0, "n_samples": 0}
    factory, z_binary, names = prepared
    n_samples = factory.n_samples
    n_positive = int(z_binary.sum())
    anchor = n_positive / n_samples if n_samples else 0.0
    taus = tau_grid(anchor, step_pct)
    if spec.rule == "certified":
        taus = taus[taus < 1.0]  # tau = 1 is not certifiable
    T = int(taus.shape[0])
    effective_max_d = min(max_d, factory.n_features)
    z64 = z_binary.astype(np.int64)
    use_mass = direction == "absence"

    best_x = {d: np.full(T, -1.0) for d in range(1, effective_max_d + 1)}
    best_key = {d: [None] * T for d in range(1, effective_max_d + 1)}
    best_combo = {d: [None] * T for d in range(1, effective_max_d + 1)}
    best_k = {d: np.zeros(T, dtype=np.int64) for d in range(1, effective_max_d + 1)}
    best_cov = {d: np.zeros(T) for d in range(1, effective_max_d + 1)}
    best_mass = {d: np.zeros(T) for d in range(1, effective_max_d + 1)}
    n_family = {d: 0 for d in range(1, effective_max_d + 1)}
    n_certifying = {d: np.zeros(T, dtype=np.int64) for d in range(1, effective_max_d + 1)}

    for combo, codes, n_cells in factory.iter_candidates(effective_max_d):
        d = len(combo)
        n_family[d] += 1
        if n_cells == 0 or n_positive == 0:
            continue
        table = np.bincount(codes * 2 + z64, minlength=2 * n_cells).reshape(n_cells, 2)
        n_cell = table.sum(axis=1)
        k_cell = table[:, 1]
        eligible = n_cell >= spec.min_samples
        if spec.rule == "purity":
            # Sorted-by-purity cumulative sums: the whole curve at once.
            n_e = n_cell[eligible]
            k_e = k_cell[eligible]
            if n_e.size == 0:
                continue
            purity = k_e / n_e
            order = np.argsort(-purity, kind="stable")
            p_sorted = purity[order]
            cum_k = np.cumsum(k_e[order])
            cum_n = np.cumsum(n_e[order])
            # number of cells with purity >= tau (with the same 1e-9
            # tolerance `min_successes_to_select` applies to tau * n)
            count = np.searchsorted(-p_sorted, -(taus - 1e-9), side="right")
            k_sel = np.where(count > 0, cum_k[np.maximum(count - 1, 0)], 0)
            n_sel = np.where(count > 0, cum_n[np.maximum(count - 1, 0)], 0)
            K = count
        else:
            occupied = int(np.count_nonzero(n_cell > 0))
            alpha_eff = spec.effective_alpha(occupied)
            k_sel = np.zeros(T, dtype=np.int64)
            n_sel = np.zeros(T, dtype=np.int64)
            K = np.zeros(T, dtype=np.int64)
            for ti, tau in enumerate(taus.tolist()):
                k_min = min_successes_to_select(
                    n_cell, CenterSpec(tau=float(tau), alpha=spec.alpha, rule="certified",
                                       min_samples=spec.min_samples, method=spec.method,
                                       multiplicity=spec.multiplicity),
                    alpha_eff,
                )
                mask = (k_cell >= k_min) & (n_cell > 0)
                k_sel[ti] = int(k_cell[mask].sum())
                n_sel[ti] = int(n_cell[mask].sum())
                K[ti] = int(mask.sum())
        cov = k_sel / n_positive
        mass = n_sel / n_samples
        x = mass if use_mass else cov
        n_certifying[d] += (K > 0)
        # lexicographic (x, -K, -mass) improvement, first-wins on ties
        better = (x > best_x[d]) | ((x == best_x[d]) & (
            (K < best_k[d]) | ((K == best_k[d]) & (mass < best_mass[d]))
        ))
        # A candidate that ties exactly on all three does not replace.
        idx = np.nonzero(better)[0]
        if idx.size:
            best_x[d][idx] = x[idx]
            best_k[d][idx] = K[idx]
            best_cov[d][idx] = cov[idx]
            best_mass[d][idx] = mass[idx]
            for ti in idx.tolist():
                best_combo[d][ti] = combo

    curves: Dict[str, Dict[str, object]] = {}
    for d in range(1, effective_max_d + 1):
        curves[str(d)] = {
            "coverage": [float(v) for v in best_cov[d]],
            "mass": [float(v) for v in best_mass[d]],
            "x": [float(v) if v >= 0 else 0.0 for v in best_x[d]],
            "n_centers": [int(v) for v in best_k[d]],
            "features": [list(c) if c is not None else [] for c in best_combo[d]],
            "feature_names": [[names[j] for j in c] if c is not None else [] for c in best_combo[d]],
            "n_certifying": [int(v) for v in n_certifying[d]],
            "n_family": int(n_family[d]),
        }
    return {
        "taus": [float(v) for v in taus],
        "anchor": float(anchor),
        "step_pct": float(step_pct),
        "x": "mass" if use_mass else "coverage",
        "n_positive": n_positive,
        "n_samples": n_samples,
        "curves": curves,
    }


def report_schema(
    X: np.ndarray,
    Z: np.ndarray,
    features: Sequence[int],
    feature_names: Optional[List[str]] = None,
    random_state: Optional[int] = 0,
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    n_permutations_centers: int = 0,
    n_permutations_familywise_coverage: int = 0,
    cv_splits: int = 5,
    cv_repeats: int = 5,
    direction: Direction = "presence",
) -> Dict[int, BranchResult]:
    """
    The full report (`_report_branches`) for ONE explicitly chosen feature
    subset, without a search: `{len(features): BranchResult}`. This is how
    a schema picked from the `Landscape` is opened.

    Statistical caveat, and the reason `n_permutations_centers` defaults to
    0 here unlike `discover_branches`: a schema chosen by looking at the
    landscape is selected on the data, so its uncorrected coverage p-value
    (`CenterReport.coverage_p_value`) is not valid - it would be the
    p-value of a hypothesis picked because it looked good. The
    cross-validated coverage stays honest (the folds are held out
    regardless of how the schema was chosen), and the familywise p-value,
    when requested, stays valid and conservative: the null of the maximum
    over the whole family bounds any member's coverage.
    """
    if n_permutations_centers < 0 or n_permutations_familywise_coverage < 0:
        raise ValueError("permutation counts must be non-negative")
    spec = center_spec if center_spec is not None else CenterSpec()
    combo = tuple(int(j) for j in features)
    if not combo or len(set(combo)) != len(combo) or len(combo) > MAX_BRANCH_D:
        raise ValueError(
            f"features must be 1 to {MAX_BRANCH_D} distinct column indices, got {list(features)}"
        )
    prepared = _prepare_search(X, Z, feature_names, positive_class, spec, direction)
    if prepared is None:
        return {}
    factory, z_binary, names = prepared
    if any(j < 0 or j >= factory.n_features for j in combo):
        raise ValueError(f"feature index out of range in {list(combo)}")
    d = len(combo)
    codes, n_cells = factory.codes(combo)
    key = coverage_score(z_binary, codes, n_cells, spec)
    return _report_branches(
        factory,
        z_binary,
        {d: (key, combo)},
        names,
        spec,
        max(d, 1),
        random_state=random_state,
        progress=None,
        n_permutations_centers=n_permutations_centers,
        n_permutations_familywise_coverage=n_permutations_familywise_coverage,
        cv_splits=cv_splits,
        cv_repeats=cv_repeats,
    )


def _report_branches(
    factory: _CandidateFactory,
    z_binary: np.ndarray,
    best_by_d: Dict[int, _Ranked],
    feature_names: Sequence[str],
    spec: CenterSpec,
    effective_max_d: int,
    random_state: Optional[int],
    progress: Optional[Callable[[int, int], None]],
    n_permutations_centers: int,
    n_permutations_familywise_coverage: int,
    cv_splits: int,
    cv_repeats: int,
    cell_bounds: bool = True,
) -> Dict[int, BranchResult]:
    """
    The reporting stage of `discover_branches` for one positive indicator:
    the familywise coverage null (optional), then a full `CenterReport`, the
    prefix series and the familywise p-value for every winning subset.
    """
    branches: Dict[int, BranchResult] = {}
    X_discrete, bin_counts, n_samples = factory._raw, factory.bin_counts, factory.n_samples

    # ---- familywise coverage null, computed once for the whole scan ------
    fw_cov_p: Optional[Callable[[float], float]] = None
    if n_permutations_familywise_coverage > 0:
        fw_cov_null = familywise_max_coverage_null(
            z_binary,
            lambda: factory.iter_candidates(effective_max_d),
            spec=spec,
            n_permutations=n_permutations_familywise_coverage,
            random_state=random_state,
            progress=progress,
        )
        fw_cov_p = fw_cov_null.p_value

    # ---- per-branch reporting ---------------------------------------------
    for d, (_, best_combo) in sorted(best_by_d.items()):
        ordered_features = list(best_combo)
        codes, n_cells = factory.codes(tuple(best_combo))

        centers = center_report(
            z_binary,
            codes,
            n_cells,
            spec,
            n_splits=cv_splits,
            n_repeats=cv_repeats,
            n_permutations=n_permutations_centers,
            random_state=random_state,
            cell_bounds=cell_bounds,
        )
        cov_by_k, k_by_k, pur_by_k = _center_prefix_series(
            z_binary, X_discrete, bin_counts, ordered_features, spec, factory=factory
        )

        branches[d] = BranchResult(
            d=d,
            selected_features=ordered_features,
            selected_feature_names=[feature_names[j] for j in ordered_features],
            centers=centers,
            coverage_by_prefix_d=cov_by_k,
            n_centers_by_prefix_d=k_by_k,
            purity_by_prefix_d=pur_by_k,
            coverage_p_value_familywise=(
                fw_cov_p(centers.coverage) if fw_cov_p is not None else None
            ),
        )

    return branches


def iter_branches_by_value(
    X: np.ndarray,
    Z: np.ndarray,
    values: Sequence[object],
    feature_names: Optional[List[str]] = None,
    max_d: int = MAX_BRANCH_D,
    random_state: Optional[int] = 0,
    center_spec: Optional[CenterSpec] = None,
    n_permutations_centers: int = DEFAULT_N_PERMUTATIONS,
    n_permutations_familywise_coverage: int = 0,
    cv_splits: int = 5,
    cv_repeats: int = 5,
    cell_bounds: bool = True,
    direction: Direction = "presence",
    skipped: Optional[Dict[str, str]] = None,
    prune_dependent: bool = False,
) -> Iterator[Tuple[str, Dict[int, BranchResult]]]:
    """
    Lazy form of `discover_branches_by_value`: the shared exhaustive search
    runs once, up front, and then `(str(v), branches)` is yielded one value
    at a time as each value's reporting stage completes, in the order of
    `values`. A consumer that may want to stop early (the live server's
    background prefetch, which is superseded whenever the user clicks a
    different column) pays for the reporting of only the values it actually
    consumed. Semantics per value are exactly those of
    `discover_branches_by_value`.

    `direction` applies to every value (see `Direction`). A value whose
    indicator fails the base-rate check (`base_rate_reason`) is not
    searched: it is yielded with an empty branch dict and, when `skipped` is
    given, its reason is recorded there under the value's key - a bulk
    caller (the Global Pattern Scan) reports it rather than aborting the
    column.
    """
    if max_d < 1 or max_d > MAX_BRANCH_D:
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
    if n_permutations_centers < 0 or n_permutations_familywise_coverage < 0:
        raise ValueError("permutation counts must be non-negative")
    spec = center_spec if center_spec is not None else CenterSpec()

    X_arr = np.asarray(X)
    Z_str = np.asarray(Z).ravel().astype(str)
    n_samples, n_features = X_arr.shape
    if Z_str.shape[0] != n_samples:
        raise ValueError(f"sample count mismatch: {Z_str.shape[0]} vs {n_samples}")
    if feature_names is None:
        feature_names = [f"X_{j + 1}" for j in range(n_features)]
    value_keys = [str(v) for v in values]
    if n_features == 0 or not value_keys:
        for key in value_keys:
            yield key, {}
        return

    X_discrete, bin_counts = discretize_dataset(X_arr)
    effective_max_d = min(max_d, n_features)
    factory = _CandidateFactory(
        X_discrete, bin_counts, n_samples, ordered=ordered_columns(X_arr),
        determined_by=(exact_dependencies(X_arr, feature_names) if prune_dependent else None),
    )

    uniq, z_codes = np.unique(Z_str, return_inverse=True)
    z_codes = z_codes.astype(np.int64).ravel()
    n_values = int(uniq.shape[0])
    code_of = {str(u): i for i, u in enumerate(uniq.tolist())}
    # Values absent from Z get a code past the observed range, so their
    # column of the contingency table is identically zero.
    absent_code = n_values
    value_codes = [code_of.get(key, absent_code) for key in value_keys]
    n_values_eff = n_values + (1 if absent_code in value_codes else 0)

    # Base-rate check per value, on the indicator actually searched.
    counts = np.bincount(z_codes, minlength=n_values_eff).astype(np.float64)
    searchable: List[bool] = []
    for key, code in zip(value_keys, value_codes):
        p_value_class = counts[code] / n_samples if n_samples > 0 else 0.0
        prevalence = p_value_class if direction == "presence" else 1.0 - p_value_class
        reason = base_rate_reason(spec, prevalence, direction)
        if reason is not None and skipped is not None:
            skipped[key] = reason
        searchable.append(reason is None)

    search_value_codes = [c for c, ok in zip(value_codes, searchable) if ok]
    best_by_searched: Dict[int, Dict[int, _Ranked]] = {}
    if search_value_codes:
        results = _exhaustive_search(
            factory, z_codes, n_values_eff, search_value_codes, spec, effective_max_d,
            complement=(direction == "absence"),
        )
        best_by_searched = dict(zip(search_value_codes, results))

    for key, code, ok in zip(value_keys, value_codes, searchable):
        best_by_d = best_by_searched.get(code, {}) if ok else {}
        if not best_by_d:
            yield key, {}
            continue
        z_binary = _oriented_indicator((z_codes == code).astype(np.int8), direction)
        yield key, _report_branches(
            factory,
            z_binary,
            best_by_d,
            feature_names,
            spec,
            effective_max_d,
            random_state=random_state,
            progress=None,
            n_permutations_centers=n_permutations_centers,
            n_permutations_familywise_coverage=n_permutations_familywise_coverage,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
            cell_bounds=cell_bounds,
        )


def discover_branches_by_value(
    X: np.ndarray,
    Z: np.ndarray,
    values: Sequence[object],
    feature_names: Optional[List[str]] = None,
    max_d: int = MAX_BRANCH_D,
    random_state: Optional[int] = 0,
    center_spec: Optional[CenterSpec] = None,
    n_permutations_centers: int = DEFAULT_N_PERMUTATIONS,
    n_permutations_familywise_coverage: int = 0,
    cv_splits: int = 5,
    cv_repeats: int = 5,
    cell_bounds: bool = True,
    direction: Direction = "presence",
    skipped: Optional[Dict[str, str]] = None,
    prune_dependent: bool = False,
) -> Dict[str, Dict[int, BranchResult]]:
    """
    `discover_branches` for SEVERAL one-vs-rest positive classes of the same
    raw target column, sharing one enumeration of the candidate family.

    For each `v` in `values` the result is exactly what
    `discover_branches(X, (Z.astype(str) == str(v)).astype(int),
    positive_class=1, ...)` returns - the same discretisation of `X`, the
    same candidate partitions (they do not depend on the target), the same
    ranking key evaluated on the same cell counts, the same reporting stage
    with the same seed - keyed by `str(v)`. Values are matched on their
    string form, which is how the Global Pattern Scan (`vsf.server`) builds
    its one-vs-rest indicators; a value absent from `Z` is a target with no
    positives and is reported as such (no centres, coverage 0), not an error.

    The saving is in the search: one `bincount` per candidate produces the
    (cells x values) table from which every value's `coverage_score` is read
    (`_exhaustive_search`), so a column with V observed values costs one
    enumeration instead of V. The reporting stage (cross-validation, the
    per-branch permutation null, the prefix series) stays per value, as does
    the familywise null when `n_permutations_familywise_coverage > 0`.

    `cell_bounds=False` forwards to `vsf.centers.center_report` and skips the
    per-cell confidence intervals of every report (see there for exactly
    which fields it affects; the branch choice, coverage, K, purity,
    cross-validation and p-values are not among them).

    See `iter_branches_by_value` for the same computation delivered one
    value at a time.
    """
    return dict(
        iter_branches_by_value(
            X,
            Z,
            values,
            feature_names=feature_names,
            max_d=max_d,
            random_state=random_state,
            center_spec=center_spec,
            n_permutations_centers=n_permutations_centers,
            n_permutations_familywise_coverage=n_permutations_familywise_coverage,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
            cell_bounds=cell_bounds,
            direction=direction,
            skipped=skipped,
            prune_dependent=prune_dependent,
        )
    )


class BranchEngine:
    """
    Thin, stateless wrapper around `discover_branches` matching v1.0's
    `AVREngine(...).fit(...)` call shape, so callers (server handlers,
    `vsf.dashboard`, tests) construct-then-fit as before.
    """

    def __init__(
        self,
        max_d: int = MAX_BRANCH_D,
        random_state: Optional[int] = 0,
        positive_class: Optional[object] = None,
        center_spec: Optional[CenterSpec] = None,
        n_permutations_centers: int = DEFAULT_N_PERMUTATIONS,
        n_permutations_familywise_coverage: int = 0,
        cv_splits: int = 5,
        cv_repeats: int = 5,
        direction: Direction = "presence",
    ):
        if max_d < 1 or max_d > MAX_BRANCH_D:
            raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
        self.max_d = max_d
        self.random_state = random_state
        self.positive_class = positive_class
        self.center_spec = center_spec
        self.n_permutations_centers = n_permutations_centers
        self.n_permutations_familywise_coverage = n_permutations_familywise_coverage
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.direction = direction

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        feature_names: Optional[List[str]] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[int, BranchResult]:
        return discover_branches(
            X,
            Z,
            feature_names=feature_names,
            max_d=self.max_d,
            random_state=self.random_state,
            progress=progress,
            positive_class=self.positive_class,
            center_spec=self.center_spec,
            n_permutations_centers=self.n_permutations_centers,
            n_permutations_familywise_coverage=self.n_permutations_familywise_coverage,
            cv_splits=self.cv_splits,
            cv_repeats=self.cv_repeats,
            direction=self.direction,
        )
