"""
Free rules: conjunctions of (column = value) conditions chosen under a
condition budget, with no common grid.

This is the selection of the `rules` / `rules_disjoint` baselines of the
paper's comparison (`experiments/baselines.py`), kept here so that the
interactive app and the experiments run the SAME code: the app's "Free
rules" overlay on the Landscape (`/api/landscape/free_rules`,
`select_free_rules`) is `rule_candidates` + `budgeted_greedy(disjoint=True)`
on all rows, which is what `baselines.select_rules(disjoint=True)` runs on a
training fold. `tests/test_free_rules.py` pins the two against each other.

Candidates: every conjunction of <= max_len (column = value) conditions that
occurs in the rows - the occupied cells of every UNCOARSENED partition of
<= max_len columns - one per distinct row set, described by the fewest
conditions. A candidate qualifies under the observed-purity rule (k / n >=
tau, n >= m) or, when a `threshold` is given, under a size-dependent count
(`certification_threshold`: the family-wise certificate of the product).

Selection: budgeted maximum coverage by the cost-effectiveness greedy; with
`disjoint=True` the chosen rules share no row, so their union keeps purity
>= tau by construction.

What `select_free_rules` reports is computed on the rows the rules were
chosen on. It is a description of those rows, not an estimate for new ones:
the held-out figures are those of `experiments/baselines.py`.

Two differences from the schemas the Landscape counts, both inherited from
the baseline and deliberate (the overlay must be the baseline):

* the rules are cells of uncoarsened partitions; a schema whose joint
  partition exceeds the grid capacity is scored by the search on a coarsened
  one (`vsf.avr._CandidateFactory`), so a rule's row set need not be a
  centre of the schema the Landscape drew for the same columns;
* `prune_dependent` does not remove candidates here.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .avr import MAX_BRANCH_D, Direction, _prepare_search, resolve_center_spec
from .centers import CenterSpec, min_successes_to_certify_heterogeneous
from .metrics import cell_codes

__all__ = [
    "Candidate",
    "FreeRule",
    "FreeRuleFamily",
    "FreeRuleSet",
    "budgeted_greedy",
    "certification_threshold",
    "qualifying",
    "rule_candidates",
    "select_free_rules",
]


@dataclass(frozen=True)
class Candidate:
    """A qualifying group on the training rows: its positive training rows and cost."""

    positives: np.ndarray  # int64 row indices (into the training subset)
    cost: int
    key: object
    negatives: Optional[np.ndarray] = None  # needed only with min_union_purity


def budgeted_greedy(
    candidates: Sequence[Candidate], budgets: Sequence[int], n_train: int,
    min_union_purity: Optional[float] = None, disjoint: bool = False,
) -> Dict[int, Tuple[List[int], int]]:
    """
    Budgeted maximum coverage by the cost-effectiveness greedy with lazy
    evaluation, run once per budget, each result compared with the best
    single affordable candidate and the better kept (Khuller, Moss & Naor,
    1999: the combination is a (1 - 1/e) / 2 approximation). Returns, per
    budget, (chosen candidate indices in order, covered training positives).
    Coverage is monotone submodular, so a stale heap entry's ratio is an
    upper bound on its current ratio and lazy re-evaluation is exact.

    `min_union_purity`: a candidate whose addition would take the union's
    training purity below it is skipped for the rest of that budget's run
    (a heuristic: the guarantee above no longer applies).

    `disjoint`: a candidate sharing any training row (positive or negative)
    with an already chosen one is skipped. Needs the candidates' negatives.
    The union of disjoint groups of purity >= tau has purity >= tau.
    """
    if (min_union_purity is not None or disjoint) and any(c.negatives is None for c in candidates):
        raise ValueError("min_union_purity / disjoint need the candidates' negatives")
    out: Dict[int, Tuple[List[int], int]] = {}
    sizes = np.array([c.positives.size for c in candidates], dtype=np.int64)
    costs = np.array([c.cost for c in candidates], dtype=np.int64)
    for budget in budgets:
        covered = np.zeros(n_train, dtype=bool)
        covered_neg = np.zeros(n_train, dtype=bool)
        occupied = np.zeros(n_train, dtype=bool)
        n_neg = 0
        heap: List[Tuple[float, int, int]] = [
            (-(sizes[i] / costs[i]), -int(sizes[i]), i)
            for i in range(len(candidates)) if sizes[i] > 0 and costs[i] <= budget
        ]
        heapq.heapify(heap)
        chosen: List[int] = []
        spent = 0
        total = 0
        while heap:
            neg_ratio, _, i = heapq.heappop(heap)
            if costs[i] > budget - spent:
                continue
            if disjoint:
                neg_d = candidates[i].negatives
                assert neg_d is not None
                if occupied[candidates[i].positives].any() or occupied[neg_d].any():
                    continue
            gain = int(np.count_nonzero(~covered[candidates[i].positives]))
            if gain == 0:
                continue
            ratio = gain / costs[i]
            if heap and ratio < -heap[0][0] - 1e-12:
                heapq.heappush(heap, (-ratio, -gain, i))
                continue
            if min_union_purity is not None:
                neg = candidates[i].negatives
                assert neg is not None
                new_neg = int(np.count_nonzero(~covered_neg[neg]))
                if (total + gain) < min_union_purity * (total + gain + n_neg + new_neg) - 1e-9:
                    continue
                covered_neg[neg] = True
                n_neg += new_neg
            chosen.append(i)
            covered[candidates[i].positives] = True
            if disjoint:
                occupied[candidates[i].positives] = True
                occupied[candidates[i].negatives] = True  # type: ignore[index]
            spent += int(costs[i])
            total += gain
        affordable = np.flatnonzero((costs <= budget) & (sizes > 0))
        if affordable.size:
            best_single = int(affordable[np.lexsort((costs[affordable], -sizes[affordable]))[0]])
            if sizes[best_single] > total:
                chosen, total = [best_single], int(sizes[best_single])
        out[int(budget)] = (chosen, total)
    return out


def qualifying(
    k: np.ndarray, n: np.ndarray, tau: float, m: int, threshold: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Groups with n >= m and, when `threshold` is None, k / n >= tau (the
    product's observed-purity rule); otherwise k >= threshold[n] (a
    size-dependent count, see `certification_threshold`).
    """
    n = np.asarray(n, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    if threshold is None:
        need = np.maximum(np.ceil(tau * n.astype(np.float64) - 1e-9).astype(np.int64), 1)
    else:
        if n.size and int(n.max()) >= threshold.size:
            raise ValueError(f"threshold covers n <= {threshold.size - 1}, got n = {int(n.max())}")
        need = threshold[n]
    return (n >= max(1, m)) & (n > 0) & (k >= need)


def certification_threshold(n_max: int, tau: float, alpha_eff: float) -> np.ndarray:
    """
    Per n = 0..n_max, the smallest positive count certified at `alpha_eff`
    for rows with possibly different success probabilities
    (`vsf.centers.min_successes_to_certify_heterogeneous`; n + 1 = never).
    """
    sizes = np.arange(n_max + 1, dtype=np.int64)
    out = np.asarray(min_successes_to_certify_heterogeneous(sizes, tau, alpha_eff), dtype=np.int64)
    out[0] = 1
    return out


def rule_candidates(
    X_train: np.ndarray, z64: np.ndarray, tau: float, m: int, max_len: int,
    threshold: Optional[np.ndarray], qualify: bool = True, top_k: Optional[int] = None,
) -> List[Candidate]:
    """
    Every conjunction of <= max_len (column = value) conditions that occurs
    in the training rows, one candidate per distinct training row set
    (described by the fewest conditions). With `qualify` the purity rule (or
    `threshold`) and the size floor filter the cells; without it only the
    size floor does, and `top_k` keeps the cells with the largest WRAcc
    (n/N)(k/n - P/N) so that the unfiltered family stays bounded. Candidate
    keys are (columns, values).
    """
    n_train, n_features = X_train.shape
    base_rate = float(z64.sum()) / max(1, n_train)
    max_len = min(max_len, n_features)
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    # (wracc, -cost, combo, cell): a heap of the best top_k cells when top_k is set
    heap: List[Tuple[float, int, Tuple[int, ...], int]] = []
    candidates: List[Candidate] = []
    seen: Dict[bytes, int] = {}
    for d in range(1, max_len + 1):
        for combo in itertools.combinations(range(n_features), d):
            codes, n_cells = cell_codes(X_train[:, combo])
            codes = np.asarray(codes, dtype=np.int64)
            table = np.bincount(codes * 2 + z64, minlength=2 * n_cells).reshape(n_cells, 2)
            n = table.sum(axis=1)
            k = table[:, 1]
            if qualify:
                q = np.flatnonzero(qualifying(k, n, tau, m, threshold))
            else:
                q = np.flatnonzero(n >= max(1, m))
            if q.size == 0:
                continue
            if top_k is None:
                order = np.argsort(codes, kind="stable")
                bounds = np.searchsorted(codes[order], np.arange(n_cells + 1))
                for c in q.tolist():
                    rows = order[bounds[c]:bounds[c + 1]]
                    key = rows.tobytes()
                    if key in seen:
                        continue
                    seen[key] = len(candidates)
                    values = tuple(int(v) for v in X_train[rows[0], list(combo)])
                    candidates.append(Candidate(
                        positives=rows[z64[rows] == 1], cost=len(combo), key=(combo, values),
                        negatives=rows[z64[rows] == 0],
                    ))
                continue
            wr = (k[q] - n[q] * base_rate) / n_train
            for c, w in zip(q.tolist(), wr.tolist()):
                item = (float(w), -d, combo, int(c))
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    if top_k is not None:
        by_combo: Dict[Tuple[int, ...], List[int]] = {}
        for _, _, combo, c in heap:
            by_combo.setdefault(combo, []).append(c)
        picks: List[Tuple[Tuple[int, ...], int]] = []
        for combo, cells in by_combo.items():
            picks.extend((combo, c) for c in cells)
        picks.sort(key=lambda t: (len(t[0]), t[0], t[1]))
        for combo, c in picks:
            codes, n_cells = cell_codes(X_train[:, combo])
            codes = np.asarray(codes, dtype=np.int64)
            order = np.argsort(codes, kind="stable")
            bounds = np.searchsorted(codes[order], np.arange(n_cells + 1))
            rows = order[bounds[c]:bounds[c + 1]]
            key = rows.tobytes()
            if key in seen:
                continue
            seen[key] = len(candidates)
            values = tuple(int(v) for v in X_train[rows[0], list(combo)])
            candidates.append(Candidate(
                positives=rows[z64[rows] == 1], cost=len(combo), key=(combo, values),
                negatives=rows[z64[rows] == 0],
            ))
    return candidates


@dataclass(frozen=True)
class FreeRule:
    """One selected rule, in selection order, with its counts on all rows."""

    features: Tuple[int, ...]   # feature indices, ascending
    rows: np.ndarray            # int64 indices of the rows the rule holds
    n: int                      # rows
    k: int                      # rows of the value among them
    new_positives: int          # rows of the value this rule added to the union

    @property
    def cost(self) -> int:
        return len(self.features)

    @property
    def purity(self) -> float:
        return self.k / self.n if self.n else 0.0


@dataclass(frozen=True)
class FreeRuleSet:
    """
    The rules `select_free_rules` chose and what their union holds, all on
    the rows they were chosen on.
    """

    rules: Tuple[FreeRule, ...]
    budget: int
    n_candidates: int           # qualifying conjunctions the greedy chose from
    n_samples: int
    n_positive: int
    feature_names: Tuple[str, ...]

    @property
    def conditions(self) -> int:
        """Conditions spent (<= budget)."""
        return int(sum(r.cost for r in self.rules))

    @property
    def k_covered(self) -> int:
        return int(sum(r.new_positives for r in self.rules))

    @property
    def n_covered(self) -> int:
        """Rows inside the union. The rules are disjoint, so sizes add."""
        return int(sum(r.n for r in self.rules))

    @property
    def coverage(self) -> float:
        return self.k_covered / self.n_positive if self.n_positive else 0.0

    @property
    def purity(self) -> float:
        return self.k_covered / self.n_covered if self.n_covered else 0.0


class FreeRuleFamily:
    """
    The qualifying conjunctions of one (X, Z, certificate) - the candidates
    `rule_candidates` returns on all rows - kept so that several budgets can
    be selected from them without enumerating again. `select(budget)` is
    `budgeted_greedy(candidates, [budget], n, disjoint=True)`.
    """

    def __init__(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        positive_class: Optional[object] = None,
        center_spec: Optional[CenterSpec] = None,
        direction: Direction = "presence",
        max_len: int = MAX_BRANCH_D,
    ) -> None:
        if not (1 <= max_len <= MAX_BRANCH_D):
            raise ValueError(f"max_len must be in [1, {MAX_BRANCH_D}], got {max_len}")
        spec = center_spec if center_spec is not None else CenterSpec()
        if spec.rule == "certified" and spec.multiplicity != "family":
            raise ValueError(
                "free rules under a certificate need multiplicity='family': a per-schema "
                "correction is not a function of the cell size alone, and a free rule has no schema"
            )
        self.spec = spec
        self.candidates: List[Candidate] = []
        self.n_samples = int(np.asarray(X).shape[0])
        self.n_positive = 0
        self.feature_names: Tuple[str, ...] = ()
        prepared = _prepare_search(X, Z, feature_names, positive_class, spec, direction)
        if prepared is None:
            return
        factory, z_binary, names = prepared
        X_all = factory._raw.astype(np.int64, copy=False)
        z64 = z_binary.astype(np.int64)
        self.n_samples = int(z64.shape[0])
        self.n_positive = int(z64.sum())
        self.feature_names = tuple(names)
        max_len = min(max_len, factory.n_features)
        threshold: Optional[np.ndarray] = None
        if spec.rule == "certified":
            resolved = resolve_center_spec(factory, spec, max_len)
            threshold = certification_threshold(self.n_samples, spec.tau, resolved.effective_alpha(0))
        self.candidates = rule_candidates(
            X_all, z64, spec.tau, max(1, spec.min_samples), max_len, threshold,
        )

    def select(self, budget: int) -> FreeRuleSet:
        """The disjoint rules chosen under `budget` conditions, in selection order."""
        budget = int(budget)
        if budget < 1:
            raise ValueError(f"budget must be a positive number of conditions, got {budget}")
        chosen, _ = budgeted_greedy(self.candidates, [budget], self.n_samples, disjoint=True)[budget]
        covered = np.zeros(self.n_samples, dtype=bool)
        rules: List[FreeRule] = []
        for i in chosen:
            cand = self.candidates[i]
            assert cand.negatives is not None
            combo, _values = cand.key  # type: ignore[misc]
            gain = int(np.count_nonzero(~covered[cand.positives]))
            covered[cand.positives] = True
            rows = np.sort(np.concatenate([cand.positives, cand.negatives]))
            rules.append(FreeRule(
                features=tuple(int(j) for j in combo), rows=rows, n=int(rows.size),
                k=int(cand.positives.size), new_positives=gain,
            ))
        return FreeRuleSet(
            rules=tuple(rules), budget=budget, n_candidates=len(self.candidates),
            n_samples=self.n_samples, n_positive=self.n_positive, feature_names=self.feature_names,
        )


def select_free_rules(
    X: np.ndarray,
    Z: np.ndarray,
    budget: int,
    feature_names: Optional[Sequence[str]] = None,
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    direction: Direction = "presence",
    max_len: int = MAX_BRANCH_D,
) -> FreeRuleSet:
    """
    The best set of pairwise disjoint rules of <= `max_len` conditions under
    a budget of `budget` conditions in total, on ALL rows of (X, Z): the
    `rules_disjoint` baseline with the whole table as its training fold.

    `center_spec` supplies tau, the size floor (`min_samples`) and the rule:
    `rule="purity"` qualifies a conjunction by its observed share;
    `rule="certified"` (with `multiplicity="family"`, the only certificate
    that is a function of the cell size alone) by the count threshold of the
    family-wise certificate at alpha / T, T the family of the search over
    d <= max_len on these rows - exactly `baselines.run_split(selection=
    "certified")`. X is prepared as every search prepares it
    (`vsf.avr._prepare_search`), so feature indices are those of the
    Landscape computed from the same arguments.
    """
    if int(budget) < 1:
        raise ValueError(f"budget must be a positive number of conditions, got {budget}")
    return FreeRuleFamily(
        X, Z, feature_names, positive_class, center_spec, direction, max_len,
    ).select(budget)
