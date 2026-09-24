"""
Budget-matched comparison of VSF with other ways of describing where a
target value lives (PLAN.md, phase 3.2).

Every method answers the same question on a training fold: "with at most B
conditions in total, which groups of rows hold as many rows of the value as
possible, each group at observed purity >= tau and size >= m?" and hands
back its groups as predicates on ALL rows, so the held-out fold is scored
by the same code for every method.

A condition is one (column, value-set) test a reader has to read. A VSF
cell of a d-column schema costs d; a rule of length L costs L; a
decision-tree leaf costs its depth (a split on a one-hot column is one
condition, "column = v" or "column != v").

Methods
-------
`vsf`          every schema of d <= 4 (training-only encoding,
               `vsf.avr.apply_fitted_partition`), its qualifying cells taken
               by descending positive count; at each budget the best schema
               on the training fold. One grid, all groups on the same columns.
`vsf_greedy`   the same, restricted to the nested chain S1 < S2 < S3 < S4
               built by adding one column at a time by the product's own
               ranking key - the construction claim C1 argues against.
`rules`        every conjunction of <= 4 (column = value) conditions that
               occurs in the training rows (the cells of every uncoarsened
               partition), qualifying ones selected by budgeted greedy
               maximum coverage (marginal new positives per condition, lazy
               evaluation, compared with the best single affordable rule).
               No common grid; groups may overlap.
`rules_pure`   `rules` with one more constraint: a rule is added only if the
               union of the chosen rules keeps training purity >= tau (a
               rule rejected once is not reconsidered). Overlapping rules
               that are each >= tau can form a union below tau; groups of
               the other methods are disjoint, so their unions stay >= tau
               by construction. This is the like-for-like rule baseline.
`vsf_partial`  one schema of d <= 4 as `vsf`, but a centre may fix only a
               subset of the schema's axes (a cell of a sub-grid; a bar,
               plate or layer of the same cube - `vsf.partial`). Cost =
               axes fixed. Disjoint greedy packing under the budget, so the
               union of the chosen centres keeps purity >= tau by
               construction. Candidate row sets all belong to the VSF family
               (they are cells of schemas of d <= 4), so the certificate
               applies unchanged.
`rules_disjoint` `rules` with the constraint that chosen rules share no
               training row: the same candidate family as `rules`, selected
               like `vsf` / `vsf_partial`. This is the like-for-like free
               baseline for a method whose groups are disjoint: `rules` and
               `rules_pure` may overlap, and an overlapping union of rules
               each >= tau can fall below tau (rules_pure guards this on the
               training fold only).
`ssdpp`        SSD++ (Proenca et al., 2022; package `rulelist`): an MDL rule
               list for the two-class target on the training rows, beam 30,
               depth <= max_d, min_support = m, at most 40 rules. A rule list is ordered
               ("else if"), so rule i applies to the rows not matched by
               rules 1..i-1: its groups are disjoint by construction. The
               qualifying rules (training purity >= tau, or the certificate
               threshold) are then chosen under the budget by the same greedy
               as the others; cost = conditions in the rule. The list is
               fitted for the whole target, so it also contains rules for
               the other class; those never qualify.
`tree`         one CART tree (scikit-learn) on one-hot training rows,
               min_samples_leaf = m; qualifying leaves selected by the same
               greedy. At each budget the tree with the most covered
               training positives over the grid max_depth 1..4 x criterion
               {entropy, gini} x class_weight {None, balanced} (balanced
               weights push splits towards the minority class, which is
               usually the target).

All selections use training rows only. Evaluation: union of the selected
groups on the held-out rows, from the counts k (positives inside), n (rows
inside) and P (held-out positives):

coverage       k / P
purity         k / n
net coverage   (k - tau / (1 - tau) * (n - k)) / P. The Lagrangian of
               "cover the most positives subject to purity >= tau": a group
               of purity exactly tau adds 0, a union below tau scores < 0,
               a pure union scores its coverage. It compares methods whose
               held-out purities differ, which coverage alone cannot.
stability      mean pairwise Jaccard index, over the splits, of the sets of
               ALL rows the selected union contains (pairs where both
               unions are empty are skipped): does the reader get the same
               description from a different sample?
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from vsf.avr import (
    MAX_BRANCH_D,
    _CandidateFactory,
    _prepare_search,
    apply_fitted_partition,
    resolve_center_spec,
)
from vsf.centers import (
    CenterSpec,
    coverage_score,
    min_successes_to_certify_heterogeneous,
    stratified_repeated_kfold,
    summarize_cv,
)
from vsf.metrics import cell_codes

__all__ = [
    "BUDGETS",
    "Group",
    "METHODS",
    "MethodResult",
    "budgeted_greedy",
    "certification_threshold",
    "Selection",
    "description_stability",
    "held_out_counts",
    "net_coverage",
    "union_members",
    "evaluate_groups",
    "SplitOutcome",
    "run_comparison",
    "run_split",
    "summarize_splits",
    "select_rules",
    "select_tree",
    "select_vsf",
    "select_vsf_partial",
    "select_ssdpp",
    "TREE_GRID",
]

#: Condition budgets every method is evaluated at.
BUDGETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class Group:
    """A selected group: its members among ALL rows and its description cost."""

    members: np.ndarray  # bool, one entry per row of the dataset
    cost: int
    label: str


@dataclass
class MethodResult:
    """Per budget, the groups a method selected on one training fold."""

    groups: Dict[int, List[Group]] = field(default_factory=dict)
    train_coverage: Dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _Candidate:
    """A qualifying group on the training rows: its positive training rows and cost."""

    positives: np.ndarray  # int64 row indices (into the training subset)
    cost: int
    key: object
    negatives: Optional[np.ndarray] = None  # needed only with min_union_purity


# --------------------------------------------------------------------------
# Budgeted selection
# --------------------------------------------------------------------------
def budgeted_greedy(
    candidates: Sequence[_Candidate], budgets: Sequence[int], n_train: int,
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


def _qualifying(
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


#: Selection rules a comparison can run under.
Selection = Literal["purity", "certified"]


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


# --------------------------------------------------------------------------
# VSF
# --------------------------------------------------------------------------
def _schema_table(
    fit: _CandidateFactory, z_train: np.ndarray, combo: Tuple[int, ...], codes: np.ndarray,
    n_cells: int, tau: float, m: int, threshold: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(qualifying cell ids sorted by positives desc then size asc, their positives)."""
    table = np.bincount(codes * 2 + z_train, minlength=2 * n_cells).reshape(n_cells, 2)
    n = table.sum(axis=1)
    k = table[:, 1]
    q = np.flatnonzero(_qualifying(k, n, tau, m, threshold))
    order = np.lexsort((n[q], -k[q]))
    return q[order], k[q][order]


def _vsf_groups(
    fit: _CandidateFactory, X_all: np.ndarray, train: np.ndarray, combo: Tuple[int, ...],
    cells: np.ndarray, names: Sequence[str],
) -> List[Group]:
    codes_fit, _ = fit.codes(combo)
    codes_all, _ = apply_fitted_partition(fit, X_all, combo)
    groups: List[Group] = []
    for c in cells.tolist():
        first = int(np.flatnonzero(codes_fit == c)[0])
        image = int(codes_all[train[first]])
        groups.append(Group(
            members=codes_all == image, cost=len(combo),
            label="+".join(names[j] for j in combo) + f"#{c}",
        ))
    return groups


def select_vsf(
    fit: _CandidateFactory, X_all: np.ndarray, train: np.ndarray, z_train: np.ndarray,
    tau: float, m: int, budgets: Sequence[int], names: Sequence[str],
    combos: Optional[Sequence[Tuple[int, ...]]] = None, max_d: int = MAX_BRANCH_D,
    threshold: Optional[np.ndarray] = None,
) -> MethodResult:
    """
    Best single schema per budget: the schema whose top floor(B / d)
    qualifying cells hold the most training positives (ties: fewer cells,
    then fewer training rows, then enumeration order). `combos` restricts
    the family (the greedy chain); None means every schema of d <= 4.
    """
    z64 = z_train.astype(np.int64)
    n_pos = max(1, int(z64.sum()))
    best: Dict[int, Tuple[Tuple[int, int], Tuple[int, ...], np.ndarray]] = {}
    source: Sequence[Tuple[Tuple[int, ...], np.ndarray, int]]
    if combos is None:
        source = fit.iter_candidates(min(max_d, fit.n_features))
    else:
        source = [(tuple(c),) + fit.codes(tuple(c)) for c in combos]
    for combo, codes, n_cells in source:
        if n_cells == 0:
            continue
        cells, k_sorted = _schema_table(fit, z64, tuple(combo), codes, n_cells, tau, m, threshold)
        if cells.size == 0:
            continue
        cum = np.cumsum(k_sorted)
        d = len(combo)
        for budget in budgets:
            c = min(int(budget) // d, int(cells.size))
            if c == 0:
                continue
            score = (int(cum[c - 1]), -c)
            incumbent = best.get(int(budget))
            if incumbent is None or score > incumbent[0]:
                best[int(budget)] = (score, tuple(combo), cells[:c])
    result = MethodResult()
    for budget in budgets:
        hit = best.get(int(budget))
        if hit is None:
            result.groups[int(budget)] = []
            result.train_coverage[int(budget)] = 0.0
            continue
        (covered, _), combo, cells = hit
        result.groups[int(budget)] = _vsf_groups(fit, X_all, train, combo, cells, names)
        result.train_coverage[int(budget)] = covered / n_pos
    return result


def select_vsf_partial(
    fit: _CandidateFactory, X_all: np.ndarray, train: np.ndarray, z_train: np.ndarray,
    tau: float, m: int, budgets: Sequence[int], names: Sequence[str],
    max_d: int = MAX_BRANCH_D, threshold: Optional[np.ndarray] = None,
) -> MethodResult:
    """
    Best single schema per budget with PARTIAL centres: the candidates of a
    schema S are the qualifying cells of every sub-schema S' of S (cost |S'|),
    packed disjointly by `budgeted_greedy(disjoint=True)`; at each budget the
    schema whose packing holds the most training positives (ties: fewer
    centres, enumeration order). Every candidate is a cell of a schema of
    d <= max_d, i.e. a member of the family the certificate counts.
    """
    z64 = z_train.astype(np.int64)
    n_train = int(train.size)
    n_pos = max(1, int(z64.sum()))
    max_d = min(max_d, fit.n_features)
    # qualifying cells per combo, as _Candidate lists (empty when none)
    per_combo: Dict[Tuple[int, ...], List[_Candidate]] = {}
    for combo, codes, n_cells in fit.iter_candidates(max_d):
        combo = tuple(combo)
        if n_cells == 0:
            per_combo[combo] = []
            continue
        cells, _ = _schema_table(fit, z64, combo, codes, n_cells, tau, m, threshold)
        if cells.size == 0:
            per_combo[combo] = []
            continue
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(n_cells + 1))
        lst: List[_Candidate] = []
        for c in cells.tolist():
            rows = order[bounds[c]:bounds[c + 1]]
            lst.append(_Candidate(
                positives=rows[z64[rows] == 1], cost=len(combo), key=(combo, int(c)),
                negatives=rows[z64[rows] == 0],
            ))
        per_combo[combo] = lst
    best: Dict[int, Tuple[Tuple[int, int], Tuple[int, ...], List[_Candidate]]] = {}
    seen_pools: set = set()
    for d in range(1, max_d + 1):
        for combo in itertools.combinations(range(fit.n_features), d):
            live = tuple(
                sub for r in range(1, d + 1) for sub in itertools.combinations(combo, r)
                if per_combo.get(sub)
            )
            if not live or live in seen_pools:
                continue
            seen_pools.add(live)
            pool: List[_Candidate] = []
            seen_rows: Dict[bytes, int] = {}
            for sub in live:
                for cand in per_combo[sub]:
                    rows = np.sort(np.concatenate([cand.positives, cand.negatives]))  # type: ignore[list-item]
                    key = rows.tobytes()
                    j = seen_rows.get(key)
                    if j is None:
                        seen_rows[key] = len(pool)
                        pool.append(cand)
                    elif cand.cost < pool[j].cost:
                        pool[j] = cand
            chosen = budgeted_greedy(pool, budgets, n_train, disjoint=True)
            for budget, (idx, covered) in chosen.items():
                if not idx:
                    continue
                score = (int(covered), -len(idx))
                inc = best.get(int(budget))
                if inc is None or score > inc[0]:
                    best[int(budget)] = (score, combo, [pool[i] for i in idx])
    result = MethodResult()
    for budget in budgets:
        hit = best.get(int(budget))
        if hit is None:
            result.groups[int(budget)] = []
            result.train_coverage[int(budget)] = 0.0
            continue
        (covered, _), _, cands = hit
        groups: List[Group] = []
        for cand in cands:
            sub, cell = cand.key  # type: ignore[misc]
            groups.extend(_vsf_groups(fit, X_all, train, tuple(sub), np.array([cell]), names))
        result.groups[int(budget)] = groups
        result.train_coverage[int(budget)] = covered / n_pos
    return result


def select_ssdpp(
    X_all: np.ndarray, train: np.ndarray, z_train: np.ndarray, tau: float, m: int,
    budgets: Sequence[int], names: Sequence[str], max_d: int = MAX_BRANCH_D,
    threshold: Optional[np.ndarray] = None, beam_width: int = 30, max_rules: int = 40,
) -> MethodResult:
    """
    SSD++ rule list (package `rulelist`) on the training rows; see the module
    docstring. `max_rules` caps the list at 40 rules: no budget here exceeds
    32 conditions, so at most 32 rules can ever be selected, and the MDL
    search on tables such as nursery otherwise grows lists of 80+ rules at
    a minute per fit.
    """
    import pandas as pd
    from rulelist import RuleList
    n_train = int(train.size)
    z64 = z_train.astype(np.int64)
    X_train = X_all[train]
    cols = [f"c{j}" for j in range(X_all.shape[1])]
    frame = pd.DataFrame({c: X_train[:, j].astype(str) for j, c in enumerate(cols)})
    target = pd.DataFrame({"y": np.where(z64 == 1, "pos", "neg")})
    model = RuleList(target_model="categorical", task="discovery", max_depth=max_d,
                     beam_width=beam_width, min_support=max(1, m), max_rules=max_rules)
    model.fit(frame, target)
    candidates: List[_Candidate] = []
    members_all: List[np.ndarray] = []
    covered_all = np.zeros(X_all.shape[0], dtype=bool)
    for sg in model._rulelist.subgroups:
        conds = [(int(str(it.parent_variable)[1:]), str(it.activation_function.keywords["category"]))
                 for it in sg.pattern]
        match = np.ones(X_all.shape[0], dtype=bool)
        for j, v in conds:
            match &= X_all[:, j].astype(str) == v
        members = match & ~covered_all          # "else if" semantics on all rows
        covered_all |= match
        rows = np.flatnonzero(members[train])
        if rows.size == 0:
            continue
        k = int(z64[rows].sum())
        if not bool(_qualifying(np.array([k]), np.array([rows.size]), tau, m, threshold)[0]):
            continue
        candidates.append(_Candidate(positives=rows[z64[rows] == 1], cost=len(conds),
                                     key=len(members_all), negatives=rows[z64[rows] == 0]))
        members_all.append(members)
        _ = names
    chosen = budgeted_greedy(candidates, budgets, n_train, disjoint=True)
    n_pos = max(1, int(z64.sum()))
    result = MethodResult()
    for budget, (idx, covered) in chosen.items():
        groups = [Group(members=members_all[int(candidates[i].key)], cost=candidates[i].cost,  # type: ignore[arg-type]
                        label=f"ssdpp#{candidates[i].key}") for i in idx]
        result.groups[budget] = groups
        result.train_coverage[budget] = covered / n_pos
    return result


def greedy_chain(
    fit: _CandidateFactory, z_train: np.ndarray, spec: CenterSpec, max_d: int = MAX_BRANCH_D,
) -> List[Tuple[int, ...]]:
    """S1 < S2 < ... < S_max_d: add the column that maximises `coverage_score`."""
    chain: List[Tuple[int, ...]] = []
    current: Tuple[int, ...] = ()
    for _ in range(min(max_d, fit.n_features)):
        best_key = None
        best_combo: Optional[Tuple[int, ...]] = None
        for j in range(fit.n_features):
            if j in current:
                continue
            combo = tuple(sorted(current + (j,)))
            codes, n_cells = fit.codes(combo)
            key = coverage_score(z_train, codes, n_cells, spec)
            if best_key is None or key > best_key:
                best_key, best_combo = key, combo
        assert best_combo is not None
        current = best_combo
        chain.append(current)
    return chain


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
def select_rules(
    X_all: np.ndarray, train: np.ndarray, z_train: np.ndarray, tau: float, m: int,
    budgets: Sequence[int], names: Sequence[str], max_len: int = MAX_BRANCH_D,
    pure_union: bool = False, threshold: Optional[np.ndarray] = None, disjoint: bool = False,
) -> MethodResult:
    """Qualifying conjunctions of <= max_len (column = value) conditions, greedy under budget."""
    n_train = int(train.size)
    z64 = z_train.astype(np.int64)
    X_train = X_all[train]
    candidates: List[_Candidate] = []
    seen: Dict[bytes, int] = {}
    max_len = min(max_len, X_all.shape[1])
    for d in range(1, max_len + 1):
        for combo in itertools.combinations(range(X_all.shape[1]), d):
            codes, n_cells = cell_codes(X_train[:, combo])
            codes = np.asarray(codes, dtype=np.int64)
            table = np.bincount(codes * 2 + z64, minlength=2 * n_cells).reshape(n_cells, 2)
            q = np.flatnonzero(_qualifying(table[:, 1], table.sum(axis=1), tau, m, threshold))
            if q.size == 0:
                continue
            order = np.argsort(codes, kind="stable")
            bounds = np.searchsorted(codes[order], np.arange(n_cells + 1))
            for c in q.tolist():
                rows = order[bounds[c]:bounds[c + 1]]
                key = rows.tobytes()
                if key in seen:  # same training rows, described by fewer or equal conditions
                    continue
                seen[key] = len(candidates)
                values = tuple(int(v) for v in X_train[rows[0], list(combo)])
                candidates.append(_Candidate(
                    positives=rows[z64[rows] == 1], cost=d, key=(combo, values),
                    negatives=rows[z64[rows] == 0],
                ))
    chosen = budgeted_greedy(candidates, budgets, n_train,
                             min_union_purity=tau if pure_union else None, disjoint=disjoint)
    n_pos = max(1, int(z64.sum()))
    result = MethodResult()
    for budget, (idx, covered) in chosen.items():
        groups = []
        for i in idx:
            combo, values = candidates[i].key  # type: ignore[misc]
            members = np.all(X_all[:, list(combo)] == np.asarray(values), axis=1)
            groups.append(Group(members=members, cost=candidates[i].cost, label=" & ".join(
                f"{names[j]}={v}" for j, v in zip(combo, values))))
        result.groups[budget] = groups
        result.train_coverage[budget] = covered / n_pos
    return result


# --------------------------------------------------------------------------
# Decision tree
# --------------------------------------------------------------------------
#: (criterion, class_weight) settings tried for the single tree.
TREE_GRID: Tuple[Tuple[str, Optional[str]], ...] = (
    ("entropy", None), ("gini", None), ("entropy", "balanced"), ("gini", "balanced"),
)

def select_tree(
    X_all: np.ndarray, train: np.ndarray, z_train: np.ndarray, tau: float, m: int,
    budgets: Sequence[int], random_state: int = 0, max_depth: int = MAX_BRANCH_D,
    threshold: Optional[np.ndarray] = None,
) -> MethodResult:
    """
    One CART tree on one-hot training rows, per budget the best over
    `TREE_GRID` x depth 1..max_depth by covered training positives (ties:
    first in grid order); its qualifying leaves chosen greedily under the
    budget.
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    H_train = encoder.fit_transform(X_all[train])
    H_all = encoder.transform(X_all)
    z64 = z_train.astype(np.int64)
    n_pos = max(1, int(z64.sum()))
    best: Dict[int, Tuple[int, List[Group]]] = {}
    for (criterion, class_weight), depth in itertools.product(TREE_GRID, range(1, max_depth + 1)):
        tree = DecisionTreeClassifier(
            criterion=criterion, class_weight=class_weight, max_depth=depth,
            min_samples_leaf=max(1, m), random_state=random_state,
        ).fit(H_train, z64)
        leaf_train = tree.apply(H_train)
        leaf_all = tree.apply(H_all)
        node_depth = _node_depths(tree)
        candidates: List[_Candidate] = []
        leaves = np.unique(leaf_train)
        for leaf in leaves.tolist():
            rows = np.flatnonzero(leaf_train == leaf)
            k = int(z64[rows].sum())
            if not bool(_qualifying(np.array([k]), np.array([rows.size]), tau, m, threshold)[0]):
                continue
            candidates.append(_Candidate(positives=rows[z64[rows] == 1],
                                         cost=max(1, int(node_depth[leaf])), key=leaf))
        chosen = budgeted_greedy(candidates, budgets, int(train.size))
        for budget, (idx, covered) in chosen.items():
            if budget in best and best[budget][0] >= covered:
                continue
            groups = [Group(members=leaf_all == candidates[i].key, cost=candidates[i].cost,
                            label=f"{criterion}-{class_weight}-depth{depth}-leaf{candidates[i].key}")
                      for i in idx]
            best[budget] = (covered, groups)
    result = MethodResult()
    for budget in budgets:
        covered, groups = best.get(int(budget), (0, []))
        result.groups[int(budget)] = groups
        result.train_coverage[int(budget)] = covered / n_pos
    return result


def _node_depths(tree: object) -> np.ndarray:
    t = tree.tree_  # type: ignore[attr-defined]
    depth = np.zeros(t.node_count, dtype=np.int64)
    stack = [0]
    while stack:
        node = stack.pop()
        for child in (t.children_left[node], t.children_right[node]):
            if child != -1:
                depth[child] = depth[node] + 1
                stack.append(int(child))
    return depth


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def held_out_counts(inside: np.ndarray, z: np.ndarray, test: np.ndarray) -> Tuple[int, int, int]:
    """(positives inside, rows inside, positives) among the held-out rows."""
    inside_test = inside[test]
    pos_test = z[test] == 1
    return (int(np.count_nonzero(inside_test & pos_test)), int(np.count_nonzero(inside_test)),
            int(np.count_nonzero(pos_test)))


def evaluate_groups(groups: Sequence[Group], z: np.ndarray, test: np.ndarray) -> Tuple[float, float, int]:
    """(held-out coverage, held-out purity or NaN, conditions spent)."""
    k_in, n_in, n_pos = held_out_counts(union_members(groups, z.shape[0]), z, test)
    return (k_in / n_pos if n_pos else 0.0,
            (k_in / n_in) if n_in else float("nan"),
            int(sum(g.cost for g in groups)))


METHODS: Tuple[str, ...] = ("vsf", "vsf_partial", "vsf_greedy", "rules", "rules_pure", "rules_disjoint", "tree", "ssdpp")


@dataclass(frozen=True, eq=False)
class SplitOutcome:
    """One method at one budget on one split."""

    k_in: int  # held-out positives inside the union
    n_in: int  # held-out rows inside the union
    n_pos: int  # held-out positives
    conditions: int
    train_coverage: float
    labels: Tuple[str, ...]
    inside: np.ndarray  # bool over ALL rows: membership in the selected union
    seconds: float  # selection time of the method on this split (all budgets)

    @property
    def coverage(self) -> float:
        return self.k_in / self.n_pos if self.n_pos else 0.0

    @property
    def purity(self) -> float:
        """NaN when no held-out row fell inside."""
        return self.k_in / self.n_in if self.n_in else float("nan")

    def net_coverage(self, tau: float) -> float:
        return net_coverage(self.k_in, self.n_in, self.n_pos, tau)


def net_coverage(k_in: int, n_in: int, n_pos: int, tau: float) -> float:
    """(k - tau / (1 - tau) * (n - k)) / P; 0 when there are no held-out positives."""
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    if not (0 <= k_in <= n_in):
        raise ValueError(f"need 0 <= k_in <= n_in, got {k_in}, {n_in}")
    if n_pos == 0:
        return 0.0
    return (k_in - tau / (1.0 - tau) * (n_in - k_in)) / n_pos


def description_stability(masks: Sequence[np.ndarray]) -> float:
    """Mean pairwise Jaccard index of boolean row masks; pairs of two empty masks are skipped (NaN if none left)."""
    if len(masks) < 2:
        return float("nan")
    M = np.vstack([np.asarray(m, dtype=bool) for m in masks]).astype(np.int64)
    inter = M @ M.T
    size = np.diag(inter)
    union = size[:, None] + size[None, :] - inter
    iu = np.triu_indices(M.shape[0], k=1)
    u = union[iu]
    keep = u > 0
    if not np.any(keep):
        return float("nan")
    return float(np.mean(inter[iu][keep] / u[keep]))


def union_members(groups: Sequence[Group], n_rows: int) -> np.ndarray:
    inside = np.zeros(n_rows, dtype=bool)
    for g in groups:
        inside |= g.members
    return inside


#: split -> method -> budget -> outcome
SplitResult = Dict[str, Dict[int, SplitOutcome]]


def run_split(
    X: np.ndarray, Z: np.ndarray, positive_class: object, tau: float, m: int,
    split_index: int, n_splits: int = 5, n_repeats: int = 5, random_state: int = 0,
    budgets: Sequence[int] = BUDGETS, methods: Sequence[str] = METHODS,
    max_d: int = MAX_BRANCH_D, selection: Selection = "purity", alpha: float = 0.05,
) -> SplitResult:
    """
    Every method on ONE split of `vsf.centers.stratified_repeated_kfold`
    (the same folds for every method, so results are paired). `max_d`
    bounds every method alike: schema dimensionality, rule length and tree
    depth (lowered only where the exhaustive families are too large to
    run, and then for all methods). Splits are independent, so a long
    comparison can be run split by split and summarised with
    `summarize_splits`.

    `selection="certified"`: a group qualifies only if its training count
    passes the family-wise certificate of the product
    (`CenterSpec(rule="certified", multiplicity="family")`) at `alpha`,
    with T the family of the VSF search over d <= max_d on the training
    fold. The same size-dependent threshold filters every method, so the
    held-out comparison is like for like. For `vsf` and the rules (whose row
    sets all belong to that family) it is a valid post-selection
    certificate; for `tree` it is only the same filter, since the leaves of
    a data-grown tree are not a pre-specified family.
    """
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if not (1 <= max_d <= MAX_BRANCH_D):
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}]")
    if selection not in ("purity", "certified"):
        raise ValueError(f"unknown selection {selection!r}")
    spec = CenterSpec(tau=tau, min_samples=max(1, m))
    prepared = _prepare_search(X, Z, None, positive_class, spec, "presence")
    if prepared is None:
        raise ValueError("no feature columns")
    factory, z, names = prepared
    X_all = factory._raw.astype(np.int64, copy=False)
    splits = stratified_repeated_kfold(z, n_splits, n_repeats, random_state)
    if not (0 <= split_index < len(splits)):
        raise ValueError(f"split_index must be in [0, {len(splits)})")
    train, test = splits[split_index]
    z_train = z[train]
    fit = _CandidateFactory(X_all[train], factory.bin_counts, int(train.size), ordered=factory.ordered)
    threshold: Optional[np.ndarray] = None
    chain_spec = spec
    family_seconds = 0.0
    if selection == "certified":
        t0 = time.perf_counter()
        chain_spec = resolve_center_spec(
            fit, CenterSpec(tau=tau, min_samples=max(1, m), rule="certified", multiplicity="family",
                            alpha=alpha),
            min(max_d, fit.n_features),
        )
        threshold = certification_threshold(int(train.size), tau, chain_spec.effective_alpha(0))
        family_seconds = time.perf_counter() - t0
    selectors: Dict[str, Callable[[], MethodResult]] = {
        "vsf": lambda: select_vsf(fit, X_all, train, z_train, tau, m, budgets, names, max_d=max_d,
                                  threshold=threshold),
        "vsf_partial": lambda: select_vsf_partial(fit, X_all, train, z_train, tau, m, budgets, names,
                                                  max_d=max_d, threshold=threshold),
        "vsf_greedy": lambda: select_vsf(fit, X_all, train, z_train, tau, m, budgets, names,
                                         combos=greedy_chain(fit, z_train, chain_spec, max_d),
                                         threshold=threshold),
        "rules": lambda: select_rules(X_all, train, z_train, tau, m, budgets, names, max_d,
                                      threshold=threshold),
        "rules_pure": lambda: select_rules(X_all, train, z_train, tau, m, budgets, names, max_d,
                                           pure_union=True, threshold=threshold),
        "rules_disjoint": lambda: select_rules(X_all, train, z_train, tau, m, budgets, names, max_d,
                                               threshold=threshold, disjoint=True),
        "tree": lambda: select_tree(X_all, train, z_train, tau, m, budgets, random_state, max_d,
                                    threshold=threshold),
        "ssdpp": lambda: select_ssdpp(X_all, train, z_train, tau, m, budgets, names, max_d,
                                      threshold=threshold),
    }
    out: SplitResult = {}
    for mth in methods:
        t0 = time.perf_counter()
        res = selectors[mth]()
        seconds = time.perf_counter() - t0 + family_seconds
        out[mth] = {}
        for b in budgets:
            groups = res.groups[b]
            spent = int(sum(g.cost for g in groups))
            if spent > b:
                raise AssertionError(f"{mth} spent {spent} conditions at budget {b}")
            inside = union_members(groups, z.shape[0])
            k_in, n_in, n_pos = held_out_counts(inside, z, test)
            out[mth][int(b)] = SplitOutcome(
                k_in=k_in, n_in=n_in, n_pos=n_pos, conditions=spent,
                train_coverage=res.train_coverage[b],
                labels=tuple(g.label for g in groups), inside=inside, seconds=seconds,
            )
    return out


def summarize_splits(
    per_split: Sequence[SplitResult], n_splits: int, n_repeats: int, tau: float,
) -> Dict[str, Dict[int, Dict[str, object]]]:
    """
    Per method and budget: Nadeau-Bengio summaries of coverage and net
    coverage (`CVCoverage`, paired across methods), per-split purities, the
    share of splits whose held-out purity reaches tau, description
    stability, mean conditions, training coverage and selection time.
    `per_split` in split order.
    """
    if len(per_split) != n_splits * n_repeats:
        raise ValueError(f"expected {n_splits * n_repeats} splits, got {len(per_split)}")
    summary: Dict[str, Dict[int, Dict[str, object]]] = {}
    for mth, per_budget in per_split[0].items():
        summary[mth] = {}
        for b in per_budget:
            outs = [r[mth][b] for r in per_split]
            cov = np.array([o.coverage for o in outs])
            pur = np.array([o.purity for o in outs])
            net = np.array([o.net_coverage(tau) for o in outs])
            reached = [o.k_in >= tau * o.n_in - 1e-9 for o in outs if o.n_in > 0]
            summary[mth][b] = {
                "coverage": summarize_cv(cov, pur, n_splits, n_repeats),
                "net_coverage": summarize_cv(net, pur, n_splits, n_repeats),
                "per_split_purity": tuple(pur.tolist()),
                "purity_reaches_tau": float(np.mean(reached)) if reached else float("nan"),
                "stability": description_stability([o.inside for o in outs]),
                "cost_mean": float(np.mean([o.conditions for o in outs])),
                "train_coverage_mean": float(np.mean([o.train_coverage for o in outs])),
                "seconds_mean": float(np.mean([o.seconds for o in outs])),
            }
    return summary


def run_comparison(
    X: np.ndarray, Z: np.ndarray, positive_class: object, tau: float, m: int,
    n_splits: int = 5, n_repeats: int = 5, random_state: int = 0,
    budgets: Sequence[int] = BUDGETS, methods: Sequence[str] = METHODS,
    max_d: int = MAX_BRANCH_D, selection: Selection = "purity", alpha: float = 0.05,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, object]:
    """All splits in one call (`run_split` then `summarize_splits`)."""
    total = n_splits * n_repeats
    per_split: List[SplitResult] = []
    for i in range(total):
        per_split.append(run_split(X, Z, positive_class, tau, m, i, n_splits, n_repeats,
                                   random_state, budgets, methods, max_d, selection, alpha))
        if progress is not None:
            progress(i + 1, total)
    labels = {mth: [{b: list(r[mth][b].labels) for b in budgets} for r in per_split] for mth in methods}
    return {"summary": summarize_splits(per_split, n_splits, n_repeats, tau), "labels": labels}
