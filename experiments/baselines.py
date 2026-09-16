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
`tree`         CART (scikit-learn, entropy) on one-hot training rows,
               max_depth 1..4, min_samples_leaf = m; qualifying leaves
               selected by the same greedy; at each budget the best depth
               on the training fold.

All selections use training rows only. Evaluation: union of the selected
groups on the held-out rows - coverage (share of held-out positives inside)
and purity (share of held-out rows inside that are positive).
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vsf.avr import (
    MAX_BRANCH_D,
    _CandidateFactory,
    _prepare_search,
    apply_fitted_partition,
)
from vsf.centers import CenterSpec, coverage_score, stratified_repeated_kfold, summarize_cv
from vsf.metrics import cell_codes

__all__ = [
    "BUDGETS",
    "Group",
    "METHODS",
    "MethodResult",
    "budgeted_greedy",
    "evaluate_groups",
    "SplitOutcome",
    "run_comparison",
    "run_split",
    "summarize_splits",
    "select_rules",
    "select_tree",
    "select_vsf",
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
    min_union_purity: Optional[float] = None,
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
    """
    if min_union_purity is not None and any(c.negatives is None for c in candidates):
        raise ValueError("min_union_purity needs the candidates' negatives")
    out: Dict[int, Tuple[List[int], int]] = {}
    sizes = np.array([c.positives.size for c in candidates], dtype=np.int64)
    costs = np.array([c.cost for c in candidates], dtype=np.int64)
    for budget in budgets:
        covered = np.zeros(n_train, dtype=bool)
        covered_neg = np.zeros(n_train, dtype=bool)
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
            spent += int(costs[i])
            total += gain
        affordable = np.flatnonzero((costs <= budget) & (sizes > 0))
        if affordable.size:
            best_single = int(affordable[np.lexsort((costs[affordable], -sizes[affordable]))[0]])
            if sizes[best_single] > total:
                chosen, total = [best_single], int(sizes[best_single])
        out[int(budget)] = (chosen, total)
    return out


def _qualifying(k: np.ndarray, n: np.ndarray, tau: float, m: int) -> np.ndarray:
    """Cells with n >= m and k / n >= tau (the product's observed-purity rule)."""
    need = np.ceil(tau * n.astype(np.float64) - 1e-9).astype(np.int64)
    return (n >= max(1, m)) & (n > 0) & (k >= np.maximum(need, 1))


# --------------------------------------------------------------------------
# VSF
# --------------------------------------------------------------------------
def _schema_table(
    fit: _CandidateFactory, z_train: np.ndarray, combo: Tuple[int, ...], codes: np.ndarray,
    n_cells: int, tau: float, m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """(qualifying cell ids sorted by positives desc then size asc, their positives)."""
    table = np.bincount(codes * 2 + z_train, minlength=2 * n_cells).reshape(n_cells, 2)
    n = table.sum(axis=1)
    k = table[:, 1]
    q = np.flatnonzero(_qualifying(k, n, tau, m))
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
        cells, k_sorted = _schema_table(fit, z64, tuple(combo), codes, n_cells, tau, m)
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
    pure_union: bool = False,
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
            q = np.flatnonzero(_qualifying(table[:, 1], table.sum(axis=1), tau, m))
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
                             min_union_purity=tau if pure_union else None)
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
def select_tree(
    X_all: np.ndarray, train: np.ndarray, z_train: np.ndarray, tau: float, m: int,
    budgets: Sequence[int], random_state: int = 0, max_depth: int = MAX_BRANCH_D,
) -> MethodResult:
    """CART of depth 1..max_depth on one-hot training rows; qualifying leaves greedy under budget."""
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    H_train = encoder.fit_transform(X_all[train])
    H_all = encoder.transform(X_all)
    z64 = z_train.astype(np.int64)
    n_pos = max(1, int(z64.sum()))
    best: Dict[int, Tuple[int, List[Group]]] = {}
    for depth in range(1, max_depth + 1):
        tree = DecisionTreeClassifier(
            criterion="entropy", max_depth=depth, min_samples_leaf=max(1, m),
            random_state=random_state,
        ).fit(H_train, z64)
        leaf_train = tree.apply(H_train)
        leaf_all = tree.apply(H_all)
        node_depth = _node_depths(tree)
        candidates: List[_Candidate] = []
        leaves = np.unique(leaf_train)
        for leaf in leaves.tolist():
            rows = np.flatnonzero(leaf_train == leaf)
            k = int(z64[rows].sum())
            if not bool(_qualifying(np.array([k]), np.array([rows.size]), tau, m)[0]):
                continue
            candidates.append(_Candidate(positives=rows[z64[rows] == 1],
                                         cost=max(1, int(node_depth[leaf])), key=leaf))
        chosen = budgeted_greedy(candidates, budgets, int(train.size))
        for budget, (idx, covered) in chosen.items():
            if budget in best and best[budget][0] >= covered:
                continue
            groups = [Group(members=leaf_all == candidates[i].key, cost=candidates[i].cost,
                            label=f"depth{depth}-leaf{candidates[i].key}") for i in idx]
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
def evaluate_groups(groups: Sequence[Group], z: np.ndarray, test: np.ndarray) -> Tuple[float, float, int]:
    """(held-out coverage, held-out purity or NaN, conditions spent)."""
    if not groups:
        return 0.0, float("nan"), 0
    inside = np.zeros(z.shape[0], dtype=bool)
    for g in groups:
        inside |= g.members
    inside_test = inside[test]
    pos_test = z[test] == 1
    n_in = int(inside_test.sum())
    k_in = int((inside_test & pos_test).sum())
    n_pos = int(pos_test.sum())
    return (k_in / n_pos if n_pos else 0.0,
            (k_in / n_in) if n_in else float("nan"),
            int(sum(g.cost for g in groups)))


METHODS: Tuple[str, ...] = ("vsf", "vsf_greedy", "rules", "rules_pure", "tree")


@dataclass(frozen=True)
class SplitOutcome:
    """One method at one budget on one split."""

    coverage: float
    purity: float  # NaN when nothing was selected or nothing held out fell inside
    conditions: int
    train_coverage: float
    labels: Tuple[str, ...]


#: split -> method -> budget -> outcome
SplitResult = Dict[str, Dict[int, SplitOutcome]]


def run_split(
    X: np.ndarray, Z: np.ndarray, positive_class: object, tau: float, m: int,
    split_index: int, n_splits: int = 5, n_repeats: int = 5, random_state: int = 0,
    budgets: Sequence[int] = BUDGETS, methods: Sequence[str] = METHODS,
    max_d: int = MAX_BRANCH_D,
) -> SplitResult:
    """
    Every method on ONE split of `vsf.centers.stratified_repeated_kfold`
    (the same folds for every method, so results are paired). `max_d`
    bounds every method alike: schema dimensionality, rule length and tree
    depth (lowered only where the exhaustive families are too large to
    run, and then for all methods). Splits are independent, so a long
    comparison can be run split by split and summarised with
    `summarize_splits`.
    """
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if not (1 <= max_d <= MAX_BRANCH_D):
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}]")
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
    results: Dict[str, MethodResult] = {}
    if "vsf" in methods:
        results["vsf"] = select_vsf(fit, X_all, train, z_train, tau, m, budgets, names, max_d=max_d)
    if "vsf_greedy" in methods:
        chain = greedy_chain(fit, z_train, spec, max_d)
        results["vsf_greedy"] = select_vsf(fit, X_all, train, z_train, tau, m, budgets, names, combos=chain)
    if "rules" in methods:
        results["rules"] = select_rules(X_all, train, z_train, tau, m, budgets, names, max_d)
    if "rules_pure" in methods:
        results["rules_pure"] = select_rules(X_all, train, z_train, tau, m, budgets, names,
                                             max_d, pure_union=True)
    if "tree" in methods:
        results["tree"] = select_tree(X_all, train, z_train, tau, m, budgets, random_state, max_d)
    out: SplitResult = {}
    for mth in methods:
        res = results[mth]
        out[mth] = {}
        for b in budgets:
            c, p, spent = evaluate_groups(res.groups[b], z, test)
            if spent > b:
                raise AssertionError(f"{mth} spent {spent} conditions at budget {b}")
            out[mth][int(b)] = SplitOutcome(
                coverage=c, purity=p, conditions=spent,
                train_coverage=res.train_coverage[b],
                labels=tuple(g.label for g in res.groups[b]),
            )
    return out


def summarize_splits(
    per_split: Sequence[SplitResult], n_splits: int, n_repeats: int,
) -> Dict[str, Dict[int, Dict[str, object]]]:
    """Nadeau-Bengio summaries per method and budget; `per_split` in split order."""
    if len(per_split) != n_splits * n_repeats:
        raise ValueError(f"expected {n_splits * n_repeats} splits, got {len(per_split)}")
    summary: Dict[str, Dict[int, Dict[str, object]]] = {}
    for mth, per_budget in per_split[0].items():
        summary[mth] = {}
        for b in per_budget:
            outs = [r[mth][b] for r in per_split]
            cov = np.array([o.coverage for o in outs])
            pur = np.array([o.purity for o in outs])
            summary[mth][b] = {
                "coverage": summarize_cv(cov, pur, n_splits, n_repeats),
                "per_split_purity": tuple(pur.tolist()),
                "cost_mean": float(np.mean([o.conditions for o in outs])),
                "train_coverage_mean": float(np.mean([o.train_coverage for o in outs])),
            }
    return summary


def run_comparison(
    X: np.ndarray, Z: np.ndarray, positive_class: object, tau: float, m: int,
    n_splits: int = 5, n_repeats: int = 5, random_state: int = 0,
    budgets: Sequence[int] = BUDGETS, methods: Sequence[str] = METHODS,
    max_d: int = MAX_BRANCH_D,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, object]:
    """All splits in one call (`run_split` then `summarize_splits`)."""
    total = n_splits * n_repeats
    per_split: List[SplitResult] = []
    for i in range(total):
        per_split.append(run_split(X, Z, positive_class, tau, m, i, n_splits, n_repeats,
                                   random_state, budgets, methods, max_d))
        if progress is not None:
            progress(i + 1, total)
    labels = {mth: [{b: list(r[mth][b].labels) for b in budgets} for r in per_split] for mth in methods}
    return {"summary": summarize_splits(per_split, n_splits, n_repeats), "labels": labels}
