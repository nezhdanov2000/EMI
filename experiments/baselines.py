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
`wracc`        the qualifying family of `rules` (training purity >= tau or
               the certificate threshold, size >= m) ranked by weighted
               relative accuracy WRAcc = (n/N)(k/n - P/N), the default
               quality of subgroup discovery (pysubgroup, SD-Map, CN2-SD),
               and read as a top-k list: rules are taken in that order while
               they fit the budget. Same candidates as `rules`, the standard
               selection order instead of budgeted maximum coverage; may
               overlap. What "run pysubgroup with a purity filter" returns.
`wracc_raw`    WRAcc top-k WITHOUT the purity filter (size >= m only): the
               list a subgroup-discovery tool returns out of the box. Not a
               solution of the task - its groups need not reach tau - and
               kept to show what the purity floor changes.
`tree_opt`     an OPTIMAL tree (DL8.5, package `pydl8.5`) of depth <= d for
               d = 1..max_d on one-hot training rows, minimising the
               class-weighted error with weights 1 : tau / (1 - tau), whose
               optimum over leaves is net coverage; leaves then as `tree`.
               The competitor CART only approximates. Needs `pydl8.5`
               (a compiled extension; Linux wheels on PyPI).
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
from typing import Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

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
    "ALL_METHODS",
    "EXTRA_METHODS",
    "METHOD_REQUIRES",
    "unavailable_methods",
    "certify_on_holdout",
    "split_halves",
    "run_comparison",
    "run_split",
    "summarize_splits",
    "select_rules",
    "select_tree",
    "select_tree_opt",
    "select_vsf",
    "select_vsf_partial",
    "select_ssdpp",
    "select_wracc",
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
#:   purity     - training purity >= tau (the product's observed rule);
#:   certified  - the family-wise certificate at alpha / T (`run_split`);
#:   split      - Webb's exploratory/holdout split: the training fold is cut
#:                at random into halves A and B, every method selects on A
#:                under the purity rule, and each selected group is kept
#:                only if its count on B passes the heterogeneous binomial
#:                threshold at alpha / T_B, T_B the number of distinct groups
#:                the method proposed (over all budgets). B's target values
#:                are independent of everything that chose the groups, so
#:                the Bonferroni bound over T_B is exact; the price is that
#:                A sees half the rows and B certifies with half the rows.
Selection = Literal["purity", "certified", "split"]


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
def _rule_candidates(
    X_train: np.ndarray, z64: np.ndarray, tau: float, m: int, max_len: int,
    threshold: Optional[np.ndarray], qualify: bool = True, top_k: Optional[int] = None,
) -> List[_Candidate]:
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
    kept: List[Tuple[Tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]] = []  # combo, codes, order, bounds
    combo_index: Dict[Tuple[int, ...], int] = {}
    picks: List[Tuple[Tuple[int, ...], int]] = []
    for d in range(1, max_len + 1):
        for combo in itertools.combinations(range(n_features), d):
            codes, n_cells = cell_codes(X_train[:, combo])
            codes = np.asarray(codes, dtype=np.int64)
            table = np.bincount(codes * 2 + z64, minlength=2 * n_cells).reshape(n_cells, 2)
            n = table.sum(axis=1)
            k = table[:, 1]
            if qualify:
                q = np.flatnonzero(_qualifying(k, n, tau, m, threshold))
            else:
                q = np.flatnonzero(n >= max(1, m))
            if q.size == 0:
                continue
            if top_k is None:
                combo_index[combo] = len(kept)
                order = np.argsort(codes, kind="stable")
                bounds = np.searchsorted(codes[order], np.arange(n_cells + 1))
                kept.append((combo, codes, order, bounds))
                picks.extend((combo, int(c)) for c in q.tolist())
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
        for combo, cells in by_combo.items():
            codes, n_cells = cell_codes(X_train[:, combo])
            codes = np.asarray(codes, dtype=np.int64)
            order = np.argsort(codes, kind="stable")
            bounds = np.searchsorted(codes[order], np.arange(n_cells + 1))
            combo_index[combo] = len(kept)
            kept.append((combo, codes, order, bounds))
            picks.extend((combo, c) for c in cells)
        picks.sort(key=lambda t: (len(t[0]), t[0], t[1]))
    candidates: List[_Candidate] = []
    seen: Dict[bytes, int] = {}
    for combo, c in picks:
        _, _, order, bounds = kept[combo_index[combo]]
        rows = order[bounds[c]:bounds[c + 1]]
        key = rows.tobytes()
        if key in seen:  # same training rows, described by fewer or equal conditions
            continue
        seen[key] = len(candidates)
        values = tuple(int(v) for v in X_train[rows[0], list(combo)])
        candidates.append(_Candidate(
            positives=rows[z64[rows] == 1], cost=len(combo), key=(combo, values),
            negatives=rows[z64[rows] == 0],
        ))
    return candidates


def _rule_groups(
    X_all: np.ndarray, candidates: Sequence[_Candidate], idx: Sequence[int], names: Sequence[str],
) -> List[Group]:
    groups: List[Group] = []
    for i in idx:
        combo, values = candidates[i].key  # type: ignore[misc]
        members = np.all(X_all[:, list(combo)] == np.asarray(values), axis=1)
        groups.append(Group(members=members, cost=candidates[i].cost, label=" & ".join(
            f"{names[j]}={v}" for j, v in zip(combo, values))))
    return groups


def select_rules(
    X_all: np.ndarray, train: np.ndarray, z_train: np.ndarray, tau: float, m: int,
    budgets: Sequence[int], names: Sequence[str], max_len: int = MAX_BRANCH_D,
    pure_union: bool = False, threshold: Optional[np.ndarray] = None, disjoint: bool = False,
) -> MethodResult:
    """Qualifying conjunctions of <= max_len (column = value) conditions, greedy under budget."""
    n_train = int(train.size)
    z64 = z_train.astype(np.int64)
    candidates = _rule_candidates(X_all[train], z64, tau, m, max_len, threshold)
    chosen = budgeted_greedy(candidates, budgets, n_train,
                             min_union_purity=tau if pure_union else None, disjoint=disjoint)
    n_pos = max(1, int(z64.sum()))
    result = MethodResult()
    for budget, (idx, covered) in chosen.items():
        result.groups[budget] = _rule_groups(X_all, candidates, idx, names)
        result.train_coverage[budget] = covered / n_pos
    return result


#: Cells kept by `select_wracc(qualify=False)`: more than any budget can hold.
WRACC_TOP_K = 256


def select_wracc(
    X_all: np.ndarray, train: np.ndarray, z_train: np.ndarray, tau: float, m: int,
    budgets: Sequence[int], names: Sequence[str], max_len: int = MAX_BRANCH_D,
    threshold: Optional[np.ndarray] = None, qualify: bool = True,
) -> MethodResult:
    """
    Conjunctions of <= max_len conditions ranked by WRAcc and read as a
    top-k list: walking down the ranking, a rule is taken when its cost
    still fits the budget (a rule that does not fit is skipped, the walk
    goes on). Ties: fewer conditions, then enumeration order. Groups may
    overlap. With `qualify` the candidates are the qualifying family of
    `rules`; without it every cell of size >= m is a candidate (the
    `WRACC_TOP_K` best by WRAcc are kept) and the purity floor plays no part
    in the selection.
    """
    z64 = z_train.astype(np.int64)
    n_train = int(train.size)
    n_pos = max(1, int(z64.sum()))
    base_rate = n_pos / n_train
    candidates = _rule_candidates(X_all[train], z64, tau, m, max_len, threshold,
                                  qualify=qualify, top_k=None if qualify else WRACC_TOP_K)
    def wracc(c: _Candidate) -> float:
        n = c.positives.size + (c.negatives.size if c.negatives is not None else 0)
        return (c.positives.size - n * base_rate) / n_train
    ranked = sorted(range(len(candidates)), key=lambda i: (-wracc(candidates[i]), candidates[i].cost, i))
    result = MethodResult()
    for budget in budgets:
        spent = 0
        idx: List[int] = []
        covered = np.zeros(n_train, dtype=bool)
        for i in ranked:
            if candidates[i].cost <= budget - spent:
                idx.append(i)
                spent += candidates[i].cost
                covered[candidates[i].positives] = True
        result.groups[int(budget)] = _rule_groups(X_all, candidates, idx, names)
        result.train_coverage[int(budget)] = int(np.count_nonzero(covered)) / n_pos
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


#: Seconds DL8.5 may spend on one fit; a fit that hits it returns its best
#: tree so far and the split records `timeout` in the group labels.
TREE_OPT_TIME_LIMIT = 600


def _class_replication(tau: float) -> Tuple[int, int]:
    """
    (copies of each positive row, copies of each negative row) that make
    unweighted misclassification equal, up to a constant, to the weighted
    error with w_pos : w_neg = 1 : tau / (1 - tau) - the Lagrangian of
    "coverage subject to purity >= tau" (net coverage). With tau = a / b in
    lowest terms the ratio is a / (b - a), so positives are copied b - a
    times and negatives a times; a leaf is then labelled positive iff its
    purity is >= tau. Denominators above 100 are rounded (tau is a user
    slider, not a measured quantity).
    """
    from fractions import Fraction
    f = Fraction(tau).limit_denominator(100)
    a, b = f.numerator, f.denominator
    if not (0 < a < b):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    return b - a, a


def _dl85_leaves(node: Dict[str, object], path: Tuple[Tuple[int, bool], ...] = ()) -> Iterator[Tuple[Tuple[Tuple[int, bool], ...], int]]:
    """(path of (feature, present), predicted label) for every leaf of a DL8.5 tree dict.
    DL8.5 puts the rows where the feature is 1 in `left`."""
    if "value" in node:
        yield path, int(node["value"])  # type: ignore[call-overload]
        return
    feat = int(node["feat"])  # type: ignore[call-overload]
    yield from _dl85_leaves(node["left"], path + ((feat, True),))  # type: ignore[arg-type]
    yield from _dl85_leaves(node["right"], path + ((feat, False),))  # type: ignore[arg-type]


def select_tree_opt(
    X_all: np.ndarray, train: np.ndarray, z_train: np.ndarray, tau: float, m: int,
    budgets: Sequence[int], max_depth: int = MAX_BRANCH_D, threshold: Optional[np.ndarray] = None,
    time_limit: int = TREE_OPT_TIME_LIMIT,
) -> MethodResult:
    """
    Optimal decision tree (DL8.5, Aglin, Nijssen & Schaus 2020; package
    `pydl8.5`) of depth <= d for each d = 1..max_depth on one-hot training
    rows, minimising the class-weighted misclassification with weights
    1 : tau / (1 - tau) - exactly the objective whose optimum over leaves
    is net coverage - via row replication (`_class_replication`), so no
    Python callback slows the search. The leaves are then treated exactly
    like CART's in `select_tree`: qualifying leaves (purity rule or
    `threshold`, on the ORIGINAL training rows) enter the budgeted greedy
    at cost = depth, and per budget the depth with the most covered
    training positives is kept. Like the CART filter, the certificate on
    these leaves is only a filter: they are grown on the target.

    A fit that reaches `time_limit` seconds returns DL8.5's best tree so
    far; its groups carry `timeout` in the label so that the summary can
    count them. Depth 1 and 2 use DL8.5's specialised algorithm and are
    exact within seconds on every table of the study.
    """
    from pydl85 import DL85Classifier
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.int8)
    H_train = np.asarray(encoder.fit_transform(X_all[train]).todense(), dtype=np.int8)
    H_all = np.asarray(encoder.transform(X_all).todense(), dtype=np.int8)
    z64 = z_train.astype(np.int64)
    n_pos = max(1, int(z64.sum()))
    r_pos, r_neg = _class_replication(tau)
    pos_rows, neg_rows = np.flatnonzero(z64 == 1), np.flatnonzero(z64 == 0)
    H_rep = np.concatenate([np.repeat(H_train[pos_rows], r_pos, axis=0), np.repeat(H_train[neg_rows], r_neg, axis=0)])
    z_rep = np.concatenate([np.ones(pos_rows.size * r_pos, dtype=np.int64), np.zeros(neg_rows.size * r_neg, dtype=np.int64)])
    best: Dict[int, Tuple[int, List[Group]]] = {}
    for depth in range(1, min(max_depth, H_train.shape[1]) + 1):
        clf = DL85Classifier(max_depth=depth, min_sup=max(1, m * min(r_pos, r_neg)), time_limit=time_limit)
        clf.fit(H_rep, z_rep)
        timed_out = bool(getattr(clf, "timeout_", False))
        candidates: List[_Candidate] = []
        members_all: List[np.ndarray] = []
        for path, label in _dl85_leaves(clf.tree_):
            if label != 1 or not path:
                continue
            in_train = np.ones(train.size, dtype=bool)
            in_all = np.ones(X_all.shape[0], dtype=bool)
            for feat, present in path:
                in_train &= (H_train[:, feat] == 1) == present
                in_all &= (H_all[:, feat] == 1) == present
            rows = np.flatnonzero(in_train)
            if rows.size == 0:
                continue
            k = int(z64[rows].sum())
            if not bool(_qualifying(np.array([k]), np.array([rows.size]), tau, m, threshold)[0]):
                continue
            candidates.append(_Candidate(positives=rows[z64[rows] == 1], cost=len(path), key=len(members_all)))
            members_all.append(in_all)
        chosen = budgeted_greedy(candidates, budgets, int(train.size))
        for budget, (idx, covered) in chosen.items():
            if budget in best and best[budget][0] >= covered:
                continue
            groups = [Group(members=members_all[int(candidates[i].key)], cost=candidates[i].cost,  # type: ignore[arg-type]
                            label=f"dl85-depth{depth}-leaf{candidates[i].key}" + ("-timeout" if timed_out else ""))
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
#: Methods added after the main run. They are NOT part of the cache key of
#: `compare_baselines.py` (which hashes METHODS), so they are computed on
#: demand and merged into the existing per-split files.
EXTRA_METHODS: Tuple[str, ...] = ("wracc", "wracc_raw", "tree_opt")
ALL_METHODS: Tuple[str, ...] = METHODS + EXTRA_METHODS
#: Optional packages a method needs (checked by `unavailable_methods`).
METHOD_REQUIRES: Dict[str, str] = {"ssdpp": "rulelist", "tree_opt": "pydl85", "tree": "sklearn"}


def unavailable_methods(methods: Sequence[str] = ALL_METHODS) -> Dict[str, str]:
    """{method: missing package} for the methods whose package does not import here."""
    import importlib.util
    out: Dict[str, str] = {}
    for mth in methods:
        pkg = METHOD_REQUIRES.get(mth)
        if pkg is not None and importlib.util.find_spec(pkg) is None:
            out[mth] = pkg
    return out


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


def split_halves(train: np.ndarray, random_state: int, split_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    (A, B): the training fold cut uniformly at random into a proposing half
    A and a certifying half B. NOT stratified on the target - a stratified
    cut would fix B's positive count from the target values and break the
    conditional-independence argument of `certify_on_holdout`. Seeded by
    (random_state, split_index) so that the cut is reproducible and differs
    between splits.
    """
    rng = np.random.default_rng([int(random_state), int(split_index), 0x5B17])
    perm = rng.permutation(train.size)
    return np.sort(train[perm[:train.size // 2]]), np.sort(train[perm[train.size // 2:]])


def certify_on_holdout(
    result: MethodResult, z: np.ndarray, holdout: np.ndarray, tau: float, alpha: float,
) -> Tuple[MethodResult, int]:
    """
    Webb's holdout step for one method: every group the method proposed (at
    any budget) is tested once on the `holdout` rows - kept iff its positive
    count there reaches `min_successes_to_certify_heterogeneous`(n, tau,
    alpha / T_B), with T_B the number of DISTINCT proposed groups (by their
    holdout row set) - and each budget keeps its passing groups in order.
    Returns the filtered result and T_B. Training coverage is recomputed as
    the share of holdout positives the kept union holds. A group with no
    holdout row cannot pass.
    """
    z_hold = z[holdout].astype(np.int64)
    n_pos = max(1, int(z_hold.sum()))
    distinct: Dict[bytes, Tuple[int, int]] = {}
    for groups in result.groups.values():
        for g in groups:
            members = g.members[holdout]
            key = np.packbits(members).tobytes()
            if key not in distinct:
                distinct[key] = (int(members.sum()), int(z_hold[members].sum()))
    t_b = len(distinct)
    if t_b == 0:
        return result, 0
    level = alpha / t_b
    sizes = np.array([n for n, _ in distinct.values()], dtype=np.int64)
    need = np.asarray(min_successes_to_certify_heterogeneous(np.maximum(sizes, 1), tau, level), dtype=np.int64)
    passes = {key: (n > 0 and k >= int(need[i])) for i, (key, (n, k)) in enumerate(distinct.items())}
    out = MethodResult()
    for budget, groups in result.groups.items():
        kept = [g for g in groups if passes[np.packbits(g.members[holdout]).tobytes()]]
        out.groups[budget] = kept
        inside = union_members(kept, z.shape[0])[holdout]
        out.train_coverage[budget] = int(z_hold[inside].sum()) / n_pos
    return out, t_b


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

    `selection="split"`: the training fold is halved at random; every
    method selects on half A under the purity rule and `certify_on_holdout`
    keeps, per budget, the proposed groups whose count on half B passes the
    threshold at alpha / T_B. Unlike the family certificate this is valid
    for every method alike (the tree and SSD++ included), because B is
    read only after the groups are fixed; its cost is the halved sample.
    The reported conditions are those of the kept groups.
    """
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if not (1 <= max_d <= MAX_BRANCH_D):
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}]")
    if selection not in ("purity", "certified", "split"):
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
    holdout: Optional[np.ndarray] = None
    if selection == "split":
        train, holdout = split_halves(train, random_state, split_index)
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
        "wracc": lambda: select_wracc(X_all, train, z_train, tau, m, budgets, names, max_d,
                                      threshold=threshold, qualify=True),
        "wracc_raw": lambda: select_wracc(X_all, train, z_train, tau, m, budgets, names, max_d,
                                          threshold=None, qualify=False),
        "tree_opt": lambda: select_tree_opt(X_all, train, z_train, tau, m, budgets, max_d,
                                            threshold=threshold),
    }
    out: SplitResult = {}
    for mth in methods:
        t0 = time.perf_counter()
        res = selectors[mth]()
        if holdout is not None:
            res, _ = certify_on_holdout(res, z, holdout, tau, alpha)
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
