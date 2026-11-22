"""
How far the greedy packings of the paper are from the exact optimum.

Two of the paper's methods select groups by a heuristic:

* `vsf_partial` - per schema, the candidates are the qualifying cells of its
  sub-schemas; a budgeted disjoint maximum-coverage packing picks a set of
  them (`baselines.budgeted_greedy(disjoint=True)`, the cost-effectiveness
  greedy of Khuller, Moss & Naor 1999 against the best single candidate),
  and the schema with the largest packing wins the budget.
* `rules_disjoint` - the same greedy over every qualifying conjunction of
  <= 4 conditions.

This script solves the same two problems EXACTLY on the same training
folds, as integer programs (scipy's HiGHS), and reports the gap. For the
partial centres the exact optimum is taken over every schema (the schema
whose exact packing is largest), so the gap includes the schema choice.

    maximise   sum_i k_i x_i
    s.t.       sum_i cost_i x_i <= B
               sum_{i : r in rows_i} x_i <= 1      for every training row r
               x_i in {0, 1}

where k_i is the number of training positives of candidate i. Schemas are
pruned by an upper bound (the positives in the union of a schema's
candidates): a schema is solved only if that bound beats the incumbent.

The tables are the small ones of the study, where the exact problems are
tractable; the folds are the first `--splits` of the same repeated
stratified CV the comparison uses (seed 0), so the greedy numbers coincide
with the comparison's training coverage on those splits.

Usage:
    python experiments/packing_exact.py                       # default tables
    python experiments/packing_exact.py --only titanic --splits 5
Output:
    experiments/results/packing_exact.csv   one row per (configuration,
                                            selection, split, budget)
    and a summary on stdout (mean and maximum gap in coverage points).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from baselines import (BUDGETS, _Candidate, _rule_candidates, _schema_table,  # noqa: E402
                       budgeted_greedy, certification_threshold)
from compare_baselines import CONFIGS, Config  # noqa: E402
from datasets import load_dataset  # noqa: E402
from vsf.avr import _CandidateFactory, _prepare_search, resolve_center_spec  # noqa: E402
from vsf.centers import CenterSpec, stratified_repeated_kfold  # noqa: E402

#: Tables small enough for the exact problems (every schema, every budget).
DEFAULT_TABLES: Tuple[str, ...] = (
    "titanic", "breast_cancer", "thyroid_recurrence", "breast_cancer_wisconsin",
    "car_evaluation", "nursery", "hmda",
)


@dataclass(frozen=True)
class ExactResult:
    covered: int
    status: str          # "optimal" | "time_limit" | "infeasible"
    seconds: float


def solve_packing(cands: Sequence[_Candidate], budget: int, n_train: int,
                  time_limit: float) -> ExactResult:
    """Exact budgeted disjoint maximum coverage over `cands` (see module docstring)."""
    t0 = time.perf_counter()
    m = len(cands)
    if m == 0:
        return ExactResult(0, "optimal", 0.0)
    k = np.array([c.positives.size for c in cands], dtype=np.float64)
    cost = np.array([c.cost for c in cands], dtype=np.float64)
    # row-incidence matrix (rows x candidates) restricted to rows in >= 2 candidates
    rows_i: List[np.ndarray] = []
    cols_i: List[np.ndarray] = []
    for i, c in enumerate(cands):
        r = np.concatenate([c.positives, c.negatives]) if c.negatives is not None else c.positives
        rows_i.append(r)
        cols_i.append(np.full(r.size, i, dtype=np.int64))
    r_all = np.concatenate(rows_i)
    c_all = np.concatenate(cols_i)
    A = sparse.csr_matrix((np.ones(r_all.size), (r_all, c_all)), shape=(n_train, m))
    shared = np.flatnonzero(np.asarray(A.sum(axis=1)).ravel() >= 2)
    constraints = [LinearConstraint(cost.reshape(1, -1), -np.inf, float(budget))]
    if shared.size:
        constraints.append(LinearConstraint(A[shared], -np.inf, np.ones(shared.size)))
    res = milp(c=-k, constraints=constraints, integrality=np.ones(m),
               bounds=Bounds(np.zeros(m), np.ones(m)),
               options={"time_limit": time_limit, "disp": False})
    seconds = time.perf_counter() - t0
    if res.x is None:
        return ExactResult(0, "infeasible" if res.status == 2 else "time_limit", seconds)
    x = np.rint(res.x).astype(bool)
    covered = int(k[x].sum())
    status = "optimal" if res.status == 0 else "time_limit"
    return ExactResult(covered, status, seconds)


def partial_pools(fit: _CandidateFactory, z64: np.ndarray, tau: float, m: int, max_d: int,
                  threshold: Optional[np.ndarray]) -> List[Tuple[Tuple[int, ...], List[_Candidate]]]:
    """
    Every schema's candidate pool, exactly as `baselines.select_vsf_partial`
    builds it (qualifying cells of all sub-schemas, one candidate per
    distinct row set, the cheapest description), deduplicated by pool.
    """
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
            lst.append(_Candidate(positives=rows[z64[rows] == 1], cost=len(combo),
                                  key=(combo, int(c)), negatives=rows[z64[rows] == 0]))
        per_combo[combo] = lst
    pools: List[Tuple[Tuple[int, ...], List[_Candidate]]] = []
    seen_pools: set = set()
    for d in range(1, max_d + 1):
        for combo in itertools.combinations(range(fit.n_features), d):
            live = tuple(sub for r in range(1, d + 1) for sub in itertools.combinations(combo, r)
                         if per_combo.get(sub))
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
            pools.append((combo, pool))
    return pools


def dedupe_rule_candidates(cands: Sequence[_Candidate]) -> List[_Candidate]:
    """One candidate per distinct training row set, the cheapest description."""
    out: List[_Candidate] = []
    seen: Dict[bytes, int] = {}
    for c in cands:
        rows = np.sort(np.concatenate([c.positives, c.negatives]))  # type: ignore[list-item]
        key = rows.tobytes()
        j = seen.get(key)
        if j is None:
            seen[key] = len(out)
            out.append(c)
        elif c.cost < out[j].cost:
            out[j] = c
    return out


def run_config(cfg: Config, X: np.ndarray, Z: np.ndarray, n_splits_used: int,
               time_limit: float, budgets: Sequence[int]) -> List[Dict[str, object]]:
    spec = CenterSpec(tau=cfg.tau, min_samples=max(1, cfg.min_samples))
    prepared = _prepare_search(X, Z, None, cfg.positive, spec, "presence")
    assert prepared is not None
    factory, z, _ = prepared
    X_all = factory._raw.astype(np.int64, copy=False)
    splits = stratified_repeated_kfold(z, 5, cfg.n_repeats, 0)[:n_splits_used]
    rows: List[Dict[str, object]] = []
    for split_index, (train, _test) in enumerate(splits):
        z_train = z[train]
        z64 = z_train.astype(np.int64)
        n_train = int(train.size)
        n_pos = max(1, int(z64.sum()))
        fit = _CandidateFactory(X_all[train], factory.bin_counts, n_train, ordered=factory.ordered)
        threshold: Optional[np.ndarray] = None
        if cfg.selection == "certified":
            cs = resolve_center_spec(
                fit, CenterSpec(tau=cfg.tau, min_samples=max(1, cfg.min_samples), rule="certified",
                                multiplicity="family", alpha=0.05), min(cfg.max_d, fit.n_features))
            threshold = certification_threshold(n_train, cfg.tau, cs.effective_alpha(0))
        max_d = min(cfg.max_d, fit.n_features)

        # --- partial centres: greedy per schema (as the comparison) and exact per schema
        pools = partial_pools(fit, z64, cfg.tau, cfg.min_samples, max_d, threshold)
        greedy_best: Dict[int, int] = {b: 0 for b in budgets}
        for _combo, pool in pools:
            for b, (idx, covered) in budgeted_greedy(pool, budgets, n_train, disjoint=True).items():
                if covered > greedy_best[b]:
                    greedy_best[b] = covered
        # upper bound per pool: positives in the union of its candidates
        ub: List[int] = []
        for _combo, pool in pools:
            u = np.zeros(n_train, dtype=bool)
            for c in pool:
                u[c.positives] = True
            ub.append(int(u.sum()))
        order = np.argsort(-np.array(ub))
        exact_best: Dict[int, int] = dict(greedy_best)   # the greedy is feasible, so a lower bound
        exact_status: Dict[int, str] = {b: "optimal" for b in budgets}
        exact_seconds: Dict[int, float] = {b: 0.0 for b in budgets}
        solved: Dict[int, int] = {b: 0 for b in budgets}
        for b in budgets:
            for j in order:
                if ub[j] <= exact_best[b]:
                    break
                r = solve_packing(pools[j][1], b, n_train, time_limit)
                solved[b] += 1
                exact_seconds[b] += r.seconds
                if r.status != "optimal":
                    exact_status[b] = r.status
                if r.covered > exact_best[b]:
                    exact_best[b] = r.covered

        # --- free disjoint rules: greedy (as the comparison) and one exact ILP per budget
        rule_cands = dedupe_rule_candidates(
            _rule_candidates(X_all[train], z64, cfg.tau, cfg.min_samples, max_d, threshold))
        rule_greedy = budgeted_greedy(rule_cands, budgets, n_train, disjoint=True)
        for b in budgets:
            r = solve_packing(rule_cands, b, n_train, time_limit)
            rows.append({
                "dataset": cfg.dataset, "positive": cfg.positive, "tau": cfg.tau, "m": cfg.min_samples,
                "selection": cfg.selection, "split": split_index, "budget": b,
                "n_train": n_train, "n_pos": n_pos, "schemas": len(pools), "ilps_solved_partial": solved[b],
                "partial_greedy": round(greedy_best[b] / n_pos, 4),
                "partial_exact": round(exact_best[b] / n_pos, 4),
                "partial_exact_status": exact_status[b],
                "partial_exact_seconds": round(exact_seconds[b], 2),
                "rules_candidates": len(rule_cands),
                "rules_disj_greedy": round(rule_greedy[b][1] / n_pos, 4),
                "rules_disj_exact": round(max(r.covered, rule_greedy[b][1]) / n_pos, 4),
                "rules_disj_exact_status": r.status,
                "rules_disj_exact_seconds": round(r.seconds, 2),
            })
            print(f"{cfg.dataset:24s} {cfg.selection:9s} split {split_index} B={b:2d} "
                  f"partial {greedy_best[b] / n_pos:.3f} -> {exact_best[b] / n_pos:.3f} "
                  f"({exact_status[b]}, {solved[b]} ILPs) | rules_disj {rule_greedy[b][1] / n_pos:.3f} "
                  f"-> {max(r.covered, rule_greedy[b][1]) / n_pos:.3f} ({r.status})", flush=True)
    return rows


def summarise(rows: Sequence[Dict[str, object]]) -> str:
    lines = ["", "gap of the greedy to the exact optimum, in coverage points (training rows):"]
    for sel in ("purity", "certified"):
        sub = [r for r in rows if r["selection"] == sel]
        if not sub:
            continue
        gp = np.array([100 * (float(r["partial_exact"]) - float(r["partial_greedy"])) for r in sub])
        gr = np.array([100 * (float(r["rules_disj_exact"]) - float(r["rules_disj_greedy"])) for r in sub])
        unsolved = sum(1 for r in sub if r["partial_exact_status"] != "optimal"
                       or r["rules_disj_exact_status"] != "optimal")
        lines.append(f"  {sel:9s} cells {len(sub):4d}  partial centres: mean {gp.mean():.2f}, "
                     f"max {gp.max():.2f}, >1 point in {int((gp > 1).sum())}  |  disjoint rules: "
                     f"mean {gr.mean():.2f}, max {gr.max():.2f}, >1 point in {int((gr > 1).sum())}"
                     f"  |  cells not proven optimal: {unsolved}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="one table (default: the small tables of the study)")
    ap.add_argument("--splits", type=int, default=5, help="first splits of the CV to use")
    ap.add_argument("--time-limit", type=float, default=120.0, help="seconds per integer program")
    ap.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "packing_exact.csv")
    args = ap.parse_args()

    tables = (args.only,) if args.only else DEFAULT_TABLES
    configs = [c for c in CONFIGS if c.dataset in tables and c.selection in ("purity", "certified")]
    if not configs:
        raise SystemExit(f"no configuration for {tables}")
    rows: List[Dict[str, object]] = []
    for cfg in configs:
        frame, entry = load_dataset(cfg.dataset)
        feats = [c for c in frame.columns if c != entry.target]
        rows.extend(run_config(cfg, frame[feats].values, frame[entry.target].values,
                               args.splits, args.time_limit, args.budgets))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print(summarise(rows))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
