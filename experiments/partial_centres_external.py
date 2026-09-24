"""Price of alignment: full-grid cells vs partial-axis centres vs free rules.

Self-contained (no dependency on vsf/): a from-scratch re-implementation
on datasets that are NOT in benchmark_data/, run before `vsf.partial` and
the `vsf_partial` / `rules_disjoint` baselines existed, as an independent
check of the mechanism (PMD Section 4.16). The in-repository comparison on
the manifest datasets is `compare_baselines.py`.

Data (put the files in --data; none is redistributed here):
  titanic_sns.csv  https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
  adult.csv        https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv
  HMDA.csv, attrition.csv, credit_data.csv, ResumeNames.csv, wa_churn.csv, mlc_churn.csv
                   https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/<pkg>/<name>.csv
                   (AER/HMDA, modeldata/attrition, modeldata/credit_data, AER/ResumeNames,
                    modeldata/wa_churn, modeldata/mlc_churn)
The two synthetic datasets need no files. Three candidate families for
"max target coverage subject to purity >= tau, n >= m, at most B conditions":

  grid     -- one schema S, |S| <= d_max; centres = cells of the FULL grid on S
              (every axis fixed). Cost of a cell = |S|. Cells are disjoint.
  partial  -- one schema S; centres = cells of ANY sub-grid S' subset of S
              (some axes left free). Cost = |S'|. Two selection variants:
              partial_disjoint (chosen centres must not overlap) and
              partial_overlap (union, greedy max coverage).
  rules    -- cells of ANY schema with |S| <= d_max (= class association rules
              of length <= d_max). Cost = |S|. rules_disjoint / rules_overlap.

Selection on the training fold only; coverage and purity of the selected
region measured on the held-out fold. Paired differences use the
Nadeau-Bengio corrected variance.

Simplifications versus vsf/: no capacity rule, no level merging, numeric
columns quantile-binned once on all rows (identical for every method).
"""
from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom

Mask = np.ndarray  # packed uint8 bit-mask over rows

_POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def popcount(packed: np.ndarray) -> np.ndarray:
    """Row-wise popcount of a (C, nbytes) uint8 matrix -> (C,) int32."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(packed).sum(axis=1, dtype=np.int32)
    return _POPCNT[packed].sum(axis=1, dtype=np.int32)


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

def qbin(s: pd.Series, q: int) -> pd.Series:
    """Quantile bins with duplicate edges collapsed; NaN -> its own level."""
    out = pd.qcut(s, q, duplicates="drop", labels=False)
    return out.astype("float").fillna(-1).astype(int)


def _finish(df: pd.DataFrame, y: pd.Series, name: str) -> Tuple[str, np.ndarray, np.ndarray, List[str]]:
    x = np.empty((len(df), df.shape[1]), dtype=np.int32)
    for j, c in enumerate(df.columns):
        codes, _ = pd.factorize(df[c].fillna("NA").astype(str), sort=True)
        x[:, j] = codes
    return name, x, y.to_numpy().astype(np.int8), list(df.columns)


def load_titanic(d: Path):
    t = pd.read_csv(d / "titanic_sns.csv")
    f = pd.DataFrame({
        "pclass": t.pclass, "sex": t.sex, "embarked": t.embarked.fillna("NA"),
        "age": qbin(t.age, 5), "sibsp": t.sibsp.clip(upper=2), "parch": t.parch.clip(upper=2),
        "fare": qbin(t.fare, 4), "alone": t.alone.astype(int)})
    return _finish(f, t.survived, "titanic")


def load_adult(d: Path):
    cols = ["age", "workclass", "fnlwgt", "education", "educnum", "marital", "occupation",
            "relationship", "race", "sex", "capgain", "caploss", "hours", "native", "cls"]
    t = pd.read_csv(d / "adult.csv", header=None, names=cols, skipinitialspace=True)
    f = pd.DataFrame({
        "age": qbin(t.age, 5), "workclass": t.workclass, "education": t.education,
        "marital": t.marital, "occupation": t.occupation, "relationship": t.relationship,
        "race": t.race, "sex": t.sex, "capgain": (t.capgain > 0).astype(int),
        "caploss": (t.caploss > 0).astype(int), "hours": qbin(t.hours, 4),
        "native": (t.native == "United-States").astype(int)})
    return _finish(f, t.cls.str.contains(">50K"), "adult")


def load_hmda(d: Path):
    t = pd.read_csv(d / "HMDA.csv")
    f = t[["chist", "mhist", "phist", "selfemp", "insurance", "condomin", "afam", "single", "hschool"]].copy()
    f["pirat"] = qbin(t.pirat, 4); f["lvrat"] = qbin(t.lvrat, 4); f["unemp"] = qbin(t.unemp, 3)
    return _finish(f, t.deny == "yes", "hmda")


def load_attrition(d: Path):
    t = pd.read_csv(d / "attrition.csv")
    keep = ["BusinessTravel", "Department", "Education", "EducationField", "EnvironmentSatisfaction",
            "Gender", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
            "OverTime", "StockOptionLevel", "WorkLifeBalance"]
    f = t[keep].copy()
    for c, q in [("Age", 4), ("MonthlyIncome", 4), ("TotalWorkingYears", 4), ("YearsAtCompany", 4),
                 ("DistanceFromHome", 3)]:
        f[c] = qbin(t[c], q)
    return _finish(f, t.Attrition == "Yes", "attrition")


def load_credit(d: Path):
    t = pd.read_csv(d / "credit_data.csv")
    f = t[["Home", "Marital", "Records", "Job"]].copy()
    for c, q in [("Seniority", 4), ("Time", 3), ("Age", 4), ("Expenses", 3), ("Income", 4),
                 ("Assets", 3), ("Amount", 4), ("Price", 4)]:
        f[c] = qbin(t[c], q)
    f["Debt"] = (t.Debt.fillna(0) > 0).astype(int)
    return _finish(f, t.Status == "bad", "credit")


def load_resume(d: Path):
    t = pd.read_csv(d / "ResumeNames.csv")
    keep = ["gender", "ethnicity", "quality", "city", "jobs", "honors", "volunteer", "military",
            "holes", "school", "email", "computer", "special", "college", "equal", "wanted",
            "requirements", "reqexp", "reqcomm", "reqeduc", "reqcomp", "reqorg", "industry"]
    f = t[keep].copy(); f["experience"] = qbin(t.experience, 3)
    return _finish(f, t.call == "yes", "resume")


def load_wa_churn(d: Path):
    t = pd.read_csv(d / "wa_churn.csv")
    drop = {"rownames", "churn", "tenure", "monthly_charges", "total_charges"}
    f = t[[c for c in t.columns if c not in drop]].copy()
    f["tenure"] = qbin(t.tenure, 4); f["monthly"] = qbin(t.monthly_charges, 4)
    f["total"] = qbin(t.total_charges, 4)
    return _finish(f, t.churn == "Yes", "wa_churn")


def load_mlc_churn(d: Path):
    t = pd.read_csv(d / "mlc_churn.csv")
    f = t[["area_code", "international_plan", "voice_mail_plan"]].copy()
    f["vmail"] = (t.number_vmail_messages > 0).astype(int)
    for c, q in [("total_day_minutes", 4), ("total_eve_minutes", 4), ("total_night_minutes", 4),
                 ("total_intl_minutes", 4), ("total_intl_calls", 3), ("account_length", 3)]:
        f[c] = qbin(t[c], q)
    f["svc_calls"] = t.number_customer_service_calls.clip(upper=4)
    return _finish(f, t.churn == "yes", "mlc_churn")


def load_synth_misaligned(d: Path):
    """Truth = two conjunctions on DISJOINT column pairs; does not fit one full grid."""
    rng = np.random.default_rng(42)
    N, M = 6000, 10
    x = rng.integers(0, 3, size=(N, M))
    r1 = (x[:, 0] == 0) & (x[:, 1] == 1)
    r2 = (x[:, 2] == 2) & (x[:, 3] == 0)
    p = np.where(r1 | r2, 0.95, 0.10)
    y = (rng.random(N) < p).astype(np.int8)
    return "synth_misaligned", x.astype(np.int32), y, [f"c{i}" for i in range(M)]


def load_synth_aligned(d: Path):
    """Truth = one full 4D cell plus a 2D cell inside the same 4 columns; fits one grid."""
    rng = np.random.default_rng(43)
    N, M = 6000, 10
    x = rng.integers(0, 3, size=(N, M))
    r1 = (x[:, 0] == 0) & (x[:, 1] == 1) & (x[:, 2] == 2) & (x[:, 3] == 0)
    r2 = (x[:, 0] == 2) & (x[:, 1] == 2)
    p = np.where(r1 | r2, 0.95, 0.10)
    y = (rng.random(N) < p).astype(np.int8)
    return "synth_aligned", x.astype(np.int32), y, [f"c{i}" for i in range(M)]


LOADERS: Dict[str, Callable[[Path], tuple]] = {
    "synth_misaligned": load_synth_misaligned, "synth_aligned": load_synth_aligned,
    "titanic": load_titanic, "adult": load_adult, "hmda": load_hmda, "attrition": load_attrition,
    "credit": load_credit, "resume": load_resume, "wa_churn": load_wa_churn, "mlc_churn": load_mlc_churn,
}


# --------------------------------------------------------------------------- #
# Candidate generation                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    schema: Tuple[int, ...]
    code: int
    n: int
    k: int
    mask: Mask        # packed bits over TRAIN rows
    key: int          # hash of the row set (dedupe)


def cell_codes(x: np.ndarray, schema: Sequence[int], card: np.ndarray) -> np.ndarray:
    code = np.zeros(x.shape[0], dtype=np.int64)
    stride = 1
    for j in schema:
        code += x[:, j].astype(np.int64) * stride
        stride *= int(card[j])
    return code


def k_star(n: np.ndarray, tau: float, level: float) -> np.ndarray:
    """Smallest k with P(Bin(n,tau) >= k) <= level and k >= floor(n*tau)+1 (Hoeffding condition)."""
    k = binom.isf(level, n, tau).astype(np.int64)     # largest k with sf(k-1) > level ... adjust below
    # binom.isf returns k such that sf(k) <= level; we need P(X >= k) = sf(k-1) <= level
    k = k + 1
    k = np.maximum(k, np.floor(n * tau).astype(np.int64) + 1)
    return k


def occupied_cell_count(x: np.ndarray, card: np.ndarray, schemas: Sequence[Tuple[int, ...]],
                        m: int, hash_lo: np.ndarray, hash_hi: np.ndarray) -> int:
    """Number of DISTINCT row sets among occupied cells (n >= 1) of all schemas: the family size T.
    Two 26-bit additive fingerprints per cell (exact in float64) plus the cell size."""
    seen: set = set()
    for s in schemas:
        code = cell_codes(x, s, card)
        size = int(np.prod(card[list(s)]))
        n = np.bincount(code, minlength=size)
        lo = np.bincount(code, weights=hash_lo, minlength=size)
        hi = np.bincount(code, weights=hash_hi, minlength=size)
        occ = np.flatnonzero(n >= 1)
        seen.update(zip(n[occ].tolist(), lo[occ].tolist(), hi[occ].tolist()))
    return len(seen)


def qualifying_cells(x: np.ndarray, y: np.ndarray, card: np.ndarray, schema: Tuple[int, ...],
                     tau: float, m: int, row_hash: np.ndarray, level: float | None = None) -> List[Candidate]:
    """Cells with n >= m and either k/n >= tau (level=None) or k >= k*(n, tau, level) (certified)."""
    code = cell_codes(x, schema, card)
    size = int(np.prod(card[list(schema)]))
    n = np.bincount(code, minlength=size)
    k = np.bincount(code, weights=y, minlength=size).astype(np.int64)
    if level is None:
        ok = np.flatnonzero((n >= m) & (k >= tau * n))
    else:
        big = n >= m
        thr = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
        if big.any():
            thr[big] = k_star(n[big], tau, level)
        ok = np.flatnonzero(big & (k >= thr))
    out: List[Candidate] = []
    for c in ok:
        rows = code == c
        out.append(Candidate(schema, int(c), int(n[c]), int(k[c]), np.packbits(rows),
                             int(row_hash[rows].sum() & ((1 << 62) - 1))))
    return out


# --------------------------------------------------------------------------- #
# Selection under a condition budget                                           #
# --------------------------------------------------------------------------- #

def greedy_select(cands: List[Candidate], pos_packed: np.ndarray, budget: int,
                  mode: str, tau: float = 0.0) -> Tuple[List[int], int]:
    """Greedy at one budget: best of the gain and gain/cost passes."""
    best: Tuple[List[int], int] = ([], 0)
    for traj in greedy_trajectories(cands, pos_packed, budget, mode, tau):
        if traj and traj[-1][2] > best[1]:
            best = ([t[0] for t in traj], traj[-1][2])
    return best


def prefix_at_budget(traj: List[Tuple[int, int, int]], budget: int) -> Tuple[List[int], int]:
    """Longest prefix of a trajectory whose cumulative cost fits the budget."""
    idx: List[int] = []
    cov = 0
    for i, cum_cost, cum_cov in traj:
        if cum_cost > budget:
            break
        idx.append(i); cov = cum_cov
    return idx, cov


def greedy_trajectories(cands: List[Candidate], pos_packed: np.ndarray, budget: int,
                        mode: str, tau: float = 0.0) -> List[List[Tuple[int, int, int]]]:
    """Two greedy passes (gain, gain/cost). Each trajectory is a list of
    (candidate index, cumulative cost, cumulative positives covered).

    mode: "disjoint" -- chosen centres must not overlap (union purity >= tau on train);
          "overlap"  -- plain budgeted max coverage, union purity unconstrained;
          "union"    -- overlap allowed, union must keep purity >= tau on train.
    """
    disjoint = mode == "disjoint"
    if not cands:
        return []
    masks = np.stack([c.mask for c in cands])
    cost = np.array([len(c.schema) for c in cands])
    pos_masks = masks & pos_packed
    out: List[List[Tuple[int, int, int]]] = []
    for by_density in (False, True):
        chosen: List[Tuple[int, int, int]] = []
        covered = np.zeros_like(pos_packed)          # covered positives
        occupied = np.zeros_like(pos_packed)         # covered rows (for disjoint)
        remaining = budget
        total = 0
        n_occ = 0
        alive = cost <= remaining
        while True:
            alive &= cost <= remaining
            if disjoint and occupied.any():
                alive &= ~((masks & occupied).any(axis=1))
            if not alive.any():
                break
            gain = popcount(pos_masks & ~covered)
            if mode == "union":
                new_rows = popcount(masks & ~occupied)
                alive &= (total + gain) >= tau * (n_occ + new_rows)
                if not alive.any():
                    break
            gain[~alive] = -1
            score = gain / cost if by_density else gain.astype(float)
            score[~alive] = -1
            i = int(np.argmax(score))
            if gain[i] <= 0:
                break
            covered |= pos_masks[i]
            occupied |= masks[i]
            n_occ = int(popcount(occupied[None, :])[0])
            total += int(gain[i])
            remaining -= int(cost[i])
            chosen.append((i, budget - remaining, total))
            alive[i] = False
        out.append(chosen)
    return out


def union_rows_test(cands: List[Candidate], idx: List[int], x_test: np.ndarray,
                    card: np.ndarray) -> np.ndarray:
    sel = np.zeros(x_test.shape[0], dtype=bool)
    for i in idx:
        c = cands[i]
        sel |= cell_codes(x_test, c.schema, card) == c.code
    return sel


# --------------------------------------------------------------------------- #
# Experiment                                                                   #
# --------------------------------------------------------------------------- #

def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for f, chunk in enumerate(np.array_split(idx, n_splits)):
            folds[f].extend(chunk.tolist())
    return [np.array(sorted(f)) for f in folds]


def run_dataset(name: str, x: np.ndarray, y: np.ndarray, taus: Sequence[float], m: int,
                budgets: Sequence[int], d_max: int, repeats: int, n_splits: int,
                seed: int, selection: str = "purity", alpha: float = 0.05) -> List[dict]:
    card = x.max(axis=0) + 1
    M = x.shape[1]
    schemas = [s for d in range(1, d_max + 1) for s in itertools.combinations(range(M), d)]
    rows: List[dict] = []
    base = float(y.mean())
    for tau in taus:
        if tau <= base:
            continue
        for r in range(repeats):
            folds = stratified_folds(y, n_splits, seed + r)
            for f, test_idx in enumerate(folds):
                t0 = time.time()
                train_mask = np.ones(len(y), dtype=bool); train_mask[test_idx] = False
                xtr, ytr = x[train_mask], y[train_mask]
                xte, yte = x[~train_mask], y[~train_mask]
                hrng = np.random.default_rng(seed + 1000 * r + f)
                row_hash = hrng.integers(0, 1 << 61, len(ytr))
                level: float | None = None
                T = 0
                if selection == "certified":
                    T = occupied_cell_count(xtr, card, schemas, m,
                                            hrng.integers(0, 1 << 26, len(ytr)).astype(float),
                                            hrng.integers(0, 1 << 26, len(ytr)).astype(float))
                    level = alpha / T
                pos_packed = np.packbits(ytr.astype(bool))
                P_te = int(yte.sum())
                # all qualifying cells of all schemas, grouped by schema
                by_schema: Dict[Tuple[int, ...], List[Candidate]] = {}
                for s in schemas:
                    cs = qualifying_cells(xtr, ytr, card, s, tau, m, row_hash, level)
                    if cs:
                        by_schema[s] = cs
                # rules pool: dedupe by row set, keep the cheapest description
                pool: Dict[int, Candidate] = {}
                for cs in by_schema.values():
                    for c in cs:
                        old = pool.get(c.key)
                        if old is None or len(c.schema) < len(old.schema):
                            pool[c.key] = c
                rules = list(pool.values())
                # partial pools: for each top-level schema, all sub-schema cells
                # partial: for each top-level schema, pool = cells of all its sub-schemas.
                # Greedy runs once at max(budgets); smaller budgets take the best prefix
                # (nested selection; a conservative bound on per-budget greedy).
                Bmax = max(budgets)
                PARTIAL = (("partial_disjoint", "disjoint"), ("partial_union", "union"), ("partial_overlap", "overlap"))
                partial_best: Dict[Tuple[str, int], Tuple[List[Candidate], List[int], int]] = {
                    (v, B): ([], [], 0) for v, _ in PARTIAL for B in budgets}
                seen_pools: set = set()
                for s in schemas:
                    subs = [ss for d in range(1, len(s) + 1) for ss in itertools.combinations(s, d)]
                    live = tuple(ss for ss in subs if ss in by_schema)
                    if not live or live in seen_pools:
                        continue
                    seen_pools.add(live)
                    seen: Dict[int, Candidate] = {}
                    for ss in live:
                        for c in by_schema[ss]:
                            o = seen.get(c.key)
                            if o is None or len(c.schema) < len(o.schema):
                                seen[c.key] = c
                    cs = list(seen.values())
                    ks = np.sort(np.array([c.k for c in cs]))[::-1]
                    bound = {B: int(ks[:B].sum()) for B in budgets}   # <= B centres of cost >= 1
                    if all(bound[B] <= partial_best[("partial_overlap", B)][2] for B in budgets):
                        continue
                    for variant, mode in PARTIAL:
                        for traj in greedy_trajectories(cs, pos_packed, Bmax, mode, tau):
                            for B in budgets:
                                idx, cov = prefix_at_budget(traj, B)
                                if cov > partial_best[(variant, B)][2]:
                                    partial_best[(variant, B)] = (cs, idx, cov)
                for B in budgets:
                    results: Dict[str, Tuple[List[Candidate], List[int], int]] = {}
                    for v, _ in PARTIAL:
                        results[v] = partial_best[(v, B)]
                    # grid: best schema at this budget, top floor(B/d) cells by k
                    best_grid = ([], [], 0)
                    for s, cs in by_schema.items():
                        kk = B // len(s)
                        if kk == 0:
                            continue
                        top = sorted(range(len(cs)), key=lambda i: -cs[i].k)[:kk]
                        cov = sum(cs[i].k for i in top)
                        if cov > best_grid[2]:
                            best_grid = (cs, top, cov)
                    results["grid"] = best_grid
                    for variant, mode in (("rules_disjoint", "disjoint"), ("rules_union", "union"), ("rules_overlap", "overlap")):
                        idx, cov = greedy_select(rules, pos_packed, B, mode, tau)
                        results[variant] = (rules, idx, cov)
                    for method, (cs, idx, cov_tr) in results.items():
                        sel = union_rows_test(cs, idx, xte, card) if idx else np.zeros(len(yte), bool)
                        n_sel = int(sel.sum()); k_sel = int(yte[sel].sum())
                        rows.append(dict(
                            dataset=name, N=len(y), M=M, base_rate=round(base, 4), tau=tau, m=m,
                            selection=selection, family_T=T,
                            budget=B, repeat=r, fold=f, method=method,
                            n_centres=len(idx),
                            conditions_used=sum(len(cs[i].schema) for i in idx),
                            schema="|".join(sorted({",".join(map(str, cs[i].schema)) for i in idx})),
                            train_coverage=cov_tr / max(int(ytr.sum()), 1),
                            test_coverage=k_sel / max(P_te, 1),
                            test_purity=(k_sel / n_sel) if n_sel else np.nan,
                            test_rows=n_sel,
                            pool_size=len(cs),
                        ))
                print(f"{name} tau={tau} r={r} f={f} schemas_with_cells={len(by_schema)} "
                      f"rules={len(rules)} {time.time() - t0:.1f}s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/external")
    ap.add_argument("--out", default="experiments/results/partial_external/part.csv")
    ap.add_argument("--datasets", nargs="*", default=list(LOADERS))
    ap.add_argument("--taus", nargs="*", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--budgets", nargs="*", type=int, default=[2, 4, 8, 16, 32])
    ap.add_argument("--d_max", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selection", choices=["purity", "certified"], default="purity")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []
    for ds in a.datasets:
        name, x, y, cols = LOADERS[ds](Path(a.data))
        print(f"== {name}: N={len(y)} M={x.shape[1]} base={y.mean():.3f}", flush=True)
        all_rows += run_dataset(name, x, y, a.taus, a.m, a.budgets, a.d_max, a.repeats, a.splits, a.seed,
                               a.selection, a.alpha)
        pd.DataFrame(all_rows).to_csv(out, index=False)
    print("written", out)


if __name__ == "__main__":
    main()
