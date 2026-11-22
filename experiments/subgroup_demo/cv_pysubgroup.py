"""
pysubgroup on titanic under the phase-3.2 protocol, paired with VSF.

Same folds (`vsf.centers.stratified_repeated_kfold`, seed 0, 5x5), same
budgets, same qualifying rule (`baselines._qualifying`: observed purity or
the product's family certificate), same held-out metrics. The VSF,
`rules` and `tree` outcomes are read from `results/cache/` written by
`compare_baselines.py --only titanic`, so every difference is paired.

pysubgroup methods (Apriori, exhaustive to depth `max_d`, StandardQF a=1 =
WRAcc, MinSupportConstraint(m); rules ranked by WRAcc and taken in that
order while the condition budget allows, i.e. the usual top-k output):
  psg_topk      no purity filter: what a subgroup-discovery user gets.
  psg_filtered  only rules that qualify under the configuration's selection.

Rules may overlap, so the union of qualifying rules can fall below tau even
on training rows; that is reported, not repaired.

Usage: python experiments/subgroup_demo/cv_pysubgroup.py
Output: experiments/results/pysubgroup_titanic.csv
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pysubgroup as ps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from baselines import BUDGETS, _qualifying, certification_threshold, net_coverage, description_stability  # noqa: E402
from compare_baselines import CONFIGS, N_SPLITS, Config, _cache_path, _load  # noqa: E402
from datasets import load_dataset  # noqa: E402
from vsf.avr import _CandidateFactory, _prepare_search, resolve_center_spec  # noqa: E402
from vsf.centers import CenterSpec, stratified_repeated_kfold  # noqa: E402

RESULTS = HERE.parent / "results"
ALPHA = 0.05
REFERENCE = ("vsf", "rules", "tree")


@dataclass(frozen=True)
class Rule:
    label: str
    cost: int
    wracc: float
    members: np.ndarray  # bool over ALL rows


@dataclass(frozen=True)
class Outcome:
    k_in: int
    n_in: int
    n_pos: int
    conditions: int
    train_purity: float
    inside: np.ndarray

    @property
    def coverage(self) -> float:
        return self.k_in / self.n_pos if self.n_pos else 0.0

    @property
    def purity(self) -> float:
        return self.k_in / self.n_in if self.n_in else float("nan")


def mine(frame: pd.DataFrame, target_col: str, positive: str, train: np.ndarray,
         m: int, max_d: int) -> Tuple[List[Rule], int]:
    """All conjunctions of <= max_d equality selectors with training support >= m, WRAcc > 0."""
    tr = frame.iloc[train].reset_index(drop=True)
    task = ps.SubgroupDiscoveryTask(
        tr, ps.BinaryTarget(target_col, positive), ps.create_selectors(tr, ignore=[target_col]),
        result_set_size=10**7, depth=max_d, qf=ps.StandardQF(a=1.0),
        constraints=[ps.MinSupportConstraint(max(1, m))],
    )
    result = ps.Apriori().execute(task)
    rules = [Rule(str(sg), len(sg.selectors), float(q), np.asarray(sg.covers(frame), dtype=bool))
             for q, sg in result.to_descriptions() if q > 0]
    rules.sort(key=lambda r: (-r.wracc, r.cost, r.label))
    return rules, len(result.to_descriptions())


def threshold_for(X: np.ndarray, Z: np.ndarray, positive: str, train: np.ndarray,
                  cfg: Config) -> Optional[np.ndarray]:
    """The product's family certificate on the training fold, exactly as `baselines.run_split`."""
    if cfg.selection != "certified":
        return None
    spec = CenterSpec(tau=cfg.tau, min_samples=max(1, cfg.min_samples))
    prepared = _prepare_search(X, Z, None, positive, spec, "presence")
    assert prepared is not None
    factory, _z, _names = prepared
    X_all = factory._raw.astype(np.int64, copy=False)
    fit = _CandidateFactory(X_all[train], factory.bin_counts, int(train.size), ordered=factory.ordered)
    chain = resolve_center_spec(
        fit, CenterSpec(tau=cfg.tau, min_samples=max(1, cfg.min_samples), rule="certified",
                        multiplicity="family", alpha=ALPHA),
        min(cfg.max_d, fit.n_features),
    )
    return certification_threshold(int(train.size), cfg.tau, chain.effective_alpha(0))


def select(rules: Sequence[Rule], budget: int) -> List[Rule]:
    chosen: List[Rule] = []
    left = budget
    for r in rules:
        if r.cost <= left:
            chosen.append(r)
            left -= r.cost
        if left == 0:
            break
    return chosen


def evaluate(chosen: Sequence[Rule], z: np.ndarray, train: np.ndarray, test: np.ndarray) -> Outcome:
    inside = np.zeros(z.size, dtype=bool)
    for r in chosen:
        inside |= r.members
    tr_in = inside[train]
    return Outcome(
        k_in=int(z[test][inside[test]].sum()), n_in=int(inside[test].sum()), n_pos=int(z[test].sum()),
        conditions=int(sum(r.cost for r in chosen)),
        train_purity=float(z[train][tr_in].mean()) if tr_in.any() else float("nan"),
        inside=inside,
    )


def nb_se(values: np.ndarray, n_splits: int) -> float:
    """Nadeau-Bengio corrected standard error of the mean over repeated K-fold splits."""
    v = float(np.var(values, ddof=1)) if values.size > 1 else 0.0
    return float(np.sqrt((1.0 / values.size + 1.0 / (n_splits - 1)) * v))


def main() -> int:
    frame, entry = load_dataset("titanic")
    feats = [c for c in frame.columns if c != entry.target]
    X, Z = frame[feats].values, frame[entry.target].values
    configs = [c for c in CONFIGS if c.dataset == "titanic"]
    configs += [Config("titanic", "survived", 0.7, selection="certified")]
    rows: List[Dict[str, object]] = []
    for cfg in configs:
        z = (Z == cfg.positive)
        splits = stratified_repeated_kfold(z.astype(np.int64), N_SPLITS, cfg.n_repeats, 0)
        per: Dict[str, Dict[int, List[Outcome]]] = {}
        ref: Dict[str, Dict[int, List[object]]] = {m: {b: [] for b in BUDGETS} for m in REFERENCE}
        n_family: List[int] = []
        t0 = time.perf_counter()
        for i, (train, test) in enumerate(splits):
            cache = _cache_path("titanic", cfg, X, Z, 0, i)
            if not cache.exists():
                raise SystemExit(f"missing {cache.name}: run compare_baselines.py --only titanic first")
            cached, _ = _load(cache)
            for mth in REFERENCE:
                for b in BUDGETS:
                    ref[mth][b].append(cached[mth][b])
            rules, n_all = mine(frame, entry.target, cfg.positive, train, cfg.min_samples, cfg.max_d)
            n_family.append(n_all)
            thr = threshold_for(X, Z, cfg.positive, train, cfg)
            ztr = z[train].astype(np.int64)
            k = np.array([int(ztr[r.members[train]].sum()) for r in rules], dtype=np.int64)
            n = np.array([int(r.members[train].sum()) for r in rules], dtype=np.int64)
            q = _qualifying(k, n, cfg.tau, cfg.min_samples, thr)
            pools = {"psg_topk": rules, "psg_filtered": [r for r, ok in zip(rules, q) if ok]}
            for name, pool in pools.items():
                for b in BUDGETS:
                    per.setdefault(name, {}).setdefault(b, []).append(evaluate(select(pool, b), z, train, test))
        secs = time.perf_counter() - t0
        allm = {**per, **ref}
        for mth, by_b in allm.items():
            for b, outs in by_b.items():
                cov = np.array([o.coverage for o in outs])
                pur = np.array([o.purity for o in outs])
                net = np.array([net_coverage(o.k_in, o.n_in, o.n_pos, cfg.tau) for o in outs])
                vcov = np.array([o.coverage for o in ref["vsf"][b]])
                vnet = np.array([net_coverage(o.k_in, o.n_in, o.n_pos, cfg.tau) for o in ref["vsf"][b]])
                reached = [o.k_in >= cfg.tau * o.n_in - 1e-9 for o in outs if o.n_in > 0]
                tp = [o.train_purity for o in outs if isinstance(o, Outcome)]
                rows.append({
                    "tau": cfg.tau, "m": cfg.min_samples, "selection": cfg.selection, "method": mth,
                    "budget": b, "coverage": round(float(cov.mean()), 4),
                    "coverage_se": round(nb_se(cov, N_SPLITS), 4),
                    "purity": round(float(np.nanmean(pur)), 4) if np.isfinite(pur).any() else float("nan"),
                    "purity_reaches_tau": round(float(np.mean(reached)), 3) if reached else float("nan"),
                    "train_union_purity": round(float(np.nanmean(tp)), 4) if tp and np.isfinite(tp).any() else float("nan"),
                    "net_coverage": round(float(net.mean()), 4),
                    "minus_vsf": round(float((cov - vcov).mean()), 4),
                    "minus_vsf_se": round(nb_se(cov - vcov, N_SPLITS), 4),
                    "net_minus_vsf": round(float((net - vnet).mean()), 4),
                    "net_minus_vsf_se": round(nb_se(net - vnet, N_SPLITS), 4),
                    "stability": round(description_stability([o.inside for o in outs]), 3),
                    "conditions": round(float(np.mean([o.conditions for o in outs])), 2),
                })
        print(f"tau={cfg.tau} m={cfg.min_samples} {cfg.selection}: {secs:.1f}s, "
              f"pysubgroup family (all qualities) median {int(np.median(n_family))}", flush=True)
    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "pysubgroup_titanic.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
