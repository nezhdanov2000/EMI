"""
Parallel ablation runner for mushroom dataset across 12 combinations.
Runs in parallel on multi-core CPU, writes directly to ablations.csv.
"""
from __future__ import annotations

import concurrent.futures
import csv
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import (
    held_out_counts, select_rules, select_vsf, select_vsf_partial, union_members,
)
from compare_baselines import CONFIGS, N_SPLITS, RESULTS
from datasets import load_dataset
from vsf.avr import _CandidateFactory, _prepare_search
from vsf.centers import CenterSpec, stratified_repeated_kfold

DIVISORS = (5, 10, 20, None)
FLOORS = (1, 5, 20)
BUDGETS_ABL = (8, 32)
METHODS_ABL = ("vsf", "vsf_partial", "rules_disjoint")


def run_single_combination(args: tuple) -> Tuple[tuple, List[Dict[str, object]], float]:
    dataset, positive, tau, n_repeats, seed, divisor, m = args
    frame, entry = load_dataset(dataset)
    feats = [c for c in frame.columns if c != entry.target]
    X, Z = frame[feats].values, frame[entry.target].values
    prepared = _prepare_search(X, Z, None, positive, CenterSpec(tau=tau, min_samples=1), "presence")
    if prepared is None:
        raise ValueError("no feature columns")
    factory, z, names = prepared
    X_all = factory._raw.astype(np.int64, copy=False)
    splits = stratified_repeated_kfold(z, N_SPLITS, n_repeats, seed)

    acc: Dict[tuple, List[tuple]] = {}
    t0 = time.perf_counter()
    print(f"[START] {dataset} tau={tau} c={divisor} m={m}", flush=True)
    for train, test in splits:
        z_train = z[train]
        fit = _CandidateFactory(X_all[train], factory.bin_counts, int(train.size), ordered=factory.ordered)
        fit.capacity = (max(1, int(train.size) // divisor) if divisor is not None
                        else int(np.iinfo(np.int64).max))
        sel = {
            "vsf": select_vsf(fit, X_all, train, z_train, tau, m, BUDGETS_ABL, names),
            "vsf_partial": select_vsf_partial(fit, X_all, train, z_train, tau, m, BUDGETS_ABL, names),
            "rules_disjoint": select_rules(X_all, train, z_train, tau, m, BUDGETS_ABL, names, disjoint=True),
        }
        for mth, res in sel.items():
            for b in BUDGETS_ABL:
                groups = res.groups[b]
                inside = union_members(groups, z.shape[0])
                k_in, n_in, n_pos = held_out_counts(inside, z, test)
                acc.setdefault((mth, b), []).append((k_in, n_in, n_pos, sum(g.cost for g in groups)))

    seconds = time.perf_counter() - t0
    combo_rows = []
    for (mth, b), vals in acc.items():
        cov = np.array([k / max(1, p) for k, _, p, _ in vals])
        pur = np.array([k / n if n else np.nan for k, n, _, _ in vals])
        combo_rows.append(dict(
            dataset=dataset, positive=positive, tau=tau, divisor=divisor if divisor is not None else "none",
            m=m, budget=b, method=mth, coverage=round(float(cov.mean()), 4),
            purity=round(float(np.nanmean(pur)), 4) if np.isfinite(pur).any() else "",
            purity_reaches_tau=round(float(np.nanmean(pur >= tau)), 3) if np.isfinite(pur).any() else "",
            conditions=round(float(np.mean([c for _, _, _, c in vals])), 2),
            splits=len(vals), seconds=round(seconds, 1),
        ))
    print(f"[DONE] {dataset} tau={tau} c={divisor} m={m} in {seconds:.0f}s", flush=True)
    return (divisor, m), combo_rows, seconds


def main() -> int:
    dataset = "mushroom"
    seed = 0
    configs = [c for c in CONFIGS if c.dataset == dataset and c.selection == "purity"]
    cfg = configs[0]
    n_repeats = min(cfg.n_repeats, 2)

    tasks = []
    for divisor in DIVISORS:
        for m in FLOORS:
            tasks.append((dataset, cfg.positive, cfg.tau, n_repeats, seed, divisor, m))

    print(f"Launching {len(tasks)} parallel ablation tasks across available CPU cores...", flush=True)
    results_map = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(12, len(tasks))) as executor:
        futures = {executor.submit(run_single_combination, t): t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            key, rows, sec = f.result()
            results_map[key] = rows

    all_rows = []
    for divisor in DIVISORS:
        for m in FLOORS:
            all_rows.extend(results_map[(divisor, m)])

    out = RESULTS / "ablations.csv"
    existing: List[Dict[str, object]] = []
    if out.exists():
        with out.open(newline="", encoding="utf-8") as fh:
            existing = [r for r in csv.DictReader(fh) if r["dataset"] != dataset]

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(existing)
        w.writerows(all_rows)

    print(f"Successfully wrote {out} ({len(all_rows)} rows for mushroom, {len(existing) + len(all_rows)} total rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
