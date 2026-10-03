"""
Parallel runner for remaining split configurations on multi-core CPU.
"""
from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset
from compare_baselines import CONFIGS, _cache_path, _save, _save_extra, N_SPLITS
from baselines import BUDGETS, METHODS, run_split

WANTED_METHODS = ("vsf", "vsf_partial", "vsf_greedy", "rules", "rules_pure", "rules_disjoint", "tree", "ssdpp", "wracc", "wracc_raw")
TARGET_DATASETS = ["chess_krkp", "splice", "adult", "connect_4"]


def run_single_split(args: tuple) -> tuple[str, float]:
    tag, positive, tau, min_samples, n_repeats, max_d, selection, split_idx, path_str, seed = args
    path = Path(path_str)
    if path.exists():
        return path.name, 0.0

    frame, entry = load_dataset(tag)
    feats = [c for c in frame.columns if c != entry.target]
    X = frame[feats].values
    Z = frame[entry.target].values

    t0 = time.perf_counter()
    print(f"[START] {tag} {positive} tau={tau} sel={selection} split={split_idx}", flush=True)

    res = run_split(
        X, Z, positive, tau, min_samples, split_idx, N_SPLITS,
        n_repeats, seed, BUDGETS, WANTED_METHODS, max_d, selection
    )
    sec = time.perf_counter() - t0

    main_res = {m: r for m, r in res.items() if m in METHODS}
    extra_res = {m: r for m, r in res.items() if m in ("wracc", "wracc_raw")}
    _save(path, main_res, sec)
    if extra_res:
        _save_extra(path, extra_res, 0.0)

    print(f"[DONE] {tag} {positive} tau={tau} sel={selection} split={split_idx} in {sec:.1f}s ({sec/60:.1f}m)", flush=True)
    return path.name, sec


def main() -> int:
    seed = 0
    tasks = []

    print("Checking remaining split configurations across target datasets...", flush=True)
    frames = {}
    for d in TARGET_DATASETS:
        frame, entry = load_dataset(d)
        feats = [c for c in frame.columns if c != entry.target]
        frames[d] = (frame[feats].values, frame[entry.target].values)

    configs = [c for c in CONFIGS if c.dataset in TARGET_DATASETS and c.selection == "split"]

    for cfg in configs:
        X, Z = frames[cfg.dataset]
        for i in range(N_SPLITS * cfg.n_repeats):
            path = _cache_path(cfg.dataset, cfg, X, Z, seed, i)
            if not path.exists():
                tasks.append((
                    cfg.dataset, cfg.positive, cfg.tau, cfg.min_samples, cfg.n_repeats,
                    cfg.max_d, cfg.selection, i, str(path), seed
                ))

    print(f"Total missing splits to compute: {len(tasks)}", flush=True)
    for t in tasks:
        print(f"  - {t[0]} {t[1]} tau={t[2]} split={t[7]}", flush=True)

    if not tasks:
        print("All target splits already completed!", flush=True)
    else:
        # Use 18 workers on 22 logical cores (keeps system completely responsive)
        max_workers = min(len(tasks), max(1, min(18, (os.cpu_count() or 4) - 2)))
        print(f"\nLaunching pool with {max_workers} parallel workers on {os.cpu_count()} CPU cores...", flush=True)
        t0 = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_split, t): t for t in tasks}
            for fut in concurrent.futures.as_completed(futures):
                name, sec = fut.result()
                print(f"[COMPLETED] {name} in {sec:.1f}s", flush=True)

        total_time = time.time() - t0
        print(f"\nAll {len(tasks)} splits finished in {total_time:.1f}s ({total_time/60:.1f}m)!", flush=True)

    # Now update summaries and tables
    print("\nUpdating comparison CSVs and LaTeX tables...", flush=True)
    for d in TARGET_DATASETS:
        print(f"Summarising {d}...", flush=True)
        subprocess.run([sys.executable, "experiments/compare_baselines.py", "--only", d,
                        "--methods", ",".join(WANTED_METHODS)], cwd=str(Path(__file__).resolve().parents[1]))

    subprocess.run([sys.executable, "experiments/pairwise_partial.py"], cwd=str(Path(__file__).resolve().parents[1]))
    subprocess.run([sys.executable, "experiments/summarize_comparison.py"], cwd=str(Path(__file__).resolve().parents[1]))
    subprocess.run([sys.executable, "experiments/paper_tables.py"], cwd=str(Path(__file__).resolve().parents[1]))
    subprocess.run([sys.executable, "experiments/compactness.py"], cwd=str(Path(__file__).resolve().parents[1]))
    print("ALL TARGET SPLITS AND TABLES UPDATED!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
