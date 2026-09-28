"""
Parallel runner for missing adult certified splits across multiple CPU cores.
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
from compare_baselines import CONFIGS, _cache_path, _save, N_SPLITS
from baselines import BUDGETS, METHODS, run_split


def run_single_split(args: tuple) -> tuple[str, float]:
    tag, positive, tau, min_samples, n_repeats, max_d, selection, split_idx, path_str, seed = args
    path = Path(path_str)
    if path.exists():
        print(f"[SKIP] Split {split_idx} already cached: {path.name}")
        return path.name, 0.0

    frame, entry = load_dataset(tag)
    feats = [c for c in frame.columns if c != entry.target]
    X = frame[feats].values
    Z = frame[entry.target].values

    t0 = time.perf_counter()
    print(f"[START] {tag} {positive} tau={tau} sel={selection} split={split_idx}")
    sys.stdout.flush()

    res = run_split(
        X, Z, positive, tau, min_samples, split_idx, N_SPLITS,
        n_repeats, seed, BUDGETS, METHODS, max_d, selection
    )
    sec = time.perf_counter() - t0
    _save(path, res, sec)
    print(f"[DONE] {tag} {positive} tau={tau} sel={selection} split={split_idx} in {sec:.1f}s ({sec/60:.1f}m)")
    sys.stdout.flush()
    return path.name, sec


def main() -> int:
    seed = 0
    frame, entry = load_dataset("adult")
    feats = [c for c in frame.columns if c != entry.target]
    X = frame[feats].values
    Z = frame[entry.target].values

    configs = [c for c in CONFIGS if c.dataset == "adult" and c.selection == "certified"]

    tasks = []
    for cfg in configs:
        for i in range(N_SPLITS * cfg.n_repeats):
            path = _cache_path("adult", cfg, X, Z, seed, i)
            if not path.exists():
                tasks.append((
                    "adult", cfg.positive, cfg.tau, cfg.min_samples, cfg.n_repeats,
                    cfg.max_d, cfg.selection, i, str(path), seed
                ))

    print(f"Total adult certified splits missing: {len(tasks)}")
    if not tasks:
        print("All splits already completed!")
        return 0

    max_workers = min(len(tasks), max(1, (os.cpu_count() or 4) - 2))
    print(f"Launching pool with {max_workers} parallel workers...")
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_split, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            name, sec = fut.result()
            print(f"Finished task: {name} in {sec:.1f}s")

    total_time = time.time() - t0
    print(f"\nAll {len(tasks)} missing adult splits finished in {total_time:.1f}s ({total_time/60:.1f}m)!")

    # Now update compare_adult.csv, summary.txt, pairwise_partial.csv, paper tables
    print("\nUpdating comparison and tables...")
    subprocess.run([sys.executable, "experiments/compare_baselines.py", "--only", "adult"])
    subprocess.run([sys.executable, "experiments/pairwise_partial.py"])
    subprocess.run([sys.executable, "experiments/paper_tables.py"])
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
