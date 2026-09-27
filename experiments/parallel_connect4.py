"""
Parallel runner for connect_4 across multiple CPU cores, followed by ablations and summaries.
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
import numpy as np


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
    print(f"[DONE] {tag} {positive} tau={tau} sel={selection} split={split_idx} in {sec:.1f}s")
    sys.stdout.flush()
    return path.name, sec


def main() -> int:
    seed = 0
    frame, entry = load_dataset("connect_4")
    feats = [c for c in frame.columns if c != entry.target]
    X = frame[feats].values
    Z = frame[entry.target].values

    configs = [c for c in CONFIGS if c.dataset == "connect_4"]

    tasks = []
    for cfg in configs:
        for i in range(N_SPLITS * cfg.n_repeats):
            path = _cache_path("connect_4", cfg, X, Z, seed, i)
            if not path.exists():
                tasks.append((
                    "connect_4", cfg.positive, cfg.tau, cfg.min_samples,
                    cfg.n_repeats, cfg.max_d, cfg.selection, i, str(path), seed
                ))

    print(f"Total remaining connect_4 split tasks: {len(tasks)}")
    if tasks:
        # Use 8 parallel worker processes (leaves 14 logical cores free for the system)
        num_workers = min(8, len(tasks))
        print(f"Starting {num_workers} parallel workers on available CPU cores...")
        sys.stdout.flush()

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_single_split, t) for t in tasks]
            for f in concurrent.futures.as_completed(futures):
                try:
                    name, sec = f.result()
                except Exception as e:
                    print(f"Error in worker: {e}", file=sys.stderr)
                    sys.stderr.flush()

    print("\n=======================================================")
    print(">>> 1. Finalizing connect_4 comparison CSV...")
    print("=======================================================")
    subprocess.run([sys.executable, "experiments/compare_baselines.py", "--only", "connect_4"], check=True)

    print("\n=======================================================")
    print(">>> 2. Running ablations on 7 datasets...")
    print("=======================================================")
    ablations = ["car_evaluation", "nursery", "hmda", "credit", "mlc_churn", "adult", "mushroom"]
    for ds in ablations:
        subprocess.run([sys.executable, "experiments/ablations.py", "--only", ds], check=True)

    print("\n=======================================================")
    print(">>> 3. Generating final paper tables and summary...")
    print("=======================================================")
    subprocess.run([sys.executable, "experiments/pairwise_partial.py"], check=True)
    with open("experiments/results/summary.txt", "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "experiments/summarize_comparison.py"], stdout=f, check=True)
    subprocess.run([sys.executable, "experiments/paper_tables.py"], check=True)

    print("\n=======================================================")
    print(">>> ALL DONE! Full phase-3 paper results ready!")
    print("=======================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
