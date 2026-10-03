"""
Parallel runner for connect_4 across CPU cores, followed by summaries and paper tables.
Optimized for Core Ultra 9 (3 workers to safely stay within 32 GB RAM, 6 BLAS threads each).
"""
from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time

# Allow BLAS/NumPy to use multiple threads per worker process
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset
from compare_baselines import CONFIGS, _cache_path, _save, _save_extra, N_SPLITS
from baselines import BUDGETS, METHODS, run_split

WANTED_METHODS = ("vsf", "vsf_partial", "vsf_greedy", "rules", "rules_pure", "rules_disjoint", "tree", "ssdpp", "wracc", "wracc_raw")


def run_single_split(args: tuple) -> tuple[str, float]:
    tag, positive, tau, min_samples, n_repeats, max_d, selection, split_idx, path_str, seed = args
    path = Path(path_str)
    if path.exists():
        print(f"[SKIP] Split {split_idx} already cached: {path.name}", flush=True)
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

    print(f"Total remaining connect_4 split tasks: {len(tasks)}", flush=True)
    for t in tasks:
        print(f"  - {t[0]} {t[1]} tau={t[2]} sel={t[6]} split={t[7]}", flush=True)

    if tasks:
        # Use 3 parallel workers (each consumes ~5-6 GB RAM in peak rule generation, perfectly fitting in 18 GB free RAM)
        num_workers = min(3, len(tasks))
        print(f"\nStarting {num_workers} parallel workers (batch 1: 3 tasks, batch 2: 2 tasks)...", flush=True)

        t0 = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(run_single_split, t): t for t in tasks}
            for f in concurrent.futures.as_completed(futures):
                try:
                    name, sec = f.result()
                    print(f"[COMPLETED] {name} in {sec:.1f}s", flush=True)
                except Exception as e:
                    print(f"Error in worker: {e}", file=sys.stderr, flush=True)

        total_sec = time.time() - t0
        print(f"\nAll connect_4 splits finished in {total_sec:.1f}s ({total_sec/60:.1f}m)!", flush=True)

    root_dir = str(Path(__file__).resolve().parents[1])

    print("\n=======================================================", flush=True)
    print(">>> 1. Finalizing comparison CSVs...", flush=True)
    print("=======================================================", flush=True)
    for d in ["connect_4", "adult", "chess_krkp", "splice"]:
        print(f"Generating compare_{d}.csv...", flush=True)
        subprocess.run([
            sys.executable, "experiments/compare_baselines.py", "--only", d,
            "--methods", ",".join(WANTED_METHODS), "--max-seconds", "0"
        ], cwd=root_dir)

    print("\n=======================================================", flush=True)
    print(">>> 2. Running pairwise partial summary and paper tables...", flush=True)
    print("=======================================================", flush=True)
    subprocess.run([sys.executable, "experiments/pairwise_partial.py"], cwd=root_dir, check=True)

    with open(Path(root_dir) / "experiments/results/summary.txt", "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "experiments/summarize_comparison.py"], stdout=f, cwd=root_dir, check=True)

    subprocess.run([sys.executable, "experiments/paper_tables.py"], cwd=root_dir, check=True)
    subprocess.run([sys.executable, "experiments/compactness.py"], cwd=root_dir, check=True)

    print("\n=======================================================", flush=True)
    print(">>> 3. Checking numbers against paper assertions...", flush=True)
    print("=======================================================", flush=True)
    subprocess.run([sys.executable, "experiments/check_numbers.py"], cwd=root_dir)

    print("\n=======================================================", flush=True)
    print(">>> ALL DONE! Full phase-3 paper results ready!", flush=True)
    print("=======================================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
