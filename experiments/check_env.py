"""
Does this Python environment reproduce the cached results?

Recomputes a few cached splits of the paper's configurations on small
tables, in memory, and compares every (method, budget) outcome with the
cache: the rows selected on the held-out fold (`k_in`, `n_in`) and the
conditions spent. Nothing is written. Run it on every machine before a
long run; all machines must print the same verdict.

    python experiments/check_env.py
    python experiments/check_env.py --datasets titanic,hmda --splits 3

Exit status 0 when every compared outcome is identical, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from baselines import BUDGETS, run_split, unavailable_methods  # noqa: E402
from compare_baselines import CONFIGS, N_SPLITS, _cache_path, _load  # noqa: E402
from datasets import load_dataset  # noqa: E402

CORE: Tuple[str, ...] = ("vsf", "vsf_partial", "vsf_greedy", "rules", "rules_pure", "rules_disjoint", "tree")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="titanic,car_evaluation,breast_cancer")
    ap.add_argument("--splits", type=int, default=2, help="splits per configuration to recompute")
    args = ap.parse_args()
    import pandas
    import scipy
    try:
        import sklearn
        sk = sklearn.__version__
    except Exception:
        sk = "MISSING"
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pandas.__version__}, "
          f"scipy {scipy.__version__}, scikit-learn {sk}")
    methods = [m for m in CORE if m not in unavailable_methods(CORE)]
    compared: Dict[str, int] = {m: 0 for m in methods}
    different: Dict[str, int] = {m: 0 for m in methods}
    examples: List[str] = []
    not_cached = 0
    for name in [d for d in args.datasets.split(",") if d]:
        frame, entry = load_dataset(name)
        feats = [c for c in frame.columns if c != entry.target]
        X, Z = frame[feats].values, frame[entry.target].values
        for cfg in [c for c in CONFIGS if c.dataset == name and c.selection == "purity"]:
            for split in range(min(args.splits, N_SPLITS * cfg.n_repeats)):
                path = _cache_path(name, cfg, X, Z, 0, split)
                if not path.exists():
                    not_cached += 1
                    continue
                cached, _ = _load(path)
                fresh = run_split(X, Z, cfg.positive, cfg.tau, cfg.min_samples, split, N_SPLITS, cfg.n_repeats,
                                  0, BUDGETS, methods, cfg.max_d, cfg.selection)  # type: ignore[arg-type]
                for m in methods:
                    if m not in cached:
                        continue
                    for b in BUDGETS:
                        compared[m] += 1
                        a, f = cached[m][b], fresh[m][b]
                        if (a.k_in, a.n_in, a.conditions) != (f.k_in, f.n_in, f.conditions):
                            different[m] += 1
                            if len(examples) < 6:
                                examples.append(f"{name} tau={cfg.tau} m={cfg.min_samples} split {split} {m} B={b}: "
                                                f"cache k={a.k_in} n={a.n_in} cond={a.conditions}; "
                                                f"here k={f.k_in} n={f.n_in} cond={f.conditions}")
    if not_cached:
        print(f"WARNING: {not_cached} split(s) were not found in the cache. If none is found at all, this environment "
              f"reads the data differently (the cache key is a hash of the table) - do not start a long run.")
    for m in methods:
        verdict = "identical" if different[m] == 0 else f"{different[m]} DIFFERENT"
        print(f"  {m:15s} {compared[m]:4d} outcomes compared: {verdict}")
    for line in examples:
        print("  e.g. " + line)
    total = sum(compared.values())
    bad = sum(different.values())
    if total == 0:
        print("VERDICT: nothing could be compared - the cache was not found for this environment")
        return 1
    if bad == 0:
        print("VERDICT: this environment reproduces the cache")
        return 0
    only_tree = all(different[m] == 0 for m in methods if m != "tree")
    print("VERDICT: " + ("only the CART tree differs (scikit-learn version); the grid and rule methods are identical"
                         if only_tree else "the grid or rule methods differ - do not start a long run, report this"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
