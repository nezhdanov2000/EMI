"""
Phase 3.2: VSF against other descriptions at equal condition budgets.

Methods, budgets and the evaluation protocol are defined in
`experiments/baselines.py`. For each configuration this writes one row per
(method, budget) with the held-out coverage (Nadeau-Bengio mean and se over
the repeated 5-fold splits), the mean held-out purity, the conditions
actually spent, the training coverage, and the paired difference of the
held-out coverage to `vsf` on the same folds.

Synthetic configurations (`--synthetic`):
  grid       positives at (x0=0, x1=0) and (x0=1, x1=1): one 2-D grid holds both
  two_rules  positives at (x0=0, x1=0) and (x2=0, x3=0): no common 2-D grid
  xor        positive iff x0 xor x1 (binary), other columns noise

Every split is cached in results/cache/ as soon as it is computed, keyed by
the data's hash and the configuration; a call runs at most `--max-seconds`
of new splits and writes the summary once every split of every
configuration is present. Large datasets are therefore finished by
repeating the same command.

Usage:
    python experiments/compare_baselines.py --only titanic
    python experiments/compare_baselines.py --only chess_krkp --max-seconds 150   # repeat until done
    python experiments/compare_baselines.py --synthetic two_rules
Results: results/compare_<name>.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import BUDGETS, METHODS, SplitOutcome, SplitResult, run_split, summarize_splits  # noqa: E402
from datasets import load_dataset  # noqa: E402
from vsf.centers import paired_gain  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class Config:
    dataset: str
    positive: str
    tau: float
    min_samples: int = 1
    n_repeats: int = 5
    max_d: int = 4


CONFIGS: List[Config] = [
    Config("titanic", "survived", 0.9),
    Config("titanic", "survived", 0.7),
    Config("titanic", "survived", 0.9, min_samples=20),
    Config("car_evaluation", "acc", 0.9),
    Config("nursery", "spec_prior", 0.9),
    Config("nursery", "priority", 0.7),
    Config("mushroom", "poisonous", 0.9, n_repeats=2),
    Config("chess_krkp", "won", 0.9, n_repeats=1),
    Config("splice_junction", "EI", 0.9, n_repeats=1, max_d=2),
]


def synthetic(name: str, seed: int = 0, n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if name == "grid":
        X = rng.integers(0, 3, size=(n, 6))
        inside = ((X[:, 0] == 0) & (X[:, 1] == 0)) | ((X[:, 0] == 1) & (X[:, 1] == 1))
        p = np.where(inside, 0.97, 0.03)
    elif name == "two_rules":
        X = rng.integers(0, 3, size=(n, 6))
        inside = ((X[:, 0] == 0) & (X[:, 1] == 0)) | ((X[:, 2] == 0) & (X[:, 3] == 0))
        p = np.where(inside, 0.97, 0.03)
    elif name == "xor":
        X = rng.integers(0, 2, size=(n, 6))
        p = np.where(X[:, 0] != X[:, 1], 0.95, 0.05)
    else:
        raise SystemExit(f"unknown synthetic dataset {name!r}")
    z = (rng.random(n) < p).astype(int)
    return X.astype(str), np.where(z == 1, "yes", "no")


N_SPLITS = 5
CACHE = RESULTS / "cache"


def _cache_path(tag: str, cfg: Config, X: np.ndarray, Z: np.ndarray, seed: int, split: int) -> Path:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(X, dtype=str)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(Z, dtype=str)).tobytes())
    h.update(json.dumps([cfg.__dict__, seed, list(BUDGETS), list(METHODS), N_SPLITS]).encode())
    return CACHE / f"{tag}_{h.hexdigest()[:16]}_split{split:03d}.json"


def _save(path: Path, result: SplitResult, seconds: float) -> None:
    payload = {
        "seconds": seconds,
        "methods": {
            mth: {str(b): {"coverage": o.coverage, "purity": None if np.isnan(o.purity) else o.purity,
                           "conditions": o.conditions, "train_coverage": o.train_coverage,
                           "labels": list(o.labels)}
                  for b, o in per.items()}
            for mth, per in result.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _load(path: Path) -> Tuple[SplitResult, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: SplitResult = {}
    for mth, per in payload["methods"].items():
        result[mth] = {
            int(b): SplitOutcome(
                coverage=float(o["coverage"]),
                purity=float("nan") if o["purity"] is None else float(o["purity"]),
                conditions=int(o["conditions"]), train_coverage=float(o["train_coverage"]),
                labels=tuple(o["labels"]),
            )
            for b, o in per.items()
        }
    return result, float(payload["seconds"])


def _pct(value: object) -> str:
    return "   -" if value == "" else f"{100 * float(value):4.0f}"  # type: ignore[arg-type]


def rows_for(tag: str, cfg: Config, per_split: List[SplitResult], seconds: float) -> List[Dict[str, object]]:
    summary = summarize_splits(per_split, N_SPLITS, cfg.n_repeats)
    out: List[Dict[str, object]] = []
    for mth in METHODS:
        for b in BUDGETS:
            e = summary[mth][b]
            cv = e["coverage"]
            pur = np.asarray(e["per_split_purity"], dtype=float)
            if mth != "vsf":
                g = paired_gain(cv, summary["vsf"][b]["coverage"])  # type: ignore[arg-type]
                diff, diff_se = g.difference, g.se
            else:
                diff, diff_se = 0.0, 0.0
            out.append({
                "dataset": tag, "positive": cfg.positive, "tau": cfg.tau,
                "min_samples": cfg.min_samples, "n_repeats": cfg.n_repeats, "max_d": cfg.max_d,
                "method": mth, "budget": b,
                "coverage": round(cv.mean, 4), "coverage_se": round(cv.se, 4),
                "purity": round(float(np.nanmean(pur)), 4) if np.any(np.isfinite(pur)) else "",
                "splits_with_groups": int(np.isfinite(pur).sum()),
                "conditions_spent": round(float(e["cost_mean"]), 2),
                "train_coverage": round(float(e["train_coverage_mean"]), 4),
                "minus_vsf": round(diff, 4), "minus_vsf_se": round(diff_se, 4),
                "fold0_groups": " | ".join(per_split[0][mth][b].labels),
                "seconds_total": round(seconds, 1),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--only", help="bundled dataset name (all its configurations)")
    src.add_argument("--synthetic", choices=("grid", "two_rules", "xor"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=150.0,
                    help="stop starting new splits after this many seconds")
    args = ap.parse_args()

    if args.synthetic:
        X, Z = synthetic(args.synthetic, args.seed)
        jobs = [(f"synthetic_{args.synthetic}", Config(args.synthetic, "yes", 0.9, min_samples=5), X, Z)]
    else:
        configs = [c for c in CONFIGS if c.dataset == args.only]
        if not configs:
            raise SystemExit(f"no configuration for {args.only!r}")
        frame, entry = load_dataset(args.only)
        feats = [c for c in frame.columns if c != entry.target]
        jobs = [(args.only, c, frame[feats].values, frame[entry.target].values) for c in configs]

    started = time.perf_counter()
    rows: List[Dict[str, object]] = []
    missing = 0
    for tag, cfg, X, Z in jobs:
        per_split: List[SplitResult] = []
        seconds = 0.0
        for i in range(N_SPLITS * cfg.n_repeats):
            path = _cache_path(tag, cfg, X, Z, args.seed, i)
            if path.exists():
                result, sec = _load(path)
            elif time.perf_counter() - started < args.max_seconds:
                t0 = time.perf_counter()
                result = run_split(X, Z, cfg.positive, cfg.tau, cfg.min_samples, i, N_SPLITS,
                                   cfg.n_repeats, args.seed, BUDGETS, METHODS, cfg.max_d)
                sec = time.perf_counter() - t0
                _save(path, result, sec)
            else:
                missing += 1
                continue
            per_split.append(result)
            seconds += sec
        label = f"{tag} {cfg.positive} tau={cfg.tau} m={cfg.min_samples} R={cfg.n_repeats} d<={cfg.max_d}"
        if len(per_split) < N_SPLITS * cfg.n_repeats:
            print(f"{label}: {len(per_split)} of {N_SPLITS * cfg.n_repeats} splits done")
            continue
        new = rows_for(tag, cfg, per_split, seconds)
        rows.extend(new)
        print(f"{label} ({seconds:.1f} s of computation)")
        for mth in METHODS:
            line = "  ".join(
                f"B{r['budget']}:{100 * float(r['coverage']):5.1f}({_pct(r['purity'])})"  # type: ignore[arg-type]
                for r in new if r["method"] == mth
            )
            print(f"  {mth:11s} {line}")
        sys.stdout.flush()
    if missing:
        print(f"{missing} split(s) left: run the same command again")
        return 0
    name = f"synthetic_{args.synthetic}" if args.synthetic else args.only
    out = RESULTS / f"compare_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
