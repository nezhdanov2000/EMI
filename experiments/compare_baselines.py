"""
Phase 3.2: VSF against other descriptions at equal condition budgets.

Methods, budgets and the evaluation protocol are defined in
`experiments/baselines.py`. For each configuration this writes one row per
(method, budget) with, over the repeated 5-fold splits:
- held-out coverage and net coverage (Nadeau-Bengio mean and se) and their
  paired differences to `vsf` on the same folds;
- mean held-out purity and the share of splits whose held-out purity
  reaches tau;
- description stability (pairwise Jaccard of the selected row sets);
- conditions actually spent, training coverage, mean selection seconds.

Synthetic configurations (`--synthetic`):
  grid       positives at (x0=0, x1=0) and (x0=1, x1=1): one 2-D grid holds both
  two_rules  positives at (x0=0, x1=0) and (x2=0, x3=0): no common 2-D grid
  xor        positive iff x0 xor x1 (binary), other columns noise

Every split is cached in results/cache/ as soon as it is computed, keyed by
the data's hash and the configuration; a call runs at most `--max-seconds`
of new splits and writes the summary once every split of every
configuration is present. Large datasets are therefore finished by
repeating the same command. Methods listed in `baselines.EXTRA_METHODS`
(added after the main run) are not part of the cache key: a cached split
that lacks one is topped up - only the missing methods run - and the
result goes to results/cache_extra/<same file name>, merged on load, so the
main cache files are never rewritten and extra methods computed on another
machine can be dropped in as files.

Selection rules: `purity` and `certified` (the family certificate) as in
the paper's main tables, and `split` (Webb's exploratory/holdout split;
`baselines.run_split`), listed for the same configurations as `certified`
so that the two certificates are compared on the same folds. `--selection`
restricts a run to one rule.

Usage:
    python experiments/compare_baselines.py --only titanic
    python experiments/compare_baselines.py --only breast_cancer_wisconsin --max-seconds 150   # repeat until done
    python experiments/compare_baselines.py --synthetic two_rules
    python experiments/compare_baselines.py --synthetic two_rules --n 20000
Results: results/compare_<name>.csv.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import (ALL_METHODS, BUDGETS, METHODS, SplitOutcome, SplitResult, run_split,  # noqa: E402
                       summarize_splits, unavailable_methods)
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
    selection: str = "purity"  # baselines.Selection


CONFIGS: List[Config] = [
    Config("titanic", "survived", 0.9),
    Config("titanic", "survived", 0.7),
    Config("titanic", "survived", 0.9, min_samples=20),
    Config("titanic", "survived", 0.7, selection="certified"),
    Config("titanic", "survived", 0.7, selection="split"),
    Config("breast_cancer", "recurrence-events", 0.7),
    Config("breast_cancer", "recurrence-events", 0.7, selection="certified"),
    Config("breast_cancer", "recurrence-events", 0.7, selection="split"),
    Config("thyroid_recurrence", "Yes", 0.9),
    Config("thyroid_recurrence", "Yes", 0.9, selection="certified"),
    Config("thyroid_recurrence", "Yes", 0.9, selection="split"),
    Config("breast_cancer_wisconsin", "malignant", 0.9),
    Config("breast_cancer_wisconsin", "malignant", 0.9, selection="certified"),
    Config("breast_cancer_wisconsin", "malignant", 0.9, selection="split"),
    # 2026-09-24: the four UCI files of the phase-3 runs restored from git
    # history (cead890) so that the partial-centre comparison is on the same
    # data as claude/phase3-baselines.md.
    Config("car_evaluation", "acc", 0.9),
    Config("car_evaluation", "acc", 0.9, selection="certified"),
    Config("car_evaluation", "acc", 0.9, selection="split"),
    Config("nursery", "spec_prior", 0.9),
    Config("nursery", "priority", 0.7),
    Config("nursery", "spec_prior", 0.9, selection="certified"),
    Config("nursery", "spec_prior", 0.9, selection="split"),
    Config("nursery", "priority", 0.7, selection="certified"),
    Config("nursery", "priority", 0.7, selection="split"),
    Config("mushroom", "poisonous", 0.9, n_repeats=2),
    Config("mushroom", "poisonous", 0.9, n_repeats=2, selection="certified"),
    Config("mushroom", "poisonous", 0.9, n_repeats=2, selection="split"),
    Config("chess_krkp", "won", 0.9, n_repeats=3),
    Config("chess_krkp", "won", 0.9, n_repeats=3, selection="certified"),
    Config("chess_krkp", "won", 0.9, n_repeats=3, selection="split"),
    # 2026-09-24: seven real tables built by experiments/build_external_benchmarks.py
    # (N 1 470 - 48 842, M 11 - 24, base rates 8 - 28 %). m = 20 throughout:
    # m = 1 lets free rules overfit tiny training-pure cells (PMD 4.16).
    Config("adult", ">50K", 0.7, min_samples=20, n_repeats=2),
    Config("adult", ">50K", 0.7, min_samples=20, n_repeats=2, selection="certified"),
    Config("adult", ">50K", 0.7, min_samples=20, n_repeats=2, selection="split"),
    Config("hmda", "yes", 0.5, min_samples=20),
    Config("hmda", "yes", 0.5, min_samples=20, selection="certified"),
    Config("hmda", "yes", 0.5, min_samples=20, selection="split"),
    Config("attrition", "Yes", 0.5, min_samples=20),
    Config("attrition", "Yes", 0.5, min_samples=20, selection="certified"),
    Config("attrition", "Yes", 0.5, min_samples=20, selection="split"),
    Config("credit", "bad", 0.7, min_samples=20),
    Config("credit", "bad", 0.7, min_samples=20, selection="certified"),
    Config("credit", "bad", 0.7, min_samples=20, selection="split"),
    Config("resume", "yes", 0.3, min_samples=20),
    Config("resume", "yes", 0.3, min_samples=20, selection="certified"),
    Config("resume", "yes", 0.3, min_samples=20, selection="split"),
    Config("wa_churn", "Yes", 0.7, min_samples=20),
    Config("wa_churn", "Yes", 0.7, min_samples=20, selection="certified"),
    Config("wa_churn", "Yes", 0.7, min_samples=20, selection="split"),
    Config("mlc_churn", "yes", 0.7, min_samples=20),
    Config("mlc_churn", "yes", 0.7, min_samples=20, selection="certified"),
    Config("mlc_churn", "yes", 0.7, min_samples=20, selection="split"),
    # 2026-09-24: the two wide PMLB tables (build_pmlb_benchmarks.py). splice has
    # M = 60: exhaustive families of size 4 are too large for every method, so
    # max_d = 3 for all methods alike (schema size, rule length, tree depth).
    Config("connect_4", "loss", 0.7, min_samples=20, n_repeats=1),
    Config("connect_4", "loss", 0.7, min_samples=20, n_repeats=1, selection="certified"),
    Config("connect_4", "loss", 0.7, min_samples=20, n_repeats=1, selection="split"),
    Config("connect_4", "win", 0.9, min_samples=20, n_repeats=1),
    Config("connect_4", "win", 0.9, min_samples=20, n_repeats=1, selection="certified"),
    Config("connect_4", "win", 0.9, min_samples=20, n_repeats=1, selection="split"),
    Config("splice", "site_b", 0.9, min_samples=20, n_repeats=3, max_d=3),
    Config("splice", "site_b", 0.9, min_samples=20, n_repeats=3, max_d=3, selection="certified"),
    Config("splice", "site_b", 0.9, min_samples=20, n_repeats=3, max_d=3, selection="split"),
]
# 2026-09-28: every certified configuration is also listed with selection="split"
# (Webb's exploratory/holdout split): the comparison of the two certificates
# in the paper's Section 4 on the same folds.
# The high-dimensional files (lung_discrete, colon, leukemia) are not listed:
# exhaustive search over d <= 4 on 325-7070 columns needs a feature pre-filter.


#: The fixed grid of the robustness check: every (dataset, target value) of the
#: paper's configurations at the same purity floors and size floors, whatever
#: the floor the paper's own configuration uses.
GRID_TAUS: Tuple[float, ...] = (0.7, 0.9)
GRID_MIN_SAMPLES: Tuple[int, ...] = (1, 20)


def grid_configs() -> List[Config]:
    """
    The fixed (tau, m) grid under the observed-purity rule: for every
    (dataset, target value) listed in CONFIGS, tau in GRID_TAUS x m in
    GRID_MIN_SAMPLES, with the repeats and `max_d` of the paper's own
    configuration of that dataset. A grid configuration that coincides with
    a paper configuration has the same cache key, so its splits are reused.
    """
    seen: Dict[Tuple[str, str], Config] = {}
    for c in CONFIGS:
        seen.setdefault((c.dataset, c.positive), c)
    out: List[Config] = []
    for (dataset, positive), ref in seen.items():
        for tau in GRID_TAUS:
            for m in GRID_MIN_SAMPLES:
                out.append(Config(dataset, positive, tau, min_samples=m, n_repeats=ref.n_repeats,
                                  max_d=ref.max_d, selection="purity"))
    return out


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
#: Methods of `baselines.EXTRA_METHODS`, per split, same file name as in CACHE.
CACHE_EXTRA = RESULTS / "cache_extra"
#: Bump when a method's selection or the cached fields change: old splits are then recomputed.
CACHE_VERSION = 5


def _cache_path(tag: str, cfg: Config, X: np.ndarray, Z: np.ndarray, seed: int, split: int) -> Path:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(X, dtype=str)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(Z, dtype=str)).tobytes())
    h.update(json.dumps([CACHE_VERSION, cfg.__dict__, seed, list(BUDGETS), list(METHODS), N_SPLITS]).encode())
    return CACHE / f"{tag}_{h.hexdigest()[:16]}_split{split:03d}.json"


def _pack(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(bool)).tobytes()).decode("ascii")


def _unpack(text: str, n_rows: int) -> np.ndarray:
    bits = np.frombuffer(base64.b64decode(text), dtype=np.uint8)
    return np.unpackbits(bits, count=n_rows).astype(bool)


def _save(path: Path, result: SplitResult, seconds: float) -> None:
    if not result:
        raise ValueError("nothing to save")
    n_rows = int(next(iter(next(iter(result.values())).values())).inside.size)
    payload = {
        "version": CACHE_VERSION,
        "seconds": seconds,
        "n_rows": n_rows,
        "methods": {
            mth: {str(b): {"k_in": o.k_in, "n_in": o.n_in, "n_pos": o.n_pos,
                           "conditions": o.conditions, "train_coverage": o.train_coverage,
                           "labels": list(o.labels), "inside": _pack(o.inside),
                           "seconds": o.seconds}
                  for b, o in per.items()}
            for mth, per in result.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    # Windows refuses the rename while another process has the target open
    # (a concurrent worker reading it); retry briefly, then fail loudly - a
    # split that was computed and not saved must never pass silently.
    last: Optional[Exception] = None
    for _attempt in range(10):
        try:
            tmp.replace(path)
            break
        except OSError as exc:
            last = exc
            time.sleep(0.5)
    else:
        raise RuntimeError(f"could not write {path} after 10 attempts") from last


def _load_one(path: Path) -> Tuple[SplitResult, float]:
    for attempt in range(10):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            break
        except Exception:
            time.sleep(0.5)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != CACHE_VERSION:
        raise ValueError(f"{path}: cache version {payload.get('version')} != {CACHE_VERSION}")
    n_rows = int(payload["n_rows"])
    result: SplitResult = {}
    for mth, per in payload["methods"].items():
        result[mth] = {
            int(b): SplitOutcome(
                k_in=int(o["k_in"]), n_in=int(o["n_in"]), n_pos=int(o["n_pos"]),
                conditions=int(o["conditions"]), train_coverage=float(o["train_coverage"]),
                labels=tuple(o["labels"]), inside=_unpack(o["inside"], n_rows),
                seconds=float(o["seconds"]),
            )
            for b, o in per.items()
        }
    return result, float(payload["seconds"])


def _load(path: Path) -> Tuple[SplitResult, float]:
    """The main cache file plus, when present, the extra-methods file of the same name."""
    result, seconds = _load_one(path)
    extra_path = CACHE_EXTRA / path.name
    if extra_path.exists():
        extra, extra_seconds = _load_one(extra_path)
        for mth, per in extra.items():
            result.setdefault(mth, per)
        seconds += extra_seconds
    return result, seconds


def _save_extra(path: Path, extra: SplitResult, seconds: float) -> None:
    """Merge `extra` into results/cache_extra/<path.name> (methods already there are kept)."""
    extra_path = CACHE_EXTRA / path.name
    merged: SplitResult = {}
    total = 0.0
    for attempt in range(10):
        if extra_path.exists():
            try:
                merged, total = _load_one(extra_path)
                break
            except Exception:
                time.sleep(0.5)
        else:
            break
    merged.update(extra)
    _save(extra_path, merged, total + seconds)


def _pct(value: object) -> str:
    return "   -" if value == "" else f"{100 * float(value):4.0f}"  # type: ignore[arg-type]


def _rounded(value: float, digits: int = 4) -> object:
    return "" if not np.isfinite(value) else round(float(value), digits)


def rows_for(tag: str, cfg: Config, per_split: List[SplitResult], seconds: float) -> List[Dict[str, object]]:
    summary = summarize_splits(per_split, N_SPLITS, cfg.n_repeats, cfg.tau)
    out: List[Dict[str, object]] = []
    for mth in ALL_METHODS:
        if mth not in summary:
            continue
        for b in BUDGETS:
            e = summary[mth][b]
            cv = e["coverage"]
            net = e["net_coverage"]
            pur = np.asarray(e["per_split_purity"], dtype=float)
            if mth != "vsf" and "vsf" in summary:
                g = paired_gain(cv, summary["vsf"][b]["coverage"])  # type: ignore[arg-type]
                gn = paired_gain(net, summary["vsf"][b]["net_coverage"])  # type: ignore[arg-type]
                diff, diff_se, ndiff, ndiff_se = g.difference, g.se, gn.difference, gn.se
            else:
                diff, diff_se, ndiff, ndiff_se = 0.0, 0.0, 0.0, 0.0
            out.append({
                "dataset": tag, "positive": cfg.positive, "tau": cfg.tau,
                "min_samples": cfg.min_samples, "n_repeats": cfg.n_repeats, "max_d": cfg.max_d,
                "selection": cfg.selection,
                "method": mth, "budget": b,
                "coverage": round(cv.mean, 4), "coverage_se": round(cv.se, 4),
                "purity": round(float(np.nanmean(pur)), 4) if np.any(np.isfinite(pur)) else "",
                "splits_with_groups": int(np.isfinite(pur).sum()),
                "purity_reaches_tau": _rounded(float(e["purity_reaches_tau"]), 3),  # type: ignore[arg-type]
                "net_coverage": round(net.mean, 4), "net_coverage_se": round(net.se, 4),
                "stability": _rounded(float(e["stability"]), 3),  # type: ignore[arg-type]
                "conditions_spent": round(float(e["cost_mean"]), 2),
                "train_coverage": round(float(e["train_coverage_mean"]), 4),
                "minus_vsf": round(diff, 4), "minus_vsf_se": round(diff_se, 4),
                "net_minus_vsf": round(ndiff, 4), "net_minus_vsf_se": round(ndiff_se, 4),
                "fold0_groups": " | ".join(per_split[0][mth][b].labels),
                "selection_seconds": round(float(e["seconds_mean"]), 3),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--only", help="bundled dataset name (all its configurations)")
    src.add_argument("--synthetic", choices=("grid", "two_rules", "xor"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000,
                    help="rows of a synthetic dataset (the certified selection needs a few thousand per cell)")
    ap.add_argument("--max-seconds", type=float, default=150.0,
                    help="stop starting new splits after this many seconds")
    ap.add_argument("--selection", choices=("purity", "certified", "split"), default=None,
                    help="compute new splits only for this selection rule (complete configurations of "
                         "the other rules are still summarised)")
    ap.add_argument("--methods", default=None,
                    help="comma-separated methods to have in every split (default: every method whose "
                         "package imports here; extra methods are topped up)")
    ap.add_argument("--reverse", action="store_true",
                    help="compute splits in reverse order to facilitate parallel workers")
    ap.add_argument("--splits", default=None,
                    help="comma-separated list of split indices to compute (e.g. 5,6,7)")
    ap.add_argument("--positive", default=None,
                    help="filter dataset configurations by positive target class")
    ap.add_argument("--config-set", choices=("paper", "grid"), default="paper",
                    help="paper: CONFIGS (the paper's tables); grid: the fixed (tau, m) grid of grid_configs(), "
                         "summarised in compare_grid_<name>.csv so the paper's files are never touched")
    ap.add_argument("--tau", type=float, default=None, help="filter dataset configurations by purity floor")
    ap.add_argument("--min-samples", type=int, default=None, help="filter dataset configurations by size floor")
    ap.add_argument("--no-csv", action="store_true",
                    help="compute and cache the requested splits only; exit 3 if any requested split is left "
                         "(used by run_jobs.py, whose jobs each cover a few splits of one configuration)")
    args = ap.parse_args()
    if args.methods is None:
        absent = unavailable_methods(ALL_METHODS)
        wanted = tuple(m for m in ALL_METHODS if m not in absent)
        for mth, pkg in absent.items():
            print(f"note: {mth} skipped, package {pkg!r} is not installed")
    else:
        wanted = tuple(m for m in args.methods.split(",") if m)
        unknown = set(wanted) - set(ALL_METHODS)
        if unknown:
            raise SystemExit(f"unknown methods: {sorted(unknown)}")
        absent = unavailable_methods(wanted)
        if absent:
            raise SystemExit("missing packages: " + ", ".join(f"{m} needs {p}" for m, p in absent.items()))

    if args.synthetic:
        X, Z = synthetic(args.synthetic, args.seed, args.n)
        base = f"synthetic_{args.synthetic}" + ("" if args.n == 2000 else f"_n{args.n}")
        jobs = [(base, Config(args.synthetic, "yes", 0.9, min_samples=5, selection=sel), X, Z)
                for sel in ("purity", "certified", "split")]
    else:
        source = CONFIGS if args.config_set == "paper" else grid_configs()
        configs = [c for c in source if c.dataset == args.only]
        if args.positive is not None:
            configs = [c for c in configs if c.positive == args.positive]
        if args.tau is not None:
            configs = [c for c in configs if abs(c.tau - args.tau) < 1e-9]
        if args.min_samples is not None:
            configs = [c for c in configs if c.min_samples == args.min_samples]
        if not configs:
            raise SystemExit(f"no configuration for {args.only!r}")
        frame, entry = load_dataset(args.only)
        feats = [c for c in frame.columns if c != entry.target]
        jobs = [(args.only, c, frame[feats].values, frame[entry.target].values) for c in configs]
    if args.selection is not None and not any(j[1].selection == args.selection for j in jobs):
        raise SystemExit(f"no configuration with selection {args.selection!r}")

    started = time.perf_counter()
    rows: List[Dict[str, object]] = []
    missing = 0
    for tag, cfg, X, Z in jobs:
        per_split: List[SplitResult] = []
        seconds = 0.0
        # --selection restricts what is COMPUTED; every complete configuration of
        # the dataset is still summarised, so compare_<name>.csv keeps all rows.
        may_compute = args.selection is None or cfg.selection == args.selection
        if args.splits is not None:
            splits_to_run = [int(s.strip()) for s in args.splits.split(",") if s.strip()]
        elif args.reverse:
            splits_to_run = list(reversed(range(N_SPLITS * cfg.n_repeats)))
        else:
            splits_to_run = list(range(N_SPLITS * cfg.n_repeats))
        for i in splits_to_run:
            path = _cache_path(tag, cfg, X, Z, args.seed, i)
            if path.exists():
                result, sec = _load(path)
                todo = [m for m in wanted if m not in result]
                if todo and may_compute and time.perf_counter() - started < args.max_seconds:
                    # top up: only the methods the cached split lacks are run
                    t0 = time.perf_counter()
                    extra = run_split(X, Z, cfg.positive, cfg.tau, cfg.min_samples, i, N_SPLITS,
                                      cfg.n_repeats, args.seed, BUDGETS, todo, cfg.max_d,
                                      cfg.selection)  # type: ignore[arg-type]
                    extra_sec = time.perf_counter() - t0
                    sec += extra_sec
                    result.update(extra)
                    _save_extra(path, extra, extra_sec)
                elif todo:
                    missing += 1
                    continue
            elif may_compute and time.perf_counter() - started < args.max_seconds:
                t0 = time.perf_counter()
                result = run_split(X, Z, cfg.positive, cfg.tau, cfg.min_samples, i, N_SPLITS,
                                   cfg.n_repeats, args.seed, BUDGETS, wanted, cfg.max_d,
                                   cfg.selection)  # type: ignore[arg-type]
                sec = time.perf_counter() - t0
                main = {mth: per for mth, per in result.items() if mth in METHODS}
                extra = {mth: per for mth, per in result.items() if mth not in METHODS}
                if main:
                    _save(path, main, sec)
                if extra:
                    _save_extra(path, extra, 0.0 if main else sec)
            else:
                missing += 1
                continue
        per_split = []
        seconds = 0.0
        for i in range(N_SPLITS * cfg.n_repeats):
            path = _cache_path(tag, cfg, X, Z, args.seed, i)
            if path.exists():
                res, sec = _load(path)
                if all(m in res for m in wanted):
                    per_split.append(res)
                    seconds += sec
        label = (f"{tag} {cfg.positive} tau={cfg.tau} m={cfg.min_samples} R={cfg.n_repeats}"
                 f" d<={cfg.max_d} {cfg.selection}")
        if len(per_split) < N_SPLITS * cfg.n_repeats:
            print(f"{label}: {len(per_split)} of {N_SPLITS * cfg.n_repeats} splits done")
            continue
        new = rows_for(tag, cfg, per_split, seconds)
        rows.extend(new)
        print(f"{label} ({seconds:.1f} s of computation)")
        for mth in ALL_METHODS:
            if not any(r["method"] == mth for r in new):
                continue
            line = "  ".join(
                f"B{r['budget']}:{100 * float(r['coverage']):5.1f}({_pct(r['purity'])}|{_pct(r['net_coverage'])})"  # type: ignore[arg-type]
                for r in new if r["method"] == mth
            )
            print(f"  {mth:11s} {line}")
        sys.stdout.flush()
    if args.no_csv:
        if missing:
            print(f"{missing} split(s) left: run the same command again")
            return 3
        print("requested splits are cached (no csv written: --no-csv)")
        return 0
    if missing:
        print(f"{missing} split(s) left: run the same command again")
        return 0
    if not args.synthetic and args.config_set == "grid" and (
            args.positive is not None or args.tau is not None or args.min_samples is not None or args.splits is not None):
        print("no compare csv written: a filtered grid run would hold only part of the dataset's rows")
        return 0
    if not rows:
        print("no complete configuration: nothing to write")
        return 0
    if "vsf" not in wanted:
        print("no compare csv written: --methods without vsf (extra methods only)")
        return 0
    name = jobs[0][0] if args.synthetic else args.only
    out = RESULTS / (f"compare_grid_{name}.csv" if (not args.synthetic and args.config_set == "grid") else f"compare_{name}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
