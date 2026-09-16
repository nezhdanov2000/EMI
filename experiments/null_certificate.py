"""
False-certificate rate of the REPORTED winner under a global null, for the
per-schema certificate the product uses and for the two post-selection
certificates of `vsf.selective`.

Every feature and the target are drawn independently, so no cell of any
schema has purity above the base rate. Under Rule C with tau above the base
rate, Proposition 2 promises that a PRE-SPECIFIED schema certifies at least
one centre with probability <= alpha. The search, however, reports the
argmax over all schemas of each dimensionality. This script measures how
often that reported winner carries at least one certified centre, per d, and
how often ANY reported branch does - for `discover_branches` with
`rule="certified"` ("per_schema") and for `certify_discovery` with
`"family_bonferroni"` and `"split"`, whose guarantee covers all reported
branches together.

Usage:
    python experiments/null_certificate.py --runs 100
    (writes experiments/results/null_certificate.csv)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsf.avr import discover_branches  # noqa: E402
from vsf.centers import CenterSpec  # noqa: E402
from vsf.selective import certify_discovery  # noqa: E402

METHODS = ("per_schema", "family_bonferroni", "split")


@dataclass(frozen=True)
class NullConfig:
    n_rows: int
    n_features: int
    n_levels: int
    prevalence: float
    tau: float
    alpha: float
    max_d: int


def one_run(cfg: NullConfig, seed: int, method: str) -> Dict[int, bool]:
    """Per dimensionality, whether the reported winner certifies >= 1 cell."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, cfg.n_levels, size=(cfg.n_rows, cfg.n_features)).astype(str)
    Z = np.where(rng.random(cfg.n_rows) < cfg.prevalence, "1", "0")
    if method == "per_schema":
        branches = discover_branches(
            X, Z, positive_class="1", max_d=cfg.max_d,
            center_spec=CenterSpec(tau=cfg.tau, rule="certified", alpha=cfg.alpha, min_samples=1),
            n_permutations_centers=0, cv_repeats=0, random_state=seed,
        )
        return {d: b.centers.n_centers > 0 for d, b in branches.items()}
    res = certify_discovery(
        X, Z, positive_class="1", max_d=cfg.max_d,
        center_spec=CenterSpec(tau=cfg.tau, alpha=cfg.alpha, min_samples=1),
        method=method, random_state=seed,  # type: ignore[arg-type]
    )
    return {d: b.n_certified > 0 for d, b in res.branches.items()}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--features", type=int, default=12)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--prevalence", type=float, default=0.45)
    ap.add_argument("--tau", type=float, default=0.50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-d", type=int, default=4)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "results" / "null_certificate.csv")
    args = ap.parse_args()

    if not args.tau > args.prevalence:
        raise SystemExit("tau must exceed the prevalence, otherwise the null is not a null")

    cfg = NullConfig(args.rows, args.features, args.levels, args.prevalence,
                     args.tau, args.alpha, args.max_d)
    rows: List[Dict[str, object]] = []
    for method in args.methods:
        hits: Dict[str, int] = {str(d): 0 for d in range(1, cfg.max_d + 1)}
        hits["any"] = 0
        t0 = time.perf_counter()
        for seed in range(args.runs):
            per_d = one_run(cfg, seed, method)
            for d, hit in per_d.items():
                hits[str(d)] += int(hit)
            hits["any"] += int(any(per_d.values()))
        elapsed = time.perf_counter() - t0
        print(f"{method}  ({elapsed:.1f} s)")
        for d, k in hits.items():
            lo, hi = wilson_interval(k, args.runs)
            rows.append({"method": method, "d": d, "runs": args.runs, "false_certificates": k,
                         "rate": k / args.runs, "ci95_low": lo, "ci95_high": hi,
                         "nominal": cfg.alpha, **cfg.__dict__})
            print(f"  d={d:>3}: {k}/{args.runs} = {k/args.runs:.2f}  "
                  f"[95% CI {lo:.2f}, {hi:.2f}]  nominal {cfg.alpha:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
