"""
Nested vs fixed-schema cross-validation, and schema stability.

For each configured (dataset, target value, tau) this reports, per d:
  * in-sample coverage of the full-data winner,
  * fixed-schema CV: the full-data winner, centres re-selected per fold
    (`vsf.centers.crossvalidated_coverage`, what the product shows),
  * nested CV: the whole search repeated on each training fold
    (`vsf.selective.nested_crossvalidation`), with partitions fitted on the
    training rows only (`encoding="train"`, the protocol), and the same with
    partitions built from all rows' features (`encoding="all_rows"`) to
    measure what the test rows' feature values change,
  * held-out pooled purity for both (coverage alone rewards selecting more
    cells; a CV coverage without its CV purity is not interpretable),
  * how often the fold winner equals the full-data winner.

Both CV variants use identical folds, so their per-split values are paired.

Usage:
    python experiments/nested_cv.py                 # all configurations
    python experiments/nested_cv.py --only titanic  # one dataset (results/nested_cv_titanic.csv)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset  # noqa: E402
from vsf.avr import discover_branches  # noqa: E402
from vsf.centers import CenterSpec, paired_gain  # noqa: E402
from vsf.selective import nested_crossvalidation  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class Config:
    dataset: str
    positive: str
    tau: float
    max_d: int = 4
    n_repeats: int = 5
    min_samples: int = 1


CONFIGS: List[Config] = [
    Config("titanic", "survived", 0.9),
    Config("titanic", "survived", 0.7),
    Config("titanic", "survived", 0.9, min_samples=20),
    Config("car_evaluation", "acc", 0.9),
    Config("nursery", "spec_prior", 0.9),
    Config("nursery", "priority", 0.7),
    Config("mushroom", "poisonous", 0.9),
    Config("chess_krkp", "won", 0.9, max_d=3),
    Config("splice_junction", "EI", 0.9, max_d=2),
]


def run(cfg: Config, seed: int) -> List[Dict[str, object]]:
    frame, entry = load_dataset(cfg.dataset)
    feats = [c for c in frame.columns if c != entry.target]
    X, Z = frame[feats].values, frame[entry.target].values
    spec = CenterSpec(tau=cfg.tau, min_samples=cfg.min_samples)
    t0 = time.perf_counter()
    nested = nested_crossvalidation(
        X, Z, feats, positive_class=cfg.positive, center_spec=spec,
        max_d=cfg.max_d, n_repeats=cfg.n_repeats, random_state=seed, encoding="train",
    )
    t_nested = time.perf_counter() - t0
    nested_all = nested_crossvalidation(
        X, Z, feats, positive_class=cfg.positive, center_spec=spec,
        max_d=cfg.max_d, n_repeats=cfg.n_repeats, random_state=seed, encoding="all_rows",
    )
    branches = discover_branches(
        X, Z, feats, positive_class=cfg.positive, center_spec=spec, max_d=cfg.max_d,
        n_permutations_centers=0, cv_repeats=0, random_state=seed,
    )
    rows: List[Dict[str, object]] = []
    if nested.undetermined_reason is not None:
        print(f"  {cfg}: {nested.undetermined_reason}")
        return rows
    d_nested = nested.select_dimensionality()
    for d, b in nested.branches.items():
        gain = paired_gain(b.nested, b.fixed_schema)
        ba = nested_all.branches[d]
        enc_gap = paired_gain(ba.nested, b.nested)
        st = b.stability
        row: Dict[str, object] = {
            "dataset": cfg.dataset, "positive": cfg.positive, "tau": cfg.tau,
            "min_samples": cfg.min_samples, "d": d, "n_repeats": cfg.n_repeats,
            "seed": seed, "winner": "+".join(b.full_data_winner_names),
            "in_sample_coverage": round(branches[d].centers.coverage, 4),
            "in_sample_purity": round(branches[d].centers.purity_pooled, 4),
            "fixed_cv_coverage": round(b.fixed_schema.mean, 4),
            "fixed_cv_se": round(b.fixed_schema.se, 4),
            "fixed_cv_purity": round(b.fixed_schema.purity_mean, 4),
            "nested_cv_coverage": round(b.nested.mean, 4),
            "nested_cv_se": round(b.nested.se, 4),
            "nested_cv_purity": round(b.nested.purity_mean, 4),
            "nested_minus_fixed_t": round(gain.t_statistic, 2),
            "nested_all_rows_coverage": round(ba.nested.mean, 4),
            "nested_all_rows_purity": round(ba.nested.purity_mean, 4),
            "all_rows_minus_train": round(enc_gap.difference, 4),
            "all_rows_minus_train_se": round(enc_gap.se, 4),
            "share_same_winner_all_rows": round(ba.stability.share_equal_to_reference, 3),
            "share_same_winner": round(st.share_equal_to_reference, 3),
            "n_distinct_winners": st.n_distinct,
            "mean_pairwise_jaccard": round(st.mean_pairwise_jaccard, 3),
            "d_star_nested": d_nested if d_nested is not None else "none",
            "seconds_nested": round(t_nested, 1),
        }
        rows.append(row)
        print(
            f"  d={d} {row['winner']:<42} in={row['in_sample_coverage']:.3f} "
            f"fixed={row['fixed_cv_coverage']:.3f}±{row['fixed_cv_se']:.3f} (pur {row['fixed_cv_purity']:.3f}) "
            f"nested={row['nested_cv_coverage']:.3f}±{row['nested_cv_se']:.3f} (pur {row['nested_cv_purity']:.3f}) "
            f"same={row['share_same_winner']:.2f} distinct={st.n_distinct} | "
            f"all-rows enc {row['nested_all_rows_coverage']:.3f} "
            f"(diff {row['all_rows_minus_train']:+.3f}±{row['all_rows_minus_train_se']:.3f})"
        )
    print(f"  d* (nested) = {d_nested}; {t_nested:.1f} s")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="run only configurations of this dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/nested_cv.csv, or results/nested_cv_<dataset>.csv with --only")
    args = ap.parse_args()
    if args.out is None:
        args.out = RESULTS / (f"nested_cv_{args.only}.csv" if args.only else "nested_cv.csv")
    configs = [c for c in CONFIGS if args.only is None or c.dataset == args.only]
    if not configs:
        raise SystemExit(f"no configuration for {args.only!r}")
    rows: List[Dict[str, object]] = []
    for cfg in configs:
        print(f"{cfg.dataset} {cfg.positive} tau={cfg.tau} m={cfg.min_samples} max_d={cfg.max_d} R={cfg.n_repeats}")
        rows.extend(run(cfg, args.seed))
        sys.stdout.flush()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
