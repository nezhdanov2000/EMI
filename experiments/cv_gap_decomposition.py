"""
Why nested CV can exceed fixed-schema CV: a decomposition.

Hypothesis (PMD Section 4.14, item 3): the full-data winner is chosen for
centres whose full-data purity only just clears tau. On a training fold
such cells drop below tau and stop being selected, so fixed-schema CV loses
their held-out positives. A fold's own winner is chosen on the training
rows, so its selected cells clear tau there by construction.

Per d this reports, averaged over the folds:
  * edge share: share of the positives inside selected cells whose purity
    (on the rows they were selected on) lies in [tau, tau + width) - for the
    full-data winner on all rows, and for each fold winner on its training
    rows;
  * held-out coverage of the full-data winner with its FULL-DATA centres
    (leaks the test fold; an upper reference), with its training-fold
    centres (= fixed-schema CV), and the part of the difference carried by
    cells that are centres on all rows but not on the training fold.

Usage:
    python experiments/cv_gap_decomposition.py --dataset titanic --positive survived --tau 0.9
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset  # noqa: E402
from vsf.avr import _exhaustive_search, _prepare_search  # noqa: E402
from vsf.centers import CenterSpec, _cell_counts, select_centers, stratified_repeated_kfold  # noqa: E402


def edge_share(k: np.ndarray, n: np.ndarray, mask: np.ndarray, tau: float, width: float) -> float:
    sel_pos = int(k[mask].sum())
    if sel_pos == 0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    edge = mask & (purity < tau + width)
    return int(k[edge].sum()) / sel_pos


def _mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--positive", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--max-d", type=int, default=4)
    ap.add_argument("--width", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    frame, entry = load_dataset(args.dataset)
    feats = [c for c in frame.columns if c != entry.target]
    spec = CenterSpec(tau=args.tau, min_samples=args.min_samples)
    prepared = _prepare_search(frame[feats].values, frame[entry.target].values, feats,
                               args.positive, spec, "presence")
    assert prepared is not None
    factory, z, names = prepared
    z64 = z.astype(np.int64)
    max_d = min(args.max_d, factory.n_features)
    full = _exhaustive_search(factory, z64, 2, [1], spec, max_d)[0]
    splits = stratified_repeated_kfold(z, 5, args.repeats, args.seed)

    acc: Dict[int, Dict[str, List[float]]] = {
        d: {k: [] for k in ("edge_full", "edge_fold", "cov_leak", "cov_fixed", "cov_nested", "lost_edge")}
        for d in full
    }
    full_codes = {d: factory.codes(tuple(c)) for d, (_, c) in full.items()}
    full_masks = {}
    for d, (codes, n_cells) in full_codes.items():
        k, n = _cell_counts(z, codes, n_cells)
        mask, _ = select_centers(k, n, spec)
        full_masks[d] = mask
        acc[d]["edge_full"].append(edge_share(k, n, mask, args.tau, args.width))

    for train, test in splits:
        best = _exhaustive_search(factory, z64, 2, [1], spec, max_d, rows=train)[0]
        for d in full:
            codes, n_cells = full_codes[d]
            k_tr, n_tr = _cell_counts(z[train], codes[train], n_cells)
            m_tr, _ = select_centers(k_tr, n_tr, spec)
            k_te, _ = _cell_counts(z[test], codes[test], n_cells)
            pos = max(1, int(k_te.sum()))
            acc[d]["cov_leak"].append(int(k_te[full_masks[d]].sum()) / pos)
            acc[d]["cov_fixed"].append(int(k_te[m_tr].sum()) / pos)
            acc[d]["lost_edge"].append(int(k_te[full_masks[d] & ~m_tr].sum()) / pos)
            fcodes, fcells = factory.codes(tuple(best[d][1]))
            fk, fn = _cell_counts(z[train], fcodes[train], fcells)
            fm, _ = select_centers(fk, fn, spec)
            acc[d]["edge_fold"].append(edge_share(fk, fn, fm, args.tau, args.width))
            tk, _ = _cell_counts(z[test], fcodes[test], fcells)
            acc[d]["cov_nested"].append(int(tk[fm].sum()) / max(1, int(tk.sum())))

    rows = []
    print(f"{args.dataset} {args.positive} tau={args.tau} m={args.min_samples} "
          f"edge width {args.width}, {len(splits)} folds")
    for d in sorted(acc):
        a = {k: _mean(v) for k, v in acc[d].items()}
        row = {"dataset": args.dataset, "positive": args.positive, "tau": args.tau,
               "min_samples": args.min_samples, "d": d,
               "winner": "+".join(names[j] for j in full[d][1]),
               **{k: round(v, 4) for k, v in a.items()}}
        rows.append(row)
        print(f"  d={d} edge(full winner)={a['edge_full']:.2f} edge(fold winners)={a['edge_fold']:.2f} | "
              f"held-out: full-data centres {a['cov_leak']:.3f}, fixed CV {a['cov_fixed']:.3f} "
              f"(lost in dropped cells {a['lost_edge']:.3f}), nested {a['cov_nested']:.3f}")
    out = args.out or (Path(__file__).resolve().parent / "results"
                       / f"cv_gap_{args.dataset}_{args.positive}_tau{args.tau}_m{args.min_samples}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
