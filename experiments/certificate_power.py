"""
Strong family-wise control and power of the certificates, on data with a
known answer.

Generating model (rows i.i.d., features independent and uniform on 3
levels): P(Z = 1 | x) = 0.95 in the cell x0 = 0, x1 = 0 (a true centre),
exactly tau = 0.80 in the cell x0 = 1, x1 = 1 (the hardest null: H_0 holds
with equality), 0.10 elsewhere. Remaining columns are noise.

Conditional on the features, a cell's target count is Poisson-binomial with
mean purity pi_c = mean of P(Z = 1 | x_i) over its rows; that is the
population purity the certificate is a claim about. A certified cell with
pi_c <= tau is a false certificate. Reported per method and N:
  * FWER - share of runs with at least one false certificate in any branch,
  * power - share of runs certifying at least one cell with pi_c >= 0.90,
  * mean certified coverage of the reported d = 2 branch.

A global-null experiment cannot show strong control: there every cell is
far below tau. The boundary cell here is what separates a certificate of
"purity > tau" from a test of independence.

Usage:
    python experiments/certificate_power.py --runs 200 --sizes 500
    (one size per call keeps each run short; results/certificate_power_n500.csv)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsf.avr import _prepare_search, discover_branches  # noqa: E402
from vsf.centers import CenterSpec, select_centers, _cell_counts  # noqa: E402
from vsf.selective import certify_discovery  # noqa: E402

METHODS = ("per_schema", "family_bonferroni", "split")
TAU = 0.80
P_GOOD, P_BOUNDARY, P_BACKGROUND = 0.95, TAU, 0.10


def generate(seed: int, n: int, m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, m))
    p = np.full(n, P_BACKGROUND)
    p[(X[:, 0] == 0) & (X[:, 1] == 0)] = P_GOOD
    p[(X[:, 0] == 1) & (X[:, 1] == 1)] = P_BOUNDARY
    Z = np.where(rng.random(n) < p, "1", "0")
    return X.astype(str), Z, p


def certified_row_sets(X: np.ndarray, Z: np.ndarray, method: str, seed: int, alpha: float) -> Dict[int, List[np.ndarray]]:
    """Rows of every certified cell, per reported dimensionality."""
    spec = CenterSpec(tau=TAU, alpha=alpha)
    prepared = _prepare_search(X, Z, None, "1", spec, "presence")
    assert prepared is not None
    factory, z, _ = prepared
    out: Dict[int, List[np.ndarray]] = {}
    if method == "per_schema":
        cspec = CenterSpec(tau=TAU, alpha=alpha, rule="certified")
        branches = discover_branches(X, Z, positive_class="1", center_spec=cspec,
                                     n_permutations_centers=0, cv_repeats=0)
        for d, b in branches.items():
            codes, n_cells = factory.codes(tuple(b.selected_features))
            k, n = _cell_counts(z, codes, n_cells)
            mask, _ = select_centers(k, n, cspec)
            out[d] = [np.nonzero(codes == c)[0] for c in np.nonzero(mask)[0]]
        return out
    res = certify_discovery(X, Z, positive_class="1", center_spec=spec,
                            method=method, random_state=seed)  # type: ignore[arg-type]
    for d, b in res.branches.items():
        codes, _ = factory.codes(b.features)
        out[d] = [np.nonzero(codes == c.cell)[0] for c in b.certified_cells]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 3000])
    ap.add_argument("--features", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/certificate_power_n<sizes>.csv")
    args = ap.parse_args()
    if args.out is None:
        tag = "_".join(str(v) for v in args.sizes)
        args.out = Path(__file__).resolve().parent / "results" / f"certificate_power_n{tag}.csv"

    rows: List[Dict[str, object]] = []
    for n in args.sizes:
        for method in args.methods:
            t0 = time.perf_counter()
            fwer = power = 0
            cov2: List[float] = []
            for seed in range(args.runs):
                X, Z, p = generate(seed, n, args.features)
                cells = certified_row_sets(X, Z, method, seed, args.alpha)
                purities = [float(p[r].mean()) for rs in cells.values() for r in rs]
                fwer += int(any(q <= TAU + 1e-12 for q in purities))
                power += int(any(q >= 0.90 for q in purities))
                pos = Z == "1"
                rows2 = cells.get(2, [])
                covered = np.zeros(n, dtype=bool)
                for r in rows2:
                    covered[r] = True
                cov2.append(float((covered & pos).sum() / pos.sum()))
            row = {
                "n": n, "method": method, "runs": args.runs, "tau": TAU, "alpha": args.alpha,
                "fwer": fwer / args.runs, "power": power / args.runs,
                "mean_coverage_d2": round(float(np.mean(cov2)), 4),
                "seconds": round(time.perf_counter() - t0, 1),
            }
            rows.append(row)
            print(f"N={n:5d} {method:18s} FWER={row['fwer']:.3f} power={row['power']:.3f} "
                  f"cov(d=2)={row['mean_coverage_d2']:.3f}  ({row['seconds']} s)")
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
