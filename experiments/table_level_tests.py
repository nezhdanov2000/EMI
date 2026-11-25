"""
Tests at the level of TABLES.

The paired marks of `pairwise_partial.py` are per (configuration, budget) cell:
they say where two methods differ on one table, with the dependence between
folds handled by the Nadeau-Bengio correction. They do not say whether one
method is better than another ACROSS tables, and the cells of one table
(several targets, floors, budgets) are not independent of each other. This
script answers the across-tables question the standard way (Demsar, 2006):

* the unit is the table: for every table, the mean over its informative
  configurations at one budget of each method's held-out coverage (or net
  coverage);
* for a pair of methods, the Wilcoxon signed-rank test on the per-table
  differences, two-sided and EXACT: zero differences are ranked and then
  dropped (Pratt), ties get mid-ranks, and the null distribution of the
  signed-rank sum is enumerated by dynamic programming over the actual
  ranks, so no normal approximation is used at 10-20 tables;
* Holm's step-down over the pairs of one (selection rule, metric) family;
* for the methods present on every table, the Friedman test with the mean
  ranks (rank 1 = best).

Input: the `pairwise_*.csv` of one or more configuration sets, pooled (a
table that occurs in two sets is averaged over all its configurations).
Only informative cells are in those files (some method above 1 % held-out
coverage), so a table with no informative configuration under a selection
rule is absent from that rule's tests; the count of tables is printed.

    python experiments/table_level_tests.py                                   # the paper's configurations
    python experiments/table_level_tests.py --config-set grid large_real
    python experiments/table_level_tests.py --config-set grid_certified large_real_certified

Writes experiments/results/table_level_tests_<sets>.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

EXPERIMENTS = Path(__file__).resolve().parent
RESULTS = EXPERIMENTS / "results"

PAIRWISE_FILE: Dict[str, str] = {
    "paper": "pairwise_partial.csv", "grid": "pairwise_grid.csv", "grid_certified": "pairwise_grid_certified.csv",
    "large_real": "pairwise_large_real.csv", "large_real_certified": "pairwise_large_real_certified.csv",
}
#: method -> suffix of its columns in the pairwise files (cov_<suffix>, net_<suffix>).
SUFFIX: Dict[str, str] = {
    "vsf": "vsf", "vsf_partial": "partial", "rules_disjoint": "rules_disjoint", "rules": "rules",
    "tree": "tree", "ssdpp": "ssdpp", "wracc": "wracc", "tree_opt": "tree_opt",
}
#: The pairs tested, fixed here and not chosen by their outcome: (first, second), difference = first - second.
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("vsf_partial", "vsf"), ("vsf_partial", "rules_disjoint"), ("vsf", "rules_disjoint"),
    ("vsf_partial", "rules"), ("vsf_partial", "tree"), ("vsf_partial", "tree_opt"),
    ("vsf_partial", "ssdpp"), ("vsf_partial", "wracc"),
)
#: Methods of the Friedman test (those of them present on every table are used).
FRIEDMAN_METHODS: Tuple[str, ...] = ("vsf", "vsf_partial", "rules_disjoint", "rules", "tree", "tree_opt", "ssdpp", "wracc")
METRICS: Tuple[str, ...] = ("cov", "net")
#: Differences smaller than this are zeros (the per-table means are sums of ratios of integers).
ZERO_TOLERANCE: float = 1e-12


def midranks(values: Sequence[float]) -> List[float]:
    """Ranks 1..n of the values in increasing order; equal values share the mean of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] - values[order[i]] <= ZERO_TOLERANCE:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def wilcoxon_exact(differences: Sequence[float]) -> Tuple[float, float, int, int, int]:
    """
    Two-sided exact Wilcoxon signed-rank test with Pratt's treatment of zeros.

    Returns (W+, p, positive, zero, negative): the sum of the ranks of the
    positive differences, the probability under the null (every non-zero
    difference keeps its rank and takes either sign with probability 1/2) of
    a rank sum at least as far from its mean as the observed one, and the
    counts of positive, zero and negative differences. With no non-zero
    difference the p-value is 1.
    """
    absolute = [abs(d) for d in differences]
    ranks = midranks(absolute)
    kept = [(r, d) for r, d in zip(ranks, differences) if abs(d) > ZERO_TOLERANCE]
    positive = sum(d > 0 for _, d in kept)
    negative = len(kept) - positive
    zero = len(differences) - len(kept)
    if not kept:
        return 0.0, 1.0, 0, zero, 0
    doubled = [int(round(2 * r)) for r, _ in kept]  # mid-ranks are multiples of 1/2
    if any(abs(2 * r - d2) > 1e-9 for (r, _), d2 in zip(kept, doubled)):
        raise ArithmeticError("a mid-rank is not a multiple of 1/2")
    total = sum(doubled)
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in doubled:
        for s in range(total, r - 1, -1):
            if counts[s - r]:
                counts[s] += counts[s - r]
    observed = sum(r for r, (_, d) in zip(doubled, kept) if d > 0)
    distance = abs(2 * observed - total)
    extreme = sum(c for s, c in enumerate(counts) if abs(2 * s - total) >= distance)
    return observed / 2.0, extreme / float(2 ** len(kept)), positive, zero, negative


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm's step-down adjusted p-values, in the order given."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for position, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - position) * p_values[i]))
        adjusted[i] = running
    return adjusted


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution (regularised upper incomplete gamma), standard library only."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    half = x / 2.0
    if half < a + 1.0:  # series for the lower tail
        term = 1.0 / a
        total = term
        n = a
        while abs(term) > abs(total) * 1e-15:
            n += 1.0
            term *= half / n
            total += term
        return max(0.0, 1.0 - total * math.exp(-half + a * math.log(half) - math.lgamma(a)))
    tiny = 1e-300  # continued fraction for the upper tail (modified Lentz)
    b = half + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    i = 1
    while True:
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
        i += 1
    return h * math.exp(-half + a * math.log(half) - math.lgamma(a))


def friedman(blocks: Sequence[Sequence[float]]) -> Tuple[float, float, List[float]]:
    """
    Friedman's test with the correction for ties: (statistic, p, mean ranks), rank 1 = the LARGEST value
    in a block. `blocks` are tables, each a sequence of one value per method.
    """
    n = len(blocks)
    k = len(blocks[0])
    sums = [0.0] * k
    tie_term = 0.0
    for block in blocks:
        ranks = midranks([-v for v in block])
        for j, r in enumerate(ranks):
            sums[j] += r
        seen: Dict[float, int] = defaultdict(int)
        for r in ranks:
            seen[r] += 1
        tie_term += sum(t ** 3 - t for t in seen.values())
    mean_ranks = [s / n for s in sums]
    denominator = n * k * (k + 1) / 12.0 - tie_term / (12.0 * (k - 1))
    if denominator <= 0:
        return 0.0, 1.0, mean_ranks
    statistic = sum((s - n * (k + 1) / 2.0) ** 2 for s in sums) / denominator
    return statistic, chi2_sf(statistic, k - 1), mean_ranks


def per_table_means(rows: Sequence[Dict[str, str]], selection: str, budget: int, metric: str) -> Dict[str, Dict[str, float]]:
    """table -> method -> mean of the metric over the table's informative configurations (methods missing on any of them are left out)."""
    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    n_rows: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r["selection"] != selection or int(r["budget"]) != budget:
            continue
        n_rows[r["dataset"]] += 1
        for method, suffix in SUFFIX.items():
            cell = r.get(f"{metric}_{suffix}", "")
            if cell not in ("", "nan", "NaN"):
                values[r["dataset"]][method].append(float(cell))
    return {table: {m: sum(v) / len(v) for m, v in per_method.items() if len(v) == n_rows[table]}
            for table, per_method in values.items()}


def analyse(rows: Sequence[Dict[str, str]], budget: int) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for selection in sorted({r["selection"] for r in rows}):
        for metric in METRICS:
            means = per_table_means(rows, selection, budget, metric)
            if not means:
                continue
            family: List[Dict[str, object]] = []
            for first, second in PAIRS:
                tables = sorted(t for t, m in means.items() if first in m and second in m)
                if len(tables) < 2:
                    continue
                differences = [means[t][first] - means[t][second] for t in tables]
                w_plus, p, positive, zero, negative = wilcoxon_exact(differences)
                ordered = sorted(differences)
                family.append({
                    "selection": selection, "metric": metric, "budget": budget, "test": "wilcoxon",
                    "first": first, "second": second, "tables": len(tables),
                    "first_better": positive, "equal": zero, "second_better": negative,
                    "mean_difference": sum(differences) / len(differences),
                    "median_difference": (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2.0,
                    "statistic": w_plus, "p": p, "p_holm": float("nan"), "mean_ranks": "",
                })
            for row, adjusted in zip(family, holm([float(r["p"]) for r in family])):  # type: ignore[arg-type]
                row["p_holm"] = adjusted
            out.extend(family)
            methods = [m for m in FRIEDMAN_METHODS if all(m in per for per in means.values())]
            if len(methods) >= 3 and len(means) >= 2:
                tables = sorted(means)
                statistic, p, mean_ranks = friedman([[means[t][m] for m in methods] for t in tables])
                out.append({
                    "selection": selection, "metric": metric, "budget": budget, "test": "friedman",
                    "first": "", "second": "", "tables": len(tables), "first_better": "", "equal": "", "second_better": "",
                    "mean_difference": "", "median_difference": "", "statistic": statistic, "p": p, "p_holm": "",
                    "mean_ranks": "; ".join(f"{m} {r:.2f}" for m, r in zip(methods, mean_ranks)),
                })
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-set", nargs="+", choices=sorted(PAIRWISE_FILE), default=["paper"],
                    help="one or more configuration sets, pooled (their pairwise_*.csv must exist)")
    ap.add_argument("--budget", type=int, default=32, help="condition budget of the comparison (default 32, the paper's summary budget)")
    args = ap.parse_args(argv)
    rows: List[Dict[str, str]] = []
    for name in args.config_set:
        path = RESULTS / PAIRWISE_FILE[name]
        if not path.is_file():
            print(f"{path} is missing: run `python experiments/pairwise_partial.py --config-set {name}` first", file=sys.stderr)
            return 1
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    result = analyse(rows, args.budget)
    if not result:
        print("no informative cell at this budget")
        return 1
    out = RESULTS / f"table_level_tests_{'+'.join(args.config_set)}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result[0].keys()), lineterminator="\n")
        writer.writeheader()
        for r in result:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"budget {args.budget}; unit = table; exact two-sided Wilcoxon signed-rank (Pratt), Holm within (selection, metric)")
    for selection in sorted({str(r["selection"]) for r in result}):
        for metric in METRICS:
            block = [r for r in result if r["selection"] == selection and r["metric"] == metric]
            if not block:
                continue
            print(f"\n== {selection}, {'coverage' if metric == 'cov' else 'net coverage'}")
            print(f"{'first':13s} {'second':15s} {'tables':>6s} {'+':>3s} {'=':>3s} {'-':>3s} {'mean diff':>10s} {'median':>8s} {'p':>8s} {'Holm':>8s}")
            for r in block:
                if r["test"] == "wilcoxon":
                    print(f"{r['first']:13s} {r['second']:15s} {r['tables']:6d} {r['first_better']:3d} {r['equal']:3d} {r['second_better']:3d} "
                          f"{r['mean_difference']:+10.4f} {r['median_difference']:+8.4f} {r['p']:8.4f} {r['p_holm']:8.4f}")
                else:
                    print(f"Friedman over {r['tables']} tables: chi2 = {r['statistic']:.2f}, p = {r['p']:.4g}; mean ranks (1 = best): {r['mean_ranks']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
