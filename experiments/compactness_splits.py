"""
Compactness of the selected description on EVERY split.

`compactness.py` counts the groups and the distinct columns of a description
on fold 0 only, from `comparison_summary.csv`. The per-split cache files keep
the group labels of every method at every budget, so nothing has to be
recomputed to count them on all splits. This script reads the cache of one
configuration set and reports, per configuration, method and budget, the
means over the splits of

* held-out coverage and conditions spent (as the comparison reports them);
* the number of groups;
* the number of DISTINCT COLUMNS the description names;
* the number of GRIDS the description needs: the smallest number of schemas
  of at most `max_d` columns such that the column set of every group lies
  inside one of them. A grid method needs one by construction (checked);
  for free rules it is the number of separate pictures a reader would need
  to see the same groups the way the lattice shows them.

The grid count is an exact minimum (iterative deepening over the number of
schemas, with the bound ceil(#columns / max_d) from below and a greedy cover
from above); an instance that exhausts NODE_LIMIT search nodes keeps the
greedy value and is counted in `inexact`.

Methods without column labels (tree, SSD++, WRAcc lists, the optimal tree)
are not reported.

    python experiments/compactness_splits.py                         # the paper's configurations
    python experiments/compactness_splits.py --config-set grid
    python experiments/compactness_splits.py --config-set large_real_certified

Writes experiments/results/compactness_splits_<set>.csv and, for --tex,
paper/tables/compactness_splits_B<B>.tex.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS.parent))

from compare_baselines import BUDGETS, CACHE_EXTRA, CONFIG_SETS, N_SPLITS, RESULTS, Config, _cache_path, configs_of  # noqa: E402
from datasets import load_dataset  # noqa: E402

#: Methods whose group labels name columns (`X_<j>` tokens).
LABELLED: Tuple[str, ...] = ("vsf", "vsf_partial", "vsf_greedy", "rules_disjoint", "rules", "rules_pure")
#: Methods that select inside one schema: their description must need exactly one grid.
ONE_GRID: Tuple[str, ...] = ("vsf", "vsf_partial")
#: Methods that decide whether a configuration is informative at a budget (as in compactness.py and pairwise_partial.py).
INFORMATIVE_BY: Tuple[str, ...] = ("vsf", "vsf_partial", "vsf_greedy", "rules_disjoint", "rules", "tree", "ssdpp")
METHOD_LABEL: Dict[str, str] = {
    "vsf": "full grid", "vsf_partial": "partial centres", "vsf_greedy": "greedy chain",
    "rules_disjoint": r"rules$_{\mathrm{disj}}$", "rules": "rules", "rules_pure": r"rules$_{\mathrm{pure}}$",
}
NODE_LIMIT: int = 200_000
_COLUMN = re.compile(r"X_(\d+)")

ColumnSet = FrozenSet[int]


def group_columns(label: str) -> ColumnSet:
    """The columns one group label names: 'X_1+X_2#0' (a grid cell) or 'X_1=0 & X_2=0' (a rule)."""
    columns = frozenset(int(m) for m in _COLUMN.findall(label))
    if not columns:
        raise ValueError(f"group label {label!r} names no column")
    return columns


def maximal_sets(sets: Iterable[ColumnSet]) -> List[ColumnSet]:
    """The distinct sets that are not contained in another one, in a fixed order."""
    distinct = sorted(set(sets), key=lambda s: (-len(s), sorted(s)))
    kept: List[ColumnSet] = []
    for s in distinct:
        if not any(s <= t for t in kept):
            kept.append(s)
    return kept


def _greedy_grids(sets: Sequence[ColumnSet], max_d: int) -> int:
    """An upper bound: first-fit over the sets, largest first, into schemas of at most max_d columns."""
    schemas: List[Set[int]] = []
    for s in sets:
        best: Optional[Set[int]] = None
        for schema in schemas:
            if len(schema | s) <= max_d and (best is None or len(schema | s) - len(schema) < len(best | s) - len(best)):
                best = schema
        if best is None:
            schemas.append(set(s))
        else:
            best |= s
    return len(schemas)


def min_grids(sets: Iterable[ColumnSet], max_d: int) -> Tuple[int, bool]:
    """
    (the smallest number of schemas of <= max_d columns that hold every set, whether that number is proved minimal).
    Zero sets need zero grids. Raises when a set is larger than max_d.
    """
    family = maximal_sets(sets)
    if not family:
        return 0, True
    if any(len(s) > max_d for s in family):
        raise ValueError(f"a group names more than {max_d} columns: {sorted(max(family, key=len))}")
    n_columns = len(frozenset().union(*family))
    lower = -(-n_columns // max_d)
    upper = _greedy_grids(family, max_d)
    if lower == upper:
        return upper, True
    nodes = 0

    def fits(k: int, index: int, schemas: List[Set[int]]) -> Optional[bool]:
        """Can family[index:] be placed into at most k schemas given the ones opened so far? None: node limit."""
        nonlocal nodes
        if index == len(family):
            return True
        nodes += 1
        if nodes > NODE_LIMIT:
            return None
        s = family[index]
        seen: Set[FrozenSet[int]] = set()
        for schema in schemas:
            merged = schema | s
            if len(merged) > max_d:
                continue
            key = frozenset(merged)
            if key in seen:
                continue
            seen.add(key)
            added = s - schema
            schema |= added
            result = fits(k, index + 1, schemas)
            schema -= added
            if result is None or result:
                return result
        if len(schemas) < k:
            schemas.append(set(s))
            result = fits(k, index + 1, schemas)
            schemas.pop()
            if result is None or result:
                return result
        return False

    for k in range(lower, upper):
        outcome = fits(k, 0, [])
        if outcome is None:
            return upper, False
        if outcome:
            return k, True
    return upper, True


def load_splits(cfg: Config, X: object, Z: object) -> Optional[List[Dict[str, Dict[str, Dict[str, object]]]]]:
    """The cached methods of every split of a configuration (main file plus extra methods), or None when incomplete."""
    splits: List[Dict[str, Dict[str, Dict[str, object]]]] = []
    for i in range(N_SPLITS * cfg.n_repeats):
        path = _cache_path(cfg.dataset, cfg, X, Z, 0, i)  # type: ignore[arg-type]
        if not path.exists():
            return None
        methods = json.loads(path.read_text(encoding="utf-8"))["methods"]
        extra = CACHE_EXTRA / path.name
        if extra.exists():
            for name, value in json.loads(extra.read_text(encoding="utf-8")).get("methods", {}).items():
                methods.setdefault(name, value)
        for per_budget in methods.values():
            for outcome in per_budget.values():
                outcome.pop("inside", None)  # the packed membership mask is not needed here
        splits.append(methods)
    return splits


def config_label(cfg: Config) -> str:
    return f"{cfg.dataset}|{cfg.positive}|tau={cfg.tau}|m={cfg.min_samples}"


def describe(cfg: Config, splits: Sequence[Dict[str, Dict[str, Dict[str, object]]]], budget: int) -> List[Dict[str, object]]:
    """One row per labelled method present on every split, or no rows when the configuration is uninformative at the budget."""
    key = str(budget)
    coverage: Dict[str, float] = {}
    for method in INFORMATIVE_BY:
        if all(method in s for s in splits):
            coverage[method] = sum(int(s[method][key]["k_in"]) / max(1, int(s[method][key]["n_pos"])) for s in splits) / len(splits)
    if not coverage or max(coverage.values()) < 0.01:
        return []
    rows: List[Dict[str, object]] = []
    for method in LABELLED:
        if not all(method in s for s in splits):
            continue
        groups: List[int] = []
        columns: List[int] = []
        grids: List[int] = []
        conditions: List[int] = []
        covered: List[float] = []
        inexact = 0
        for s in splits:
            outcome = s[method][key]
            sets = [group_columns(str(label)) for label in outcome["labels"]]  # type: ignore[union-attr]
            n_grids, exact = min_grids(sets, cfg.max_d)
            if method in ONE_GRID and n_grids > 1:
                raise AssertionError(f"{config_label(cfg)} {method} B={budget}: a one-grid method needs {n_grids} grids")
            inexact += 0 if exact else 1
            groups.append(len(sets))
            columns.append(len(frozenset().union(*sets)) if sets else 0)
            grids.append(n_grids)
            conditions.append(int(outcome["conditions"]))  # type: ignore[arg-type]
            covered.append(int(outcome["k_in"]) / max(1, int(outcome["n_pos"])))  # type: ignore[arg-type]
        n = len(splits)
        rows.append({
            "dataset": cfg.dataset, "positive": cfg.positive, "tau": cfg.tau, "m": cfg.min_samples,
            "selection": cfg.selection, "budget": budget, "method": method, "n_splits": n,
            "coverage": sum(covered) / n, "conditions": sum(conditions) / n, "groups": sum(groups) / n,
            "columns": sum(columns) / n, "grids": sum(grids) / n, "max_grids": max(grids),
            "share_one_grid": sum(g <= 1 for g in grids) / n, "inexact": inexact,
        })
    return rows


def summary_table(rows: Sequence[Dict[str, object]], budget: int, methods: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Means over the configurations on which every method of `methods` has a row (paired), per selection rule."""
    by_config: Dict[Tuple[str, str], Dict[str, Dict[str, object]]] = defaultdict(dict)
    for r in rows:
        if int(r["budget"]) == budget:  # type: ignore[arg-type]
            by_config[(str(r["selection"]), f"{r['dataset']}|{r['positive']}|{r['tau']}|{r['m']}")][str(r["method"])] = r
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for selection in sorted({k[0] for k in by_config}):
        paired = [v for k, v in by_config.items() if k[0] == selection and all(m in v for m in methods)]
        if not paired:
            continue
        table[selection] = {}
        for method in methods:
            cell = {"n": float(len(paired))}
            for field in ("coverage", "conditions", "groups", "columns", "grids", "share_one_grid"):
                cell[field] = sum(float(v[method][field]) for v in paired) / len(paired)  # type: ignore[arg-type]
            cell["max_grids"] = float(max(int(v[method]["max_grids"]) for v in paired))  # type: ignore[arg-type]
            table[selection][method] = cell
    return table


def to_latex(table: Dict[str, Dict[str, Dict[str, float]]], methods: Sequence[str]) -> str:
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"selection & method & configs & coverage & conditions & groups & columns & grids\\", r"\midrule"]
    for selection in ("purity", "certified", "split"):
        if selection not in table:
            continue
        for method in methods:
            c = table[selection][method]
            lines.append(f"{selection} & {METHOD_LABEL[method]} & {int(c['n'])} & {c['coverage']:.2f} & {c['conditions']:.1f} & "
                         f"{c['groups']:.1f} & {c['columns']:.1f} & {c['grids']:.2f}\\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-set", choices=CONFIG_SETS, default="paper")
    ap.add_argument("--budget", type=int, default=32, help="budget of the printed summary and of --tex (the csv holds every budget)")
    ap.add_argument("--methods", default="vsf,vsf_partial,rules_disjoint,rules",
                    help="methods of the paired summary, comma-separated (default: vsf,vsf_partial,rules_disjoint,rules)")
    ap.add_argument("--only", default=None, help="comma-separated dataset names")
    ap.add_argument("--tex", action="store_true", help="also write paper/tables/compactness_splits_B<budget>.tex")
    args = ap.parse_args(argv)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = [m for m in methods if m not in LABELLED]
    if unknown:
        ap.error(f"methods without column labels: {unknown}")
    only = set(args.only.split(",")) if args.only else None
    frames: Dict[str, Tuple[object, object]] = {}
    rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    for cfg in configs_of(args.config_set):
        if only is not None and cfg.dataset not in only:
            continue
        if cfg.dataset not in frames:
            frame, entry = load_dataset(cfg.dataset)
            features = [c for c in frame.columns if c != entry.target]
            frames[cfg.dataset] = (frame[features].values, frame[entry.target].values)
        X, Z = frames[cfg.dataset]
        splits = load_splits(cfg, X, Z)
        if splits is None:
            skipped.append(f"{config_label(cfg)} {cfg.selection}")
            continue
        for budget in BUDGETS:
            rows.extend(describe(cfg, splits, int(budget)))
        print(f"read {config_label(cfg)} {cfg.selection}", flush=True)
    for name in skipped:
        print(f"incomplete, skipped: {name}")
    if not rows:
        print("no informative configuration")
        return 1
    out = RESULTS / f"compactness_splits_{args.config_set}.csv"
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {out} ({len(rows)} rows)")
    inexact = sum(int(r["inexact"]) for r in rows)  # type: ignore[arg-type]
    print(f"grid counts not proved minimal: {inexact} split(s)")
    table = summary_table(rows, args.budget, methods)
    print(f"\npaired means at B = {args.budget} (configurations on which every listed method has a row)")
    print(f"{'selection':10s} {'method':15s} {'cfgs':>4s} {'cover':>6s} {'cond':>6s} {'groups':>6s} {'cols':>6s} {'grids':>6s} {'1 grid':>7s} {'max':>4s}")
    for selection, per_method in table.items():
        for method in methods:
            c = per_method[method]
            print(f"{selection:10s} {method:15s} {int(c['n']):4d} {c['coverage']:6.3f} {c['conditions']:6.1f} {c['groups']:6.1f} "
                  f"{c['columns']:6.2f} {c['grids']:6.2f} {c['share_one_grid']:7.2f} {int(c['max_grids']):4d}")
    if args.tex:
        tex = EXPERIMENTS.parent / "paper" / "tables" / f"compactness_splits_B{args.budget}.tex"
        tex.write_text(to_latex(table, methods), encoding="utf-8", newline="\n")
        print(f"wrote {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
