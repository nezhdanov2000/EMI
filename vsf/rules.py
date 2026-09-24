"""
Rules: the cells of the winning schemas read as conjunctive rules
(Project_Master_Document.md Section 4.15).

A cell of a d-axis schema is a conjunction of d conditions
``column_1 = value_1 ∧ ... ∧ column_d = value_d`` together with what the
data says about it: how many rows satisfy it (`n`), how many of them carry
the target value (`k`), the share `k / n` (the cell's purity) and whether the
certificate in force makes it a discrete centre. Listing every cell of the
winning 1D-4D schemas as such a rule, filtering the list by the conditions
it contains and binning it by purity, is the "Rules" view of the webapp.

Nothing here is a new estimate: the numbers are the displayed partition's
own (`vsf.vis._FullCellStatistics`, computed on all rows), the same ones the
lattice hover shows. What is added is each rule's *generalisations* - the
purity of the same conjunction with one condition dropped (its immediate
parents) and the most pure proper sub-conjunction overall - so that a card
can say "Male ∧ child 60 %, against Male 85 %": learning more about an
object changes the probability, and the reader should see by how much. No
rule is hidden on that account; a refinement that LOWERS the purity is the
informative case, not the redundant one.

Generalisations are computed from the data, not from other winning
schemas: the parent of the 2D cell `sex = male ∧ age = child` is the 1D
conjunction `sex = male` on the same rows, whether or not `sex` is the 1D
winner.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .centers import CenterSpec
from .vis import Translations, _FullCellStatistics, humanize_col, humanize_val

__all__ = ["enumerate_rules", "union_coverage"]


def enumerate_rules(
    X: np.ndarray,
    Z: np.ndarray,
    schemas: Sequence[Sequence[int]],
    feature_names: Sequence[str],
    spec: CenterSpec,
    translations: Optional[Translations] = None,
    min_rows: int = 1,
) -> Dict[str, object]:
    """
    Every occupied cell of every schema in `schemas` (feature-index lists,
    one per dimensionality) as a rule dict, with its generalisations.

    `Z` is the 0/1 indicator the rules are read against (already oriented:
    the complement under an absence search). `spec` decides `certified`
    exactly as the lattice does (`_FullCellStatistics`: the displayed
    partition, all rows, the certificate's own multiplicity policy).
    Cells with fewer than `min_rows` rows are omitted.

    Returns ``{"rules": [...], "n_samples", "n_positive", "schemas"}`` where
    each rule is::

        {"d", "features": [j, ...], "conditions": [{"feature", "column",
         "column_label", "value", "value_label"}, ...], "n", "k", "purity",
         "purity_lower", "certified",
         "parents": [{"features", "n", "k", "purity"}, ...],   # drop one condition
         "best_generalisation": {"features", "n", "k", "purity"} | None}

    `parents` are the d immediate generalisations (one condition removed),
    in the order of the removed condition; `best_generalisation` is the
    proper sub-conjunction (any size) with the highest purity, ties to the
    fewest conditions. Both are None/empty for d = 1.
    """
    X_arr = np.asarray(X, dtype=object)
    z = (np.asarray(Z).ravel() == 1)
    n_samples = int(z.shape[0])
    n_positive = int(z.sum())
    min_rows = max(1, int(min_rows))

    # One full-data statistics object per distinct feature subset: the
    # schemas' own subsets and every proper sub-conjunction of them.
    stats_cache: Dict[Tuple[int, ...], _FullCellStatistics] = {}

    def stats_for(subset: Tuple[int, ...]) -> _FullCellStatistics:
        st = stats_cache.get(subset)
        if st is None:
            st = _FullCellStatistics([X_arr[:, j] for j in subset], z, spec)
            stats_cache[subset] = st
        return st

    def lookup(subset: Tuple[int, ...], key: Tuple[str, ...]) -> Dict[str, object]:
        st = stats_for(subset)
        rec = st.lookup(key)
        return {
            "features": list(subset), "n": int(rec["n"]), "k": int(rec["k"]),
            "purity": float(rec["purity"]),
        }

    rules: List[Dict[str, object]] = []
    schema_list: List[List[int]] = []
    for schema in schemas:
        feats = tuple(int(j) for j in schema)
        if not feats or len(set(feats)) != len(feats):
            raise ValueError(f"a schema must be a list of distinct feature indices, got {list(schema)}")
        if any(j < 0 or j >= X_arr.shape[1] for j in feats):
            raise ValueError(f"schema {list(feats)} names a feature outside [0, {X_arr.shape[1]})")
        schema_list.append(list(feats))
        d = len(feats)
        st = stats_for(feats)
        # Proper sub-conjunctions, by position within the schema.
        subsets_by_size = {
            r: list(combinations(range(d), r)) for r in range(1, d)
        }
        for key, i in st.index.items():
            n = int(st.n_per_cell[i])
            if n < min_rows:
                continue
            k = int(st.k_per_cell[i])
            conditions = [
                {
                    "feature": int(feats[p]),
                    "column": str(feature_names[feats[p]]),
                    "column_label": humanize_col(str(feature_names[feats[p]]), translations),
                    "value": str(key[p]),
                    "value_label": humanize_val(str(feature_names[feats[p]]), key[p], translations),
                }
                for p in range(d)
            ]
            parents: List[Dict[str, object]] = []
            best: Optional[Dict[str, object]] = None
            if d > 1:
                for r in range(1, d):
                    for pos in subsets_by_size[r]:
                        sub = tuple(feats[p] for p in pos)
                        sub_key = tuple(key[p] for p in pos)
                        g = lookup(sub, sub_key)
                        if r == d - 1:
                            parents.append(g)
                        # highest purity wins; on ties the fewer conditions
                        # (smaller r is visited first, so strict > keeps it)
                        if best is None or g["purity"] > best["purity"]:
                            best = g
                # `parents` in the order of the REMOVED condition, so that
                # parents[p] is "this rule without condition p".
                parents = [
                    next(g for g in parents if int(feats[p]) not in g["features"])
                    for p in range(d)
                ]
            rules.append({
                "d": d,
                "features": list(feats),
                "conditions": conditions,
                "n": n,
                "k": k,
                "purity": (k / n) if n else 0.0,
                "purity_lower": float(st.lower[i]),
                "certified": bool(st.certified[i]),
                "parents": parents,
                "best_generalisation": best,
            })
    # Highest purity first, then more rows, then fewer conditions: the order
    # a reader wants a rule list in.
    rules.sort(key=lambda r: (-r["purity"], -r["n"], r["d"]))
    return {
        "rules": rules,
        "n_samples": n_samples,
        "n_positive": n_positive,
        "schemas": schema_list,
        "min_rows": min_rows,
    }


def union_coverage(
    X: np.ndarray,
    Z: np.ndarray,
    rules: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """
    What a SET of rules covers together, as a predictor of the value: the
    rows satisfying at least one rule (their union - rules overlap, a cell
    of a 2D schema lies inside cells of its 1D sub-schemas, so the sum of
    the rules' rows is not the answer), how many of those rows carry the
    value (`k`), the share of all value rows reached (`coverage` - the
    recall of "predict the value where some rule fires") and the share of
    the value among the covered rows (`precision`). Each rule is
    ``{"features": [j, ...], "values": [v, ...]}`` in the columns of `X`.
    """
    X_arr = np.asarray(X, dtype=object)
    z = (np.asarray(Z).ravel() == 1)
    n = int(z.shape[0])
    n_pos = int(z.sum())
    str_cols: Dict[int, np.ndarray] = {}
    eq_cache: Dict[Tuple[int, str], np.ndarray] = {}

    def eq(j: int, v: str) -> np.ndarray:
        key = (j, v)
        m = eq_cache.get(key)
        if m is None:
            col = str_cols.get(j)
            if col is None:
                col = X_arr[:, j].astype(str)
                str_cols[j] = col
            m = col == v
            eq_cache[key] = m
        return m

    covered = np.zeros(n, dtype=bool)
    for r in rules:
        feats = [int(j) for j in r["features"]]
        vals = [str(v) for v in r["values"]]
        if len(feats) != len(vals) or not feats:
            raise ValueError("each rule needs equally many features and values")
        if any(j < 0 or j >= X_arr.shape[1] for j in feats):
            raise ValueError(f"rule names a feature outside [0, {X_arr.shape[1]})")
        m = eq(feats[0], vals[0]).copy()
        for j, v in zip(feats[1:], vals[1:]):
            m &= eq(j, v)
        covered |= m
    n_cov = int(covered.sum())
    k_cov = int((covered & z).sum())
    return {
        "n_rules": int(len(rules)),
        "n_covered": n_cov,
        "k_covered": k_cov,
        "mass": (n_cov / n) if n else 0.0,
        "coverage": (k_cov / n_pos) if n_pos else 0.0,
        "precision": (k_cov / n_cov) if n_cov else 0.0,
        "n_samples": n,
        "n_positive": n_pos,
    }
