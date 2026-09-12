"""
VSF dataset screen: what the framework will do with each column, and which
columns are renamings of each other, BEFORE any target is chosen
(Project_Master_Document.md Section 4.12).

Why before the target
---------------------
Every quantity here is a function of the FEATURE columns alone (the target
screen of `target_report` is the one exception and is computed after the
target is named, for leakage). The reader therefore decides what to exclude
without having seen a single result, which is the only point in the workflow
at which a decision about the feature space carries no selection effect. The
decision is recorded: excluded columns are part of the analysis key, the
legend and the export, because they change the candidate family, the
familywise null and every landscape count.

What a dependency is here
-------------------------
For two columns X (a levels) and Y (b levels), the DETERMINATION of Y by X is

    delta(X -> Y) = (1/N) * sum_{x} max_{y} n(x, y),

the share of rows on which the best deterministic map X -> Y is right;
1 - delta is the g3 error of the functional dependency X -> Y in the sense of
Kivinen & Mannila (1995) - the smallest share of rows that must be removed to
make the dependency hold - and `exceptions` reports N * (1 - delta) as an
integer count. delta = 1 iff X functionally determines Y; delta(X -> Y) =
delta(Y -> X) = 1 iff the two columns induce the SAME partition of the rows
(a renaming).

delta alone is not readable, because a constant predictor already reaches the
largest class share of Y: a column in which one level covers 99 % of the rows
is "determined" to 0.99 by anything. The screen therefore also reports

    strength = (delta - baseline) / (1 - baseline),   baseline = max_y n(y)/N,

the share of the possible improvement over that constant predictor that is
realised (Cohen's kappa applied to the majority rule), and ranks by it.

What each finding licenses
--------------------------
* EQUIVALENT (delta = 1 both ways): the partitions are identical up to
  renaming. Dropping either column is lossless for every candidate and every
  certificate; only the size of the candidate family changes.
* EXACT one way (delta(X -> Y) = 1, delta(Y -> X) < 1): a schema holding both
  X and Y has EXACTLY the partition of the schema without Y whenever the
  smaller schema fits the grid capacity (`vsf.avr` prunes those candidates
  when `prune_dependent=True`). Dropping Y as a column is NOT lossless: Y
  alone is a coarsening that X alone cannot reproduce, and coarser cells are
  larger, hence easier to certify.
* APPROXIMATE (delta < 1): nothing is proved. The screen reports the exact
  number of exception rows and leaves the decision to the reader.

Nothing here is applied automatically. The screen is a report.

Cost
----
One contingency table per unordered pair of columns, both directions read off
it: O(M^2 N) time, O(a b) memory per pair with a sort-based fallback above
`_CONTINGENCY_CAP` cells. The pair screen is skipped, with the reason in the
payload, when M (M - 1) N / 2 exceeds `_PAIR_BUDGET`; the per-column profile
is always computed. Measured: `titanic` (891 x 9) 4 ms, `audiology`
(226 x 70) 0.1 s, `mushroom` (8124 x 22) 0.05 s.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pmd import discretize_feature, grid_capacity

__all__ = [
    "DEFAULT_MIN_STRENGTH",
    "ColumnProfile",
    "DependencyPair",
    "DatasetScreen",
    "column_profiles",
    "dependency_pairs",
    "exact_dependencies",
    "screen_dataset",
    "target_report",
]

#: Smallest `strength` an inexact pair must reach to be reported. Exact
#: dependencies are always reported whatever their strength.
DEFAULT_MIN_STRENGTH: float = 0.90
#: A level covering at least this share of the rows makes a column a poor
#: axis: every other level is a small cell. Advisory flag only.
DOMINANT_LEVEL_SHARE: float = 0.90
#: Labels that NAME a placeholder for "no value recorded". A category whose
#: label matches one of these (case-insensitive, as a whole label or as a
#: word inside it) and which covers at least `PLACEHOLDER_MIN_SHARE` of the
#: rows is reported: the framework cannot know whether such a category means
#: missing data or a real value, and the difference decides whether a centre
#: built on it is a statement about the characteristic or about the way the
#: data were collected (`deck = unknown_deck` on `titanic`, 77 % of the rows,
#: is the running example). Naming is all that is checked; the reading is the
#: reader's.
PLACEHOLDER_TOKENS: Tuple[str, ...] = (
    "?", "-", "na", "n/a", "nan", "null", "none", "nil", "missing", "unknown",
    "unspecified", "undefined", "unrecorded", "notrecorded", "notavailable",
    "notapplicable", "other", "blank", "empty",
)
#: Smallest share a placeholder-looking category must cover to be reported.
PLACEHOLDER_MIN_SHARE: float = 0.05
#: Largest contingency table built densely; above it the pair goes through
#: the sort-based counter (same exact result).
_CONTINGENCY_CAP: int = 1 << 24
#: Budget of the pair screen, in table cells (pairs x rows). Above it the
#: pair screen is skipped and says so.
_PAIR_BUDGET: int = 5 * 10 ** 8


@dataclass(frozen=True)
class ColumnProfile:
    """
    What one column is, as the encoding (`vsf.pmd.discretize_feature`) sees
    it: its number of categories, its largest category, how many categories
    hold a single row, and the flags that say what the framework will do
    with it.

    flags
        `"constant"` - one category: every schema containing this column has
        the partition of the schema without it, so it adds candidates and no
        resolution. `"exceeds_capacity"` - the column alone has more
        categories than the grid capacity floor(N/10), so even its 1D grid is
        coarsened (Section 2.3) and no cell can be large. `"dominant_level"`
        - one category covers at least `DOMINANT_LEVEL_SHARE` of the rows, so
        the column splits the data very unevenly. `"placeholder_level"` - a
        category whose LABEL names a placeholder ("?", "unknown",
        "unknown_deck", "not recorded", ...) covers at least
        `PLACEHOLDER_MIN_SHARE` of the rows (`placeholder_level` and
        `placeholder_share` name it): a centre built on that category is a
        statement about how the data were collected, unless the label means
        a real value here. The framework checks the name only; the reading is
        the reader's.
    """

    name: str
    n_levels: int
    n_rows: int
    largest_level: str
    largest_level_count: int
    largest_level_share: float
    n_singleton_levels: int
    n_missing: int
    placeholder_level: Optional[str] = None
    placeholder_share: float = 0.0
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DependencyPair:
    """
    One unordered pair of columns, with the determination measured in both
    directions (`delta`, `strength`, `exceptions`; see the module docstring).
    `kind` is `"equivalent"` (both directions exact), `"exact"` (one
    direction exact: `source` determines `target_col`) or `"approximate"`.
    """

    a: str
    b: str
    delta_ab: float
    delta_ba: float
    strength_ab: float
    strength_ba: float
    exceptions_ab: int
    exceptions_ba: int
    kind: str

    @property
    def best_strength(self) -> float:
        return max(self.strength_ab, self.strength_ba)

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["best_strength"] = self.best_strength
        return out


@dataclass(frozen=True)
class DatasetScreen:
    """The per-column profile and the reported dependency pairs of one dataset."""

    n_rows: int
    n_columns: int
    grid_capacity: int
    min_strength: float
    columns: List[ColumnProfile]
    pairs: List[DependencyPair]
    pairs_skipped: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "grid_capacity": self.grid_capacity,
            "min_strength": self.min_strength,
            "columns": [c.to_dict() for c in self.columns],
            "pairs": [p.to_dict() for p in self.pairs],
            "pairs_skipped": self.pairs_skipped,
        }


# --------------------------------------------------------------------------
# Encoding helpers
# --------------------------------------------------------------------------
def _encode(X: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    """Category codes and category counts of every column, as the search encodes them."""
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols, counts = [], []
    for j in range(arr.shape[1]):
        codes, k = discretize_feature(arr[:, j])
        cols.append(np.asarray(codes, dtype=np.int64))
        counts.append(int(k))
    return cols, counts


def _contingency_max(x: np.ndarray, y: np.ndarray, a: int, b: int) -> Tuple[int, int]:
    """
    (sum_x max_y n(x, y), sum_y max_x n(x, y)) - the correctly mapped row
    counts of the best deterministic maps X -> Y and Y -> X. Exact integers;
    dense table when a * b fits `_CONTINGENCY_CAP`, otherwise a sort-based
    count of the occupied pairs only.
    """
    if a <= 0 or b <= 0 or x.shape[0] == 0:
        return 0, 0
    if a * b <= _CONTINGENCY_CAP:
        table = np.bincount(x * b + y, minlength=a * b).reshape(a, b)
        return int(table.max(axis=1).sum()), int(table.max(axis=0).sum())
    key = x * b + y
    uniq, cnt = np.unique(key, return_counts=True)
    gx, gy = uniq // b, uniq % b
    best_x = np.zeros(a, dtype=np.int64)
    best_y = np.zeros(b, dtype=np.int64)
    np.maximum.at(best_x, gx, cnt)
    np.maximum.at(best_y, gy, cnt)
    return int(best_x.sum()), int(best_y.sum())


def _strength(delta: float, baseline: float) -> float:
    """(delta - baseline) / (1 - baseline), the share of the possible gain over the majority rule."""
    if baseline >= 1.0:
        return 1.0 if delta >= 1.0 else 0.0
    return float((delta - baseline) / (1.0 - baseline))


# --------------------------------------------------------------------------
# Per-column profile
# --------------------------------------------------------------------------
def column_profiles(
    X: np.ndarray, feature_names: Optional[Sequence[str]] = None
) -> List[ColumnProfile]:
    """
    One `ColumnProfile` per column of a raw (N, M) matrix, computed on the
    same category encoding the search uses. `n_missing` counts the rows whose
    value is missing (None/NaN/NA - one category, coded last by
    `vsf.pmd.discretize_feature`).
    """
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n, m = arr.shape
    names = [f"X_{j + 1}" for j in range(m)] if feature_names is None else [str(f) for f in feature_names]
    if len(names) != m:
        raise ValueError(f"feature_names has {len(names)} entries for {m} columns")
    capacity = grid_capacity(n)
    out: List[ColumnProfile] = []
    for j in range(m):
        codes, k = discretize_feature(arr[:, j])
        codes = np.asarray(codes, dtype=np.int64)
        counts = np.bincount(codes, minlength=max(k, 1))
        top = int(np.argmax(counts)) if counts.size else 0
        top_count = int(counts[top]) if counts.size else 0
        label_idx = np.nonzero(codes == top)[0]
        label = _label(arr[label_idx[0], j]) if label_idx.size else ""
        missing = int(np.count_nonzero([_is_missing(v) for v in arr[:, j].tolist()])) if n else 0
        ph_label, ph_share = None, 0.0
        if n and k:
            # First occurrence of every category, to read its raw label.
            _, first_idx = np.unique(codes, return_index=True)
            for level, idx in enumerate(first_idx.tolist()):
                share = float(counts[level]) / n
                if share >= PLACEHOLDER_MIN_SHARE and _is_placeholder(_label(arr[idx, j])) and share > ph_share:
                    ph_label, ph_share = _label(arr[idx, j]), share
        flags: List[str] = []
        if k <= 1:
            flags.append("constant")
        if k > capacity:
            flags.append("exceeds_capacity")
        if n and top_count / n >= DOMINANT_LEVEL_SHARE:
            flags.append("dominant_level")
        if n and missing / n >= 0.5:
            flags.append("mostly_missing")
        if ph_label is not None:
            flags.append("placeholder_level")
        out.append(ColumnProfile(
            name=names[j], n_levels=int(k), n_rows=int(n),
            largest_level=label, largest_level_count=top_count,
            largest_level_share=(top_count / n) if n else 0.0,
            n_singleton_levels=int(np.count_nonzero(counts == 1)),
            n_missing=missing, placeholder_level=ph_label, placeholder_share=ph_share,
            flags=flags,
        ))
    return out


def _flatten_label(text: str) -> str:
    """A label reduced to its letters and digits, for matching `PLACEHOLDER_TOKENS`."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _is_placeholder(label: str) -> bool:
    """
    True when a category's LABEL names a placeholder for a missing value
    (`PLACEHOLDER_TOKENS`), as the whole label or as one of its words after
    splitting on the usual separators: "?", "unknown", "unknown_deck",
    "not recorded", "N/A".
    """
    text = str(label).strip().lower()
    if not text:
        return True
    flat = _flatten_label(text)
    if not flat:  # "-", "--", ".": punctuation only
        return True
    if flat in {_flatten_label(t) for t in PLACEHOLDER_TOKENS} - {""}:
        return True
    words = [w for w in text.replace("-", " ").replace("_", " ").replace("/", " ").split() if w]
    return any(w in PLACEHOLDER_TOKENS for w in words)


def _is_missing(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    return type(v).__name__ in ("NAType", "NaTType")


def _label(v: object) -> str:
    if _is_missing(v):
        return "missing"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


# --------------------------------------------------------------------------
# Pairwise dependencies
# --------------------------------------------------------------------------
def dependency_pairs(
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    min_strength: float = DEFAULT_MIN_STRENGTH,
) -> Tuple[List[DependencyPair], Optional[str]]:
    """
    Every unordered pair of columns whose determination reaches
    `min_strength` in at least one direction, plus every exact dependency
    whatever its strength, strongest first. The second element is None, or
    the reason the pair screen was skipped (cost budget).
    """
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n, m = arr.shape
    names = [f"X_{j + 1}" for j in range(m)] if feature_names is None else [str(f) for f in feature_names]
    if not (0.0 <= min_strength <= 1.0):
        raise ValueError(f"min_strength must be in [0, 1], got {min_strength}")
    if m < 2 or n == 0:
        return [], None
    budget = m * (m - 1) // 2 * n
    if budget > _PAIR_BUDGET:
        return [], (
            f"the pairwise screen needs {m * (m - 1) // 2} column pairs over {n} rows "
            f"({budget:.3g} cell counts), above the budget of {_PAIR_BUDGET:.3g}; "
            "run vsf.screen.dependency_pairs directly on a subset of the columns"
        )
    codes, counts = _encode(arr)
    baselines = [
        (float(np.bincount(c, minlength=max(k, 1)).max()) / n) if n else 0.0
        for c, k in zip(codes, counts)
    ]
    out: List[DependencyPair] = []
    for i in range(m):
        for j in range(i + 1, m):
            hit_ij, hit_ji = _contingency_max(codes[i], codes[j], counts[i], counts[j])
            d_ij, d_ji = hit_ij / n, hit_ji / n
            s_ij, s_ji = _strength(d_ij, baselines[j]), _strength(d_ji, baselines[i])
            exact_ij, exact_ji = hit_ij == n, hit_ji == n
            if exact_ij and exact_ji:
                kind = "equivalent"
            elif exact_ij or exact_ji:
                kind = "exact"
            else:
                kind = "approximate"
            if kind == "approximate" and max(s_ij, s_ji) < min_strength:
                continue
            out.append(DependencyPair(
                a=names[i], b=names[j],
                delta_ab=d_ij, delta_ba=d_ji,
                strength_ab=s_ij, strength_ba=s_ji,
                exceptions_ab=int(n - hit_ij), exceptions_ba=int(n - hit_ji),
                kind=kind,
            ))
    order = {"equivalent": 0, "exact": 1, "approximate": 2}
    out.sort(key=lambda p: (order[p.kind], -p.best_strength, p.a, p.b))
    return out, None


def exact_dependencies(
    X: np.ndarray, feature_names: Optional[Sequence[str]] = None
) -> Dict[int, Tuple[int, ...]]:
    """
    `{j: (i, ...)}` - for every column j, the columns that determine it
    EXACTLY (delta = 1), by index. This is what `vsf.avr`'s
    `prune_dependent` consumes: a candidate holding both i and j has the
    partition of the candidate without j (see that module). A constant
    column is determined by every other column, which is correct: adding it
    to any schema refines nothing.
    """
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n, m = arr.shape
    if m < 2 or n == 0:
        return {}
    codes, counts = _encode(arr)
    determined: Dict[int, List[int]] = {}
    for i in range(m):
        for j in range(i + 1, m):
            hit_ij, hit_ji = _contingency_max(codes[i], codes[j], counts[i], counts[j])
            if hit_ij == n:
                determined.setdefault(j, []).append(i)
            if hit_ji == n:
                determined.setdefault(i, []).append(j)
    return {j: tuple(sorted(v)) for j, v in determined.items()}


def screen_dataset(
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    min_strength: float = DEFAULT_MIN_STRENGTH,
) -> DatasetScreen:
    """The per-column profile and the dependency pairs of a raw (N, M) matrix."""
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    profiles = column_profiles(arr, feature_names)
    pairs, skipped = dependency_pairs(arr, feature_names, min_strength)
    return DatasetScreen(
        n_rows=int(arr.shape[0]), n_columns=int(arr.shape[1]),
        grid_capacity=grid_capacity(int(arr.shape[0])),
        min_strength=float(min_strength),
        columns=profiles, pairs=pairs, pairs_skipped=skipped,
    )


# --------------------------------------------------------------------------
# Target screen (leakage)
# --------------------------------------------------------------------------
def target_report(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """
    How well each column determines the TARGET indicator, strongest first:
    `delta` (the share of rows the best map column -> target gets right),
    `strength` against the majority-class baseline, and the number of
    exception rows. A column at strength 1 reproduces the target exactly -
    a tautology or a leak, and the search will "find" it as the whole
    answer. This is the one part of the screen that depends on the target,
    so it is computed after the target is named, not before.

    `Z` is the indicator the search will run on (0/1), or any categorical
    target column.
    """
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n, m = arr.shape
    names = [f"X_{j + 1}" for j in range(m)] if feature_names is None else [str(f) for f in feature_names]
    z_codes, z_k = discretize_feature(np.asarray(Z, dtype=object).ravel())
    z_codes = np.asarray(z_codes, dtype=np.int64)
    if n == 0 or m == 0:
        return []
    baseline = float(np.bincount(z_codes, minlength=max(z_k, 1)).max()) / n
    codes, counts = _encode(arr)
    out: List[Dict[str, object]] = []
    for j in range(m):
        hit, _ = _contingency_max(codes[j], z_codes, counts[j], z_k)
        delta = hit / n
        out.append({
            "feature": names[j],
            "delta": delta,
            "strength": _strength(delta, baseline),
            "exceptions": int(n - hit),
            "exact": hit == n,
            "n_levels": int(counts[j]),
        })
    out.sort(key=lambda e: (-float(e["strength"]), str(e["feature"])))
    return out
