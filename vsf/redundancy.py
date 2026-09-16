"""
VSF redundant centres: which discrete centres of the solution landscape
describe (almost) the same objects (Project_Master_Document.md Section 4.11).

The landscape (Section 4.9) counts SCHEMAS. Two schemas built from different
characteristics can certify centres that hold almost the same rows - on
`titanic` at tau = 0.7, `title = Mr` and `sex = male AND title = Mr` hold
the identical 517 rows, `sex = male` holds them and 60 more - and the
landscape shows them as unrelated points. This
module works one level down, on the CENTRES of every schema the search
scored, and answers one question: which of them are the same group of
objects described by different characteristics?

Definitions
-----------
Let R(c) be the set of rows of centre c (all rows of the cell, positives and
negatives alike: two centres that capture the same positives but different
negatives are NOT the same group). The similarity of two centres is their
MUTUAL CONTAINMENT

    s(a, b) = |R(a) & R(b)| / max(|R(a)|, |R(b)|)
            = min( |R(a) & R(b)| / |R(a)| ,  |R(a) & R(b)| / |R(b)| ),

the smaller of the two one-sided shares. s(a, b) >= t says that EACH centre
lies at least a fraction t inside the other. A one-sided share would call a
small centre nested in a large one a duplicate of it; Jaccard would do the
opposite trade and is harder to state in words. s = 1 iff the row sets are
identical.

Grouping at a user threshold t is the LEADER (sequential) clustering of
Hartigan (1975, Sec. 3.2) over centres taken in a fixed rank order:

    rank: fewest characteristics (d) first, then the highest Clopper-Pearson
          lower bound of the cell's purity at the schema's own Bonferroni
          level alpha / C_occ (the interval the lattice prints for the
          cell), then the larger cell, then enumeration order.

A centre joins the already-chosen leader it is most similar to, provided
that similarity is at least t; otherwise it becomes a leader itself. This
gives exactly two guarantees, both checked by the test suite:

  * every member of a group lies at least t inside its representative and
    the representative at least t inside it (s(member, leader) >= t);
  * no two representatives are duplicates of each other (s < t pairwise).

It does NOT guarantee that two members of one group are t-similar to each
other (s is not transitive; complete-linkage clustering would, at the price
of splitting groups the reader sees as one). The interface therefore
reports every member's similarity TO THE REPRESENTATIVE, the quantity the
guarantee is about.

Centres with identical row sets always fall in the same group (they have
identical similarities to every leader); the computation therefore runs on
DISTINCT row sets and maps centres back to them.

Scope and cost
--------------
The centres are those of the SEARCH partition of every candidate - the same
cells, under the same `CenterSpec`, as `vsf.avr.Landscape` counts (pinned in
`tests/test_redundancy.py`); for a schema that the grid-capacity rule
coarsened, a centre's description lists every category its cell holds.
Pairs are kept only when s >= `PAIR_FLOOR` (0.5): below it a pair shares
less than half of the larger centre, which no threshold offered to the user
calls "almost the same". Since s(a, b) <= min/max of the two sizes, a
centre of size n can only reach the floor against centres of size in
[PAIR_FLOOR * n, n / PAIR_FLOOR]; candidates are enumerated in that size
band only, and the band is exact (no pair above the floor is skipped).
Intersections are exact integer counts from a blocked 0/1 matrix product
in float32 (exact below 2^24 rows; checked). Worst case O(S^2 N) for S
distinct row sets; measured on two cores: `titanic` 892 sets in 0.1 s,
`mushroom` 19 327 sets in 10.6 s, `chess_krkp` 18 657 sets in 12.0 s;
computed once per (target, value, certificate, direction, min_rows) and
cached by the server.

What is not claimed
-------------------
Grouping is DESCRIPTIVE. It changes no centre, no coverage and no p-value;
it reports which certified cells coincide. Its output depends on t, on the
rank rule and on `min_rows` (a display floor on centre size that defaults
to 1 in the library), all of which are explicit parameters and appear in
every response.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .avr import (
    MAX_BRANCH_D,
    Direction,
    Landscape,
    _CandidateFactory,
    _prepare_search,
    resolve_center_spec,
)
from .centers import CenterSpec, _beta_quantile, clopper_pearson_lower, min_successes_to_select

__all__ = [
    "PAIR_FLOOR",
    "DEFAULT_GROUP_THRESHOLD",
    "MAX_DISTINCT_SETS",
    "CenterCatalog",
    "CenterGrouping",
    "collect_centers",
]

#: Smallest mutual containment a pair must reach to be stored at all, and
#: the smallest threshold `CenterCatalog.group` accepts.
PAIR_FLOOR: float = 0.5
#: Threshold the interface opens with ("each centre holds at least 80 % of
#: the other's rows").
DEFAULT_GROUP_THRESHOLD: float = 0.8
#: Refuse, with a message, above this many distinct centre row sets: the
#: pair pass is quadratic in it. Raising `min_rows` shrinks it.
MAX_DISTINCT_SETS: int = 60_000
#: Tolerance of the threshold comparison s >= t on s = integer / integer.
_EPS: float = 1e-12
#: Rows of the left operand per block of the pair pass (upper bound).
_BLOCK: int = 256
#: Columns of the right operand per chunk of the pair pass (upper bound).
_CHUNK: int = 2048
#: Memory budget, in float32 cells, of one unpacked operand of the pair pass;
#: block and chunk shrink with N so that neither exceeds it (64 MB each) for
#: N <= 250 000 rows; above that the floors (16 and 64 sets) bound them.
_OPERAND_CELLS: int = 16_000_000
#: Width of a histogram bin of nearest-neighbour similarity.
_HIST_STEP: float = 0.05

GroupSort = Literal["coverage", "members"]
#: Which alternatives a view lists, by how their schema stands to the
#: reference's: "related" - one column set is nested in the other (the same
#: lineage of the search lattice); "unrelated" - neither is.
Kinship = Literal["all", "related", "unrelated"]


def _value_label(v: object) -> str:
    """Display label of one raw category value (missing values become 'missing')."""
    if v is None:
        return "missing"
    if isinstance(v, float):
        if math.isnan(v):
            return "missing"
        if v.is_integer():
            return str(int(v))
    return str(v)


def _clopper_pearson_lower_per_level(k: np.ndarray, n: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    `vsf.centers.clopper_pearson_lower` with a per-element level: the same
    closed forms (0 for k = 0, alpha^(1/n) for k = n) and the same 40-step
    bisection of the incomplete beta, evaluated in one vectorised pass
    instead of one pass per distinct level (equality pinned in the tests).
    """
    k_arr = np.asarray(k, dtype=np.int64)
    n_arr = np.asarray(n, dtype=np.int64)
    a_arr = np.asarray(alpha, dtype=np.float64)
    if np.any((a_arr <= 0.0) | (a_arr >= 1.0)):
        raise ValueError("alpha must be in (0, 1) elementwise")
    if np.any(k_arr < 0) or np.any(n_arr < 0) or np.any(k_arr > n_arr):
        raise ValueError("require 0 <= k <= n elementwise")
    out = np.zeros(k_arr.shape, dtype=np.float64)
    full = (k_arr == n_arr) & (n_arr > 0)
    if np.any(full):
        out[full] = np.power(a_arr[full], 1.0 / n_arr[full].astype(np.float64))
    mask = (k_arr > 0) & ~full
    if np.any(mask):
        a = k_arr[mask].astype(np.float64)
        b = (n_arr[mask] - k_arr[mask] + 1).astype(np.float64)
        out[mask] = _beta_quantile(a_arr[mask], a, b)  # type: ignore[arg-type]
    return out


def _popcount_rows(packed: np.ndarray) -> np.ndarray:
    """Number of set bits in each row of a uint8 bit-packed matrix."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(packed).sum(axis=-1, dtype=np.int64)
    return np.unpackbits(packed, axis=-1).sum(axis=-1, dtype=np.int64)  # numpy < 2.0


@dataclass(frozen=True)
class CenterGrouping:
    """
    The groups of one threshold: `leader_of_set[u]` is the distinct row set
    representing set u's group (u itself for a representative),
    `sim_to_leader[u]` the mutual containment of u with it (1 for a
    representative). `leaders` lists the representatives in rank order.
    """

    threshold: float
    leader_of_set: np.ndarray
    sim_to_leader: np.ndarray
    leaders: np.ndarray


class CenterCatalog:
    """
    Every discrete centre of every schema the exhaustive search scored,
    with its row set, the similarity graph between them (pairs with mutual
    containment >= `PAIR_FLOOR`) and the rank rule of the grouping. Built
    by `collect_centers`; `group` groups it at a threshold; `groups_page`
    and `group_detail` serialise the result for `/api/centers/*`.

    Arrays indexed by schema `q`: `schema_features[q]`, `schema_alpha[q]`
    (Bonferroni per-cell level alpha / C_occ of its search partition).
    Indexed by centre `c`: `center_schema`, `center_cell` (dense cell code
    of the search partition), `center_set`. Indexed by distinct row set
    `u`: `set_n`, `set_k`, `set_bits` (row membership, bit-packed),
    `set_best_center` (the centre whose description represents the set),
    `set_lower`, `set_rank`.
    """

    def __init__(
        self,
        *,
        factory: _CandidateFactory,
        raw_X: np.ndarray,
        z: np.ndarray,
        spec: CenterSpec,
        direction: Direction,
        feature_names: Sequence[str],
        min_rows: int,
        landscape: Landscape,
        schema_features: List[Tuple[int, ...]],
        schema_alpha: np.ndarray,
        schema_landscape_index: np.ndarray,
        center_schema: np.ndarray,
        center_cell: np.ndarray,
        center_set: np.ndarray,
        set_n: np.ndarray,
        set_k: np.ndarray,
        set_bits: np.ndarray,
        set_index: Dict[bytes, int],
    ) -> None:
        self.factory = factory
        self.raw_X = raw_X
        self.z = z.astype(np.int8, copy=False)
        self.z_bits = np.packbits(self.z.astype(bool))
        self.spec = spec
        self.direction: Direction = direction
        self.feature_names = [str(f) for f in feature_names]
        self.min_rows = int(min_rows)
        self.landscape = landscape
        self.n_samples = int(z.shape[0])
        self.n_positive = int(z.sum())
        self.schema_features = schema_features
        self.schema_d = np.asarray([len(f) for f in schema_features], dtype=np.int64)
        self.schema_alpha = schema_alpha
        self.schema_landscape_index = schema_landscape_index
        self.center_schema = center_schema
        self.center_cell = center_cell
        self.center_set = center_set
        self.set_n = set_n
        self.set_k = set_k
        self.set_bits = set_bits
        self.set_index = set_index
        self.schema_of_features: Dict[Tuple[int, ...], int] = {
            f: q for q, f in enumerate(schema_features)
        }
        #: Column sets of the schemas, for the nesting test of `_kinship`.
        self._schema_sets: List[frozenset] = [frozenset(f) for f in schema_features]
        #: Centres of each distinct row set, CSR-style; built on first use.
        self._set_centers_indptr: Optional[np.ndarray] = None
        self._set_centers_indices: Optional[np.ndarray] = None
        self.n_centers = int(center_schema.shape[0])
        self.n_sets = int(set_n.shape[0])
        self._rank_sets()
        self._build_pairs()
        self._landscape_coordinates()
        self._groupings: Dict[float, CenterGrouping] = {}
        self._codes_cache: Dict[int, np.ndarray] = {}
        self._labels: Dict[int, List[str]] = {}
        # The server shares one catalogue between request threads; the
        # caches above are the only mutable state after construction.
        self._lock = threading.Lock()

    # -- rank ---------------------------------------------------------------
    def _rank_sets(self) -> None:
        """
        Best centre per distinct set (fewest characteristics, then the
        fewest occupied cells in its schema - the largest Bonferroni level,
        hence the highest lower bound for the same (k, n) - then enumeration
        order), its Clopper-Pearson lower bound, and the rank of every set.
        """
        S, C = self.n_sets, self.n_centers
        d_c = self.schema_d[self.center_schema]
        a_c = self.schema_alpha[self.center_schema]
        # lexsort: last key is primary. Within a set: smaller d, larger alpha, earlier centre.
        order = np.lexsort((np.arange(C), -a_c, d_c, self.center_set))
        first = np.ones(C, dtype=bool)
        first[1:] = self.center_set[order][1:] != self.center_set[order][:-1]
        best = np.empty(S, dtype=np.int64)
        best[self.center_set[order][first]] = order[first]
        self.set_best_center = best
        self.set_d = d_c[best]
        self.set_first_center = np.full(S, C, dtype=np.int64)
        np.minimum.at(self.set_first_center, self.center_set, np.arange(C, dtype=np.int64))
        self.set_lower = _clopper_pearson_lower_per_level(self.set_k, self.set_n, a_c[best])
        rank_order = np.lexsort((self.set_first_center, -self.set_n, -self.set_lower, self.set_d))
        self.set_rank = np.empty(S, dtype=np.int64)
        self.set_rank[rank_order] = np.arange(S, dtype=np.int64)
        self.rank_order = rank_order
        self.set_multiplicity = np.bincount(self.center_set, minlength=S).astype(np.int64)

    # -- pairs --------------------------------------------------------------
    def _build_pairs(self) -> None:
        """
        Every pair of distinct sets with mutual containment >= PAIR_FLOOR,
        as a symmetric CSR graph (`pair_indptr`, `pair_indices`,
        `pair_inter` = exact |R(a) & R(b)|). Sets are visited by decreasing
        size; set i is compared only with the later (not larger) sets j
        whose size is at least PAIR_FLOOR * |R(i)| - the exact band outside
        which s < PAIR_FLOOR.
        """
        S, N = self.n_sets, self.n_samples
        if N >= (1 << 24):  # pragma: no cover - float32 counts exact below 2^24
            raise ValueError("the pair pass counts intersections in float32 and needs N < 2^24 rows")
        order = np.argsort(-self.set_n, kind="stable")
        sz = self.set_n[order]
        block = max(16, min(_BLOCK, _OPERAND_CELLS // max(N, 1)))
        chunk = max(64, min(_CHUNK, _OPERAND_CELLS // max(N, 1)))
        us: List[np.ndarray] = []
        vs: List[np.ndarray] = []
        inters: List[np.ndarray] = []

        def dense(ix: np.ndarray) -> np.ndarray:
            return np.unpackbits(self.set_bits[ix], axis=1, count=N).astype(np.float32)

        # position of the last set whose size is >= floor * sz[i]; sz is non-increasing
        neg = -sz
        for i0 in range(0, S, block):
            i1 = min(S, i0 + block)
            band_end = int(np.searchsorted(neg, -PAIR_FLOOR * sz[i1 - 1] + _EPS, side="right"))
            band_end = max(band_end, i1)
            left = dense(order[i0:i1])
            for j0 in range(i0, band_end, chunk):
                j1 = min(band_end, j0 + chunk)
                inter = left @ dense(order[j0:j1]).T
                inter_i = np.rint(inter).astype(np.int64)
                ii = np.arange(i0, i1)[:, None]
                jj = np.arange(j0, j1)[None, :]
                big = np.maximum(sz[i0:i1, None], sz[None, j0:j1])
                keep = (jj > ii) & (inter_i >= PAIR_FLOOR * big - _EPS) & (inter_i > 0)
                if np.any(keep):
                    r, c = np.nonzero(keep)
                    us.append(order[i0 + r])
                    vs.append(order[j0 + c])
                    inters.append(inter_i[r, c])
        u = np.concatenate(us) if us else np.zeros(0, dtype=np.int64)
        v = np.concatenate(vs) if vs else np.zeros(0, dtype=np.int64)
        w = np.concatenate(inters) if inters else np.zeros(0, dtype=np.int64)
        src = np.concatenate([u, v])
        dst = np.concatenate([v, u])
        ww = np.concatenate([w, w])
        o = np.lexsort((dst, src))
        src, dst, ww = src[o], dst[o], ww[o]
        self.pair_indptr = np.zeros(S + 1, dtype=np.int64)
        np.add.at(self.pair_indptr, src + 1, 1)
        self.pair_indptr = np.cumsum(self.pair_indptr)
        self.pair_indices = dst.astype(np.int64)
        self.pair_inter = ww.astype(np.int64)
        big = np.maximum(self.set_n[src], self.set_n[dst]) if src.size else np.zeros(0, dtype=np.int64)
        self.pair_sim = (self.pair_inter / big) if src.size else np.zeros(0, dtype=np.float64)
        self.n_pairs = int(u.shape[0])
        # nearest-neighbour similarity per set (0 when no neighbour reaches the floor)
        nn = np.zeros(S, dtype=np.float64)
        if src.size:
            np.maximum.at(nn, src, self.pair_sim)
        self.set_nn_sim = nn

    def neighbours(self, u: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(sets, intersections, similarities) of set u's stored pairs."""
        lo, hi = int(self.pair_indptr[u]), int(self.pair_indptr[u + 1])
        return self.pair_indices[lo:hi], self.pair_inter[lo:hi], self.pair_sim[lo:hi]

    def intersection(self, a: int, b: int) -> int:
        """Exact |R(a) & R(b)| of two distinct sets (from the packed rows)."""
        if a == b:
            return int(self.set_n[a])
        return int(_popcount_rows(self.set_bits[a] & self.set_bits[b]))

    def positives_in_intersection(self, a: int, b: int) -> int:
        """Exact number of target-indicator rows in R(a) & R(b)."""
        return int(_popcount_rows(self.set_bits[a] & self.set_bits[b] & self.z_bits))

    # -- landscape coordinates ------------------------------------------------
    def _landscape_coordinates(self) -> None:
        """
        Lattice cell of every schema in the landscape as drawn for its own
        dimensionality (`schema_iy_d`) and for the whole family
        (`schema_iy_all`); the column `schema_ix` is the same in both.
        """
        L = self.landscape
        li = self.schema_landscape_index
        n = L.N_BINS
        if li.size == 0:
            empty = np.zeros(0, dtype=np.int64)
            self.schema_ix, self.schema_iy_d, self.schema_iy_all = empty, empty, empty
            return
        x = L.x_values()[li]
        nc = L.n_centers[li].astype(np.float64)
        self.schema_ix = Landscape.bin_index(x, n)
        k_all = L.k_max(None)
        self.schema_iy_all = Landscape.bin_index(nc / float(max(k_all, 1)), n)
        k_d = {d: L.k_max(d) for d in range(1, MAX_BRANCH_D + 1)}
        kd = np.asarray([k_d[int(d)] for d in self.schema_d], dtype=np.float64)
        self.schema_iy_d = Landscape.bin_index(nc / np.maximum(kd, 1.0), n)

    # -- grouping ------------------------------------------------------------
    def group(self, threshold: float) -> CenterGrouping:
        """
        Leader clustering of the distinct row sets at `threshold` (module
        docstring). Cached per threshold.
        """
        t = float(threshold)
        if not (PAIR_FLOOR - _EPS <= t <= 1.0 + _EPS):
            raise ValueError(f"threshold must be in [{PAIR_FLOOR}, 1], got {threshold}")
        t = min(max(t, PAIR_FLOOR), 1.0)
        with self._lock:
            hit = self._groupings.get(t)
            if hit is not None:
                return hit
            out = self._group_unlocked(t)
            if len(self._groupings) >= 16:
                self._groupings.pop(next(iter(self._groupings)))
            self._groupings[t] = out
            return out

    def _group_unlocked(self, t: float) -> CenterGrouping:
        S = self.n_sets
        leader_of = np.full(S, -1, dtype=np.int64)
        sim = np.zeros(S, dtype=np.float64)
        is_leader = np.zeros(S, dtype=bool)
        leaders: List[int] = []
        for u in self.rank_order.tolist():
            nb, _, s = self.neighbours(u)
            ok = is_leader[nb] & (s >= t - _EPS)
            if np.any(ok):
                cand, cs = nb[ok], s[ok]
                best_s = cs.max()
                tie = cand[cs >= best_s - _EPS]
                chosen = int(tie[np.argmin(self.set_rank[tie])])
                leader_of[u] = chosen
                sim[u] = float(cs[cand == chosen][0])
            else:
                is_leader[u] = True
                leader_of[u] = u
                sim[u] = 1.0
                leaders.append(u)
        return CenterGrouping(t, leader_of, sim, np.asarray(leaders, dtype=np.int64))

    # -- descriptions ----------------------------------------------------------
    def _schema_codes(self, q: int) -> np.ndarray:
        with self._lock:
            hit = self._codes_cache.get(q)
            if hit is None:
                hit, _ = self.factory.codes(self.schema_features[q])
                if len(self._codes_cache) >= 256:
                    self._codes_cache.pop(next(iter(self._codes_cache)))
                self._codes_cache[q] = hit
            return hit

    def _column_labels(self, j: int) -> List[str]:
        """
        One display label per CATEGORY of column j (its `vsf.pmd` code), in
        code order - the encoder's sorted order, missing last. Labels are
        made unique within the column: two categories that would print
        alike (NaN and the string "missing", 2.0 and "2" in a mixed column)
        are shown by their `repr` instead, so a description can always be
        read back into exactly the rows it stands for.
        """
        with self._lock:
            hit = self._labels.get(j)
            if hit is not None:
                return hit
            col = self.factory._raw[:, j]
            _, first = np.unique(col, return_index=True)
            reps = self.raw_X[first, j].tolist()
            labels = [_value_label(v) for v in reps]
            seen: Dict[str, int] = {}
            for lab in labels:
                seen[lab] = seen.get(lab, 0) + 1
            # The missing category keeps "missing"; a real value that prints
            # like another category is shown by its repr.
            missing = [_value_label(v) == "missing" and not isinstance(v, str) for v in reps]
            labels = [
                lab if (seen[lab] == 1 or miss) else repr(v)
                for v, lab, miss in zip(reps, labels, missing)
            ]
            self._labels[j] = labels
            return labels

    def category_labels(self, j: int) -> np.ndarray:
        """The label of every row's category in column j (see `_column_labels`)."""
        labels = np.asarray(self._column_labels(j), dtype=object)
        col = self.factory._raw[:, j]
        return labels[np.searchsorted(np.unique(col), col)]

    def describe(self, c: int) -> List[Dict[str, object]]:
        """
        The conjunction a centre stands for: per characteristic of its
        schema, the categories its cell holds, as unique labels
        (`_column_labels`), in the encoder's order. One category per
        characteristic unless the grid-capacity rule merged levels of that
        column in this schema (then `merged` is true and every category
        present is listed); the conjunction selects exactly the cell's
        rows (pinned in the tests).
        """
        q = int(self.center_schema[c])
        return self._describe_rows(self.schema_features[q], self._schema_codes(q) == int(self.center_cell[c]))

    def _describe_rows(self, features: Sequence[int], rows: np.ndarray) -> List[Dict[str, object]]:
        """`describe` for the cell `rows` of the schema on `features`."""
        out: List[Dict[str, object]] = []
        for j in features:
            col = self.factory._raw[:, j]
            levels = np.unique(col)
            present = np.searchsorted(levels, np.unique(col[rows]))
            labels = self._column_labels(j)
            out.append({
                "feature": self.feature_names[j],
                "values": [labels[i] for i in present.tolist()],
                "merged": int(present.shape[0]) > 1,
            })
        return out

    def _center_dict(self, c: int) -> Dict[str, object]:
        q = int(self.center_schema[c])
        u = int(self.center_set[c])
        n, k = int(self.set_n[u]), int(self.set_k[u])
        feats = self.schema_features[q]
        return {
            "center": int(c),
            "set": u,
            "d": int(self.schema_d[q]),
            "schema_features": [int(j) for j in feats],
            "schema_feature_names": [self.feature_names[j] for j in feats],
            "conditions": self.describe(c),
            "n": n,
            "k": k,
            "purity": (k / n) if n else 0.0,
            "purity_lower": float(clopper_pearson_lower(k, n, float(self.schema_alpha[q]))),
            "alpha_eff": float(self.schema_alpha[q]),
            "share_of_value": (k / self.n_positive) if self.n_positive else 0.0,
            "landscape": {
                "ix": int(self.schema_ix[q]), "iy": int(self.schema_iy_d[q]),
                "iy_all": int(self.schema_iy_all[q]),
            },
        }

    # -- summaries -------------------------------------------------------------
    def _similarity_histogram(self, values: np.ndarray, scope: str) -> Dict[str, object]:
        """
        `values` (one mutual containment per compared centre) binned in
        steps of `_HIST_STEP` from PAIR_FLOOR to 1. Exact identity (1) is
        counted on its own, and everything the pair store does not hold
        (below PAIR_FLOOR) goes to `below_floor`, so the three parts always
        sum to `values.size`.
        """
        v = np.asarray(values, dtype=np.float64).ravel()
        edges = np.round(np.arange(PAIR_FLOOR, 1.0 + _HIST_STEP / 2, _HIST_STEP), 10)
        below = int(np.count_nonzero(v < PAIR_FLOOR - _EPS))
        identical = int(np.count_nonzero(v >= 1.0 - _EPS))
        counts = []
        for i in range(edges.shape[0] - 1):
            lo, hi = edges[i], edges[i + 1]
            last = i == edges.shape[0] - 2
            m = (v >= lo - _EPS) & ((v < hi - _EPS) if not last else (v < 1.0 - _EPS))
            counts.append(int(np.count_nonzero(m)))
        return {
            "scope": scope,
            "floor": PAIR_FLOOR, "step": _HIST_STEP,
            "edges": [float(e) for e in edges],
            "counts": counts, "below_floor": below, "identical": identical,
            "n_centers": int(v.shape[0]),
        }

    def _schema_nests(self, q_a: int, q_b: int) -> bool:
        """Whether one schema's column set contains the other's."""
        s_a = self._schema_sets[int(q_a)]
        s_b = self._schema_sets[int(q_b)]
        return s_a <= s_b or s_b <= s_a

    def _centers_of_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Centres grouped by distinct row set, CSR-style (indptr, indices)."""
        if self._set_centers_indptr is None:
            with self._lock:
                if self._set_centers_indptr is None:
                    order = np.argsort(self.center_set, kind="stable")
                    counts = np.bincount(self.center_set, minlength=self.n_sets)
                    self._set_centers_indices = order
                    self._set_centers_indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return self._set_centers_indptr, self._set_centers_indices

    def _comparison_mask(self, center: int, kinship: Kinship) -> np.ndarray:
        """
        Which centres `center` is compared against under `kinship`: every
        other centre ("all"), or only those whose schema is nested with its
        own ("related") or is not ("unrelated"). The centre itself is always
        excluded, so the mask has at most `n_centers - 1` true entries. The
        nesting test runs once per SCHEMA, not once per centre.
        """
        c = int(center)
        keep = np.ones(self.n_centers, dtype=bool)
        keep[c] = False
        if kinship == "all":
            return keep
        q_c = int(self.center_schema[c])
        n_schemas = len(self.schema_features)
        by_schema = np.fromiter(
            (self._schema_nests(q, q_c) for q in range(n_schemas)), dtype=bool, count=n_schemas,
        )
        nests = by_schema[self.center_schema]
        return keep & (nests if kinship == "related" else ~nests)

    def _nearest_similarity(self, center: int, kinship: Kinship) -> float:
        """
        The mutual containment of `center` with its closest other centre
        under `kinship`, or 0 when it has none at or above PAIR_FLOOR. The
        stored neighbours are walked in descending similarity and the walk
        stops at the first set holding an admissible centre, so the cost is
        a couple of steps rather than a pass over every centre.
        """
        c = int(center)
        q_c = int(self.center_schema[c])
        indptr, indices = self._centers_of_set()
        want = kinship == "related"

        def admissible(v: int) -> bool:
            for b in indices[indptr[v]:indptr[v + 1]].tolist():
                if b == c:
                    continue
                if kinship == "all" or self._schema_nests(int(self.center_schema[b]), q_c) == want:
                    return True
            return False

        u = int(self.center_set[c])
        if admissible(u):  # another centre with the identical row set
            return 1.0
        nb, _, sim = self.neighbours(u)
        if nb.size:
            for i in np.argsort(-sim, kind="stable").tolist():
                if admissible(int(nb[i])):
                    return float(sim[i])
        return 0.0

    def center_similarity_values(self, center: int, kinship: Kinship = "all") -> np.ndarray:
        """
        The mutual containment of `center` with every centre it is compared
        against under `kinship`, one value per such centre. Centres sharing
        its row set score 1; pairs the store does not hold fall below
        PAIR_FLOOR by construction (the store keeps every pair at or above
        the floor).
        """
        c = int(center)
        if c < 0 or c >= self.n_centers:
            raise ValueError(f"center must be in [0, {self.n_centers}), got {center}")
        if kinship not in ("all", "related", "unrelated"):
            raise ValueError(f"kinship must be 'all', 'related' or 'unrelated', got {kinship!r}")
        u = int(self.center_set[c])
        nb, _, sim = self.neighbours(u)
        per_set = np.zeros(self.n_sets, dtype=np.float64)
        if nb.size:
            per_set[nb] = sim
        per_set[u] = 1.0
        return per_set[self.center_set][self._comparison_mask(c, kinship)]

    def nearest_neighbour_histogram(
        self, centers: Optional[np.ndarray] = None, scope: str = "all", kinship: Kinship = "all"
    ) -> Dict[str, object]:
        """
        For every centre of `centers` (all of them when None), the mutual
        containment with its closest OTHER centre: 1 when another centre has
        the identical row set, otherwise the set's nearest stored neighbour,
        or "below the floor" when none reaches PAIR_FLOOR. `centers` chooses
        whose nearest neighbour is counted; `kinship` restricts, per centre,
        which other centres it may be nearest to - so the histogram is drawn
        on the same population as the lists beside it.
        """
        sets = self.center_set if centers is None else self.center_set[np.asarray(centers, dtype=np.int64)]
        if kinship == "all":
            per_set = np.where(self.set_multiplicity > 1, 1.0, self.set_nn_sim)
            return self._similarity_histogram(per_set[sets], scope)
        if kinship not in ("related", "unrelated"):
            raise ValueError(f"kinship must be 'all', 'related' or 'unrelated', got {kinship!r}")
        idx = np.arange(self.n_centers) if centers is None else np.asarray(centers, dtype=np.int64)
        values = np.fromiter(
            (self._nearest_similarity(int(c), kinship) for c in idx.tolist()),
            dtype=np.float64, count=int(idx.shape[0]),
        )
        return self._similarity_histogram(values, scope)

    def _pool_histograms(self, hists: Sequence[Dict[str, object]], scope: str) -> Dict[str, object]:
        """
        The elementwise sum of histograms drawn on the same axis: the
        distribution of every PAIR they cover, rather than one value per
        centre. Two centres of one schema hold disjoint cells, so pooling
        the per-centre strips of one schema counts every pair once.
        """
        out = self._similarity_histogram(np.zeros(0, dtype=np.float64), scope)
        counts = list(out["counts"])
        for h in hists:
            for i, c in enumerate(h["counts"]):
                counts[i] += int(c)
            out["below_floor"] += int(h["below_floor"])
            out["identical"] += int(h["identical"])
            out["n_centers"] += int(h["n_centers"])
        out["counts"] = counts
        return out

    def center_similarity_histogram(self, center: int, kinship: Kinship = "all") -> Dict[str, object]:
        """
        For ONE centre, the mutual containment with every centre it is
        compared against under `kinship` (not just its nearest): the
        distribution behind the "other descriptions" count of that centre,
        so the threshold can be read off the card it applies to.
        """
        return self._similarity_histogram(self.center_similarity_values(center, kinship), "center")

    def _group_members(self, grouping: CenterGrouping, leader: int) -> np.ndarray:
        """Centres of the group represented by distinct set `leader`."""
        sets = np.nonzero(grouping.leader_of_set == int(leader))[0]
        return np.nonzero(np.isin(self.center_set, sets))[0]

    def _group_summary(self, grouping: CenterGrouping, leader: int, members: np.ndarray) -> Dict[str, object]:
        rep = int(self.set_best_center[leader])
        sets = np.unique(self.center_set[members])
        union = np.bitwise_or.reduce(self.set_bits[sets], axis=0)
        n_union = int(_popcount_rows(union))
        k_union = int(_popcount_rows(union & self.z_bits))
        schemas = np.unique(self.center_schema[members])
        dims = np.bincount(self.schema_d[schemas], minlength=MAX_BRANCH_D + 1)[1:]
        feat_count: Dict[str, int] = {}
        for q in schemas.tolist():
            for j in self.schema_features[q]:
                feat_count[self.feature_names[j]] = feat_count.get(self.feature_names[j], 0) + 1
        cells: Dict[Tuple[int, int, int, int], int] = {}
        for q in schemas.tolist():
            key = (int(self.schema_d[q]), int(self.schema_ix[q]), int(self.schema_iy_d[q]), int(self.schema_iy_all[q]))
            cells[key] = cells.get(key, 0) + 1
        others = sets[sets != leader]
        min_sim = float(grouping.sim_to_leader[others].min()) if others.size else 1.0
        rep_features = self.schema_features[int(self.center_schema[rep])]
        return {
            "group": int(leader),
            "kinship_counts": self._kinship_counts(members[members != rep], leader, rep_features),
            "representative": self._center_dict(rep),
            "n_members": int(members.shape[0]),
            "n_distinct": int(sets.shape[0]),
            "n_schemas": int(schemas.shape[0]),
            "min_similarity": min_sim,
            "schemas_by_d": [int(v) for v in dims],
            "features": sorted(
                ({"feature": f, "schemas": c} for f, c in feat_count.items()),
                key=lambda e: (-int(e["schemas"]), str(e["feature"])),
            ),
            "union": {"n": n_union, "k": k_union, "purity": (k_union / n_union) if n_union else 0.0},
            "landscape_cells": [
                {"d": d, "ix": ix, "iy": iy, "iy_all": iya, "schemas": c}
                for (d, ix, iy, iya), c in sorted(cells.items())
            ],
        }

    def groups_page(
        self,
        threshold: float,
        sort: GroupSort = "coverage",
        d: Optional[int] = None,
        cell: Optional[Tuple[Optional[int], int, int]] = None,
        limit: int = 20,
        offset: int = 0,
        kinship: Kinship = "all",
    ) -> Dict[str, object]:
        """
        One page of groups at `threshold`, with the run's summary and the
        nearest-neighbour histogram. `sort="coverage"`: representative's
        share of the value first (then its size, then rank);
        `sort="members"`: the most-described groups first. `d` keeps groups
        with at least one member of that dimensionality; `cell=(d_or_None,
        ix, iy)` keeps groups with a member whose schema falls in that cell
        of the landscape lattice drawn for d (None: all d together).

        `kinship` does NOT filter the grouping - the grouping is the
        clustering itself, and it is paged here - but it does restrict which
        other centres each centre may be nearest to in the histogram, so the
        picture matches the member lists the same filter produces.
        """
        if sort not in ("coverage", "members"):
            raise ValueError(f"sort must be 'coverage' or 'members', got {sort!r}")
        if kinship not in ("all", "related", "unrelated"):
            raise ValueError(f"kinship must be 'all', 'related' or 'unrelated', got {kinship!r}")
        if limit < 1 or offset < 0:
            raise ValueError("limit must be >= 1 and offset >= 0")
        g = self.group(threshold)
        leaders = g.leaders
        # per-centre group (by leader set) and per-group counts
        center_leader = g.leader_of_set[self.center_set]
        n_members = np.bincount(center_leader, minlength=self.n_sets)
        keep = np.ones(leaders.shape[0], dtype=bool)
        if d is not None:
            has_d = np.zeros(self.n_sets, dtype=bool)
            has_d[center_leader[self.schema_d[self.center_schema] == int(d)]] = True
            keep &= has_d[leaders]
        if cell is not None:
            cd, cix, ciy = cell
            q = self.center_schema
            hit = self.schema_ix[q] == int(cix)
            if cd is None:
                hit &= self.schema_iy_all[q] == int(ciy)
            else:
                hit &= (self.schema_d[q] == int(cd)) & (self.schema_iy_d[q] == int(ciy))
            has_c = np.zeros(self.n_sets, dtype=bool)
            has_c[center_leader[hit]] = True
            keep &= has_c[leaders]
        sel = leaders[keep]
        if sort == "coverage":
            order = np.lexsort((self.set_rank[sel], -self.set_n[sel], -self.set_k[sel]))
        else:
            order = np.lexsort((self.set_rank[sel], -self.set_k[sel], -n_members[sel]))
        sel = sel[order]
        page = sel[int(offset): int(offset) + int(limit)]
        groups = [
            self._group_summary(g, int(u), np.nonzero(center_leader == int(u))[0])
            for u in page.tolist()
        ]
        multi = int(np.count_nonzero(n_members[leaders] > 1))
        return {
            "threshold": g.threshold,
            "pair_floor": PAIR_FLOOR,
            "min_rows": self.min_rows,
            "direction": self.direction,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_centers": self.n_centers,
            "n_distinct": self.n_sets,
            "n_pairs": self.n_pairs,
            "n_groups": int(leaders.shape[0]),
            "n_groups_with_duplicates": multi,
            "n_centers_in_duplicate_groups": int(n_members[leaders][n_members[leaders] > 1].sum()),
            "kinship": kinship,
            "histogram": self.nearest_neighbour_histogram(None, "all", kinship),
            "total": int(sel.shape[0]),
            "offset": int(offset),
            "limit": int(limit),
            "sort": sort,
            "groups": groups,
        }

    def _kinship(
        self, c: int, ref_features: Optional[Sequence[int]], n_m: int, n_r: int, inter: int
    ) -> Dict[str, object]:
        """
        How centre `c` stands to the reference centre, as TWO independent
        facts. `lineage` compares the two schemas' column sets - "child"
        (the reference's columns are a strict subset of this centre's),
        "parent", "same_schema", or "unrelated". `relation` compares the
        two ROW sets - "identical", "inside" (this centre's rows are a
        subset of the reference's), "contains", or "crossing".

        The second is measured, never inferred from the first. A child
        schema refines its parent's partition only while no column of
        either was coarsened to fit the grid capacity (Section 4.3); when
        one was, a "child" cell can straddle two parent cells. `consistent`
        is False exactly in that case (lineage and relation disagree), so
        such a pair is reported as the artefact it is instead of being
        presented as a sub-cell of its parent.
        """
        if inter >= n_m and inter >= n_r:
            relation = "identical"
        elif inter >= n_m:
            relation = "inside"
        elif inter >= n_r:
            relation = "contains"
        else:
            relation = "crossing"
        if ref_features is None:
            return {"lineage": None, "relation": relation, "consistent": True}
        s_a = set(self.schema_features[int(self.center_schema[c])])
        s_b = set(int(j) for j in ref_features)
        if s_a == s_b:
            lineage = "same_schema"
        elif s_b < s_a:
            lineage = "child"
        elif s_a < s_b:
            lineage = "parent"
        else:
            lineage = "unrelated"
        if lineage == "child":
            consistent = relation in ("identical", "inside")
        elif lineage == "parent":
            consistent = relation in ("identical", "contains")
        else:
            consistent = True
        return {"lineage": lineage, "relation": relation, "consistent": consistent}

    def _member_entry(self, c: int, ref: int, ref_features: Optional[Sequence[int]] = None) -> Dict[str, object]:
        """
        Centre `c` described against the distinct set `ref` (a group's
        representative, or a branch centre): both one-sided shares, the
        similarity expected by chance, the rescaled similarity, what each
        side holds that the other does not, and - when `ref_features` names
        the reference's schema - the kinship of the two (see `_kinship`).
        """
        u = int(self.center_set[c])
        n_m, k_m = int(self.set_n[u]), int(self.set_k[u])
        n_r, k_r = int(self.set_n[ref]), int(self.set_k[ref])
        inter = self.intersection(u, ref)
        k_inter = self.positives_in_intersection(u, ref) if u != ref else k_r
        sim = inter / max(n_m, n_r)
        # E|R(a) & R(b)| = n_a n_b / N for two row sets of these sizes placed
        # at random (hypergeometric mean), so E[s] = min(n_a, n_b) / N.
        chance = min(n_m, n_r) / self.n_samples if self.n_samples else 0.0
        only_m_n, only_m_k = n_m - inter, k_m - k_inter
        only_r_n, only_r_k = n_r - inter, k_r - k_inter
        entry = self._center_dict(c)
        entry.update({
            "identical": u == ref,
            "similarity": sim,
            "share_in_representative": inter / n_m if n_m else 0.0,
            "share_of_representative": inter / n_r if n_r else 0.0,
            "chance_similarity": chance,
            "similarity_above_chance": ((sim - chance) / (1.0 - chance)) if chance < 1.0 else 0.0,
            "intersection": {"n": inter, "k": k_inter},
            "only_here": {"n": only_m_n, "k": only_m_k, "purity": (only_m_k / only_m_n) if only_m_n else None},
            "only_in_representative": {"n": only_r_n, "k": only_r_k, "purity": (only_r_k / only_r_n) if only_r_n else None},
            "kinship": self._kinship(c, ref_features, n_m, n_r, inter),
        })
        return entry

    def _kinship_mask(
        self, centers: np.ndarray, ref_features: Sequence[int], kinship: Kinship
    ) -> np.ndarray:
        """
        Which of `centers` a `kinship` filter keeps. "related": the two
        schemas' column sets are nested one way or the other - the centre
        comes from the same lineage of the search lattice as the reference.
        "unrelated": they are not - a description built on other columns.
        Schemas of equal dimensionality are always unrelated (two distinct
        column sets of the same size cannot be nested).
        """
        if kinship == "all":
            return np.ones(centers.shape[0], dtype=bool)
        if kinship not in ("related", "unrelated"):
            raise ValueError(f"kinship must be 'all', 'related' or 'unrelated', got {kinship!r}")
        s_b = frozenset(int(j) for j in ref_features)
        related = np.asarray([
            (lambda s_a: s_a <= s_b or s_b <= s_a)(self._schema_sets[int(self.center_schema[c])])
            for c in centers.tolist()
        ], dtype=bool) if centers.size else np.zeros(0, dtype=bool)
        return related if kinship == "related" else ~related

    def _kinship_counts(
        self, centers: np.ndarray, ref: int, ref_features: Sequence[int]
    ) -> Dict[str, int]:
        """
        The `related` / `unrelated` split of `centers` against the
        reference, and how many of them contradict their lineage: a child
        schema whose rows are not inside the parent's, or a parent schema
        whose rows do not contain the child's. Those are the pairs the
        capacity coarsening produced, and the reader is told how many there
        are rather than being shown them as sub-cells.
        """
        if not centers.size:
            return {"related": 0, "unrelated": 0, "inconsistent": 0}
        rel = self._kinship_mask(centers, ref_features, "related")
        n_r = int(self.set_n[ref])
        bad = 0
        for c in centers[rel].tolist():
            u = int(self.center_set[c])
            k = self._kinship(int(c), ref_features, int(self.set_n[u]), n_r, self.intersection(u, ref))
            if not k["consistent"]:
                bad += 1
        return {
            "related": int(np.count_nonzero(rel)),
            "unrelated": int(np.count_nonzero(~rel)),
            "inconsistent": bad,
        }

    def _alternatives(self, u: int, t: float, exclude: int) -> np.ndarray:
        """
        Every centre (other than `exclude`) whose row set is identical to set
        u or has mutual containment >= t with it, most similar first, then
        fewest characteristics, then rank.
        """
        nb, _, sim = self.neighbours(u)
        sets = np.concatenate(([u], nb[sim >= t - _EPS])).astype(np.int64)
        set_sim = np.concatenate(([1.0], sim[sim >= t - _EPS]))
        sim_of = dict(zip(sets.tolist(), set_sim.tolist()))
        cs = np.nonzero(np.isin(self.center_set, sets))[0]
        cs = cs[cs != int(exclude)]
        s_c = np.asarray([sim_of[int(v)] for v in self.center_set[cs].tolist()], dtype=np.float64)
        order = np.lexsort((self.set_rank[self.center_set[cs]], self.schema_d[self.center_schema[cs]], -s_c))
        return cs[order]

    def _listed_centers(self, schema_centers: np.ndarray, t: float, kinship: Kinship) -> np.ndarray:
        """
        Which centres of a schema the interface lists at `t` under `kinship`:
        all of them with no filter, and otherwise only those that have an
        alternative of the chosen kind. The rest are dropped from the cards,
        so the pooled picture separates their pairs from the listed ones.
        """
        if kinship == "all":
            return np.ones(schema_centers.shape[0], dtype=bool)
        out = np.zeros(schema_centers.shape[0], dtype=bool)
        for i, c in enumerate(schema_centers.tolist()):
            alts = self._alternatives(int(self.center_set[c]), t, int(c))
            ref = self.schema_features[int(self.center_schema[c])]
            out[i] = bool(self._kinship_mask(alts, ref, kinship).any())
        return out

    def branch_view(
        self,
        features: Sequence[int],
        threshold: float,
        limit: int = 50,
        offset: int = 0,
        anchor: Optional[int] = None,
        kinship: Kinship = "all",
    ) -> Dict[str, object]:
        """
        The centres of ONE schema (a branch, or a schema opened from the
        landscape) and, for each, every centre of every other schema that
        describes (almost) the same rows: mutual containment >= `threshold`
        with THAT centre directly - not through a group representative, so
        every listed alternative carries the guarantee itself. Centres of
        the schema below `min_rows` are reported with `compared = false`.
        `anchor` (a cell code of the schema) restricts the answer to one
        centre and pages its alternatives with `limit`/`offset`.

        `kinship` restricts the alternatives to those whose schema is
        nested with the branch's ("related") or is not ("unrelated"); the
        unfiltered split is reported per centre as `kinship_counts`
        whichever filter is on. Every derived quantity - `total`,
        `n_identical`, `union`, `simplest`, the schema and characteristic
        counts - is computed on the filtered set, so a card never mixes
        numbers from two different filters.
        """
        if limit < 1 or offset < 0:
            raise ValueError("limit must be >= 1 and offset >= 0")
        if kinship not in ("all", "related", "unrelated"):
            raise ValueError(f"kinship must be 'all', 'related' or 'unrelated', got {kinship!r}")
        t = float(threshold)
        if not (PAIR_FLOOR - _EPS <= t <= 1.0 + _EPS):
            raise ValueError(f"threshold must be in [{PAIR_FLOOR}, 1], got {threshold}")
        combo = tuple(sorted(int(j) for j in features))
        if not combo or len(set(combo)) != len(combo) or len(combo) > MAX_BRANCH_D or any(
            j < 0 or j >= self.factory.n_features for j in combo
        ):
            raise ValueError(f"features must be 1 to {MAX_BRANCH_D} distinct column indices, got {list(features)}")
        codes, n_cells = self.factory.codes(combo)
        mask, n_cell, k_cell, alpha_eff = _select_cells(codes, n_cells, self.z, self.spec)
        cells = np.nonzero(mask)[0]
        # largest centres first
        cells = cells[np.lexsort((cells, -n_cell[cells]))]
        if anchor is not None:
            if int(anchor) not in set(cells.tolist()):
                raise ValueError(f"cell {anchor} is not a centre of schema {list(combo)}")
            cells = np.asarray([int(anchor)], dtype=np.int64)
        q = self.schema_of_features.get(combo)
        order = np.argsort(codes, kind="stable")
        starts = np.concatenate(([0], np.cumsum(n_cell)))
        anchors: List[Dict[str, object]] = []
        for cell in cells.tolist():
            rows_idx = order[starts[cell]:starts[cell + 1]]
            rows = np.zeros(self.n_samples, dtype=bool)
            rows[rows_idx] = True
            n, k = int(n_cell[cell]), int(k_cell[cell])
            base: Dict[str, object] = {
                "cell": int(cell),
                "conditions": self._describe_rows(combo, rows),
                "n": n, "k": k, "purity": k / n if n else 0.0,
                "purity_lower": float(clopper_pearson_lower(k, n, alpha_eff)),
                "alpha_eff": float(alpha_eff),
                "share_of_value": (k / self.n_positive) if self.n_positive else 0.0,
            }
            u = self.set_index.get(np.packbits(rows).tobytes()) if n >= self.min_rows else None
            if u is None or q is None:
                base.update({"compared": False, "total": 0, "alternatives": []})
                anchors.append(base)
                continue
            own = np.nonzero((self.center_schema == q) & (self.center_cell == int(cell)))[0]
            c_self = int(own[0])
            base["histogram"] = self.center_similarity_histogram(c_self, kinship)
            all_alts = self._alternatives(int(u), t, c_self)
            base["kinship_counts"] = self._kinship_counts(all_alts, int(u), combo)
            alts = all_alts[self._kinship_mask(all_alts, combo, kinship)]
            page = alts[int(offset): int(offset) + int(limit)] if anchor is not None else alts[: int(limit)]
            schemas = np.unique(self.center_schema[alts]) if alts.size else np.zeros(0, dtype=np.int64)
            dims = np.bincount(self.schema_d[schemas], minlength=MAX_BRANCH_D + 1)[1:]
            feat_count: Dict[str, int] = {}
            for qq in schemas.tolist():
                for j in self.schema_features[qq]:
                    feat_count[self.feature_names[j]] = feat_count.get(self.feature_names[j], 0) + 1
            simplest = None
            if alts.size:
                best = alts[np.lexsort((self.set_rank[self.center_set[alts]], self.schema_d[self.center_schema[alts]]))[0]]
                simplest = self._member_entry(int(best), int(u), combo)
            union = self.set_bits[int(u)].copy()
            for v in np.unique(self.center_set[alts]).tolist():
                union |= self.set_bits[v]
            n_union = int(_popcount_rows(union))
            k_union = int(_popcount_rows(union & self.z_bits))
            base.update({
                "compared": True,
                "set": int(u),
                "landscape": {
                    "ix": int(self.schema_ix[q]), "iy": int(self.schema_iy_d[q]), "iy_all": int(self.schema_iy_all[q]),
                },
                "total": int(alts.shape[0]),
                "n_identical": int(np.count_nonzero(self.center_set[alts] == int(u))),
                "n_schemas": int(schemas.shape[0]),
                "schemas_by_d": [int(v) for v in dims],
                "features": sorted(
                    ({"feature": f, "schemas": cnt} for f, cnt in feat_count.items()),
                    key=lambda e: (-int(e["schemas"]), str(e["feature"])),
                ),
                "union": {"n": n_union, "k": k_union, "purity": (k_union / n_union) if n_union else 0.0},
                "simplest": simplest,
                "landscape_cells": self._landscape_cells(schemas),
                "offset": int(offset) if anchor is not None else 0,
                "limit": int(limit),
                "alternatives": [self._member_entry(int(c), int(u), combo) for c in page.tolist()],
            })
            anchors.append(base)
        # Pooled over every compared centre of the schema, not only the cards
        # in this response: with `anchor` set the response carries one card,
        # while both branch-level pictures stay branch-level. The pool is split
        # into the centres the cards list and the centres the filter drops, so
        # the drawn picture can show the listed part as the sum of the strips
        # on screen and the rest as what is being hidden.
        schema_centers = np.nonzero(self.center_schema == q)[0] if q is not None else np.zeros(0, dtype=np.int64)
        listed = self._listed_centers(schema_centers, t, kinship)
        by_center: Dict[int, Dict[str, object]] = {}
        if anchor is None:
            for a in anchors:
                if a.get("histogram") is not None:
                    by_center[int(a["cell"])] = a["histogram"]
        strips, strips_listed = [], []
        for c, keep in zip(schema_centers.tolist(), listed.tolist()):
            h = by_center.get(int(self.center_cell[c])) or self.center_similarity_histogram(int(c), kinship)
            strips.append(h)
            if keep:
                strips_listed.append(h)
        return {
            "threshold": t,
            "pair_floor": PAIR_FLOOR,
            "min_rows": self.min_rows,
            "direction": self.direction,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "features": list(combo),
            "feature_names": [self.feature_names[j] for j in combo],
            "d": len(combo),
            "n_centers": int(cells.shape[0]) if anchor is None else int(mask.sum()),
            "coverage": (int(k_cell[mask].sum()) / self.n_positive) if self.n_positive else 0.0,
            # Scoped to this schema (whose nearest neighbour is counted) and to
            # the same kinship as the lists below it (what each may be nearest
            # to), so the picture and the cards always describe one population.
            "histogram": self.nearest_neighbour_histogram(schema_centers, "branch", kinship),
            # The same pairs the cards' own strips draw, pooled: one value per
            # (branch centre, other centre) pair instead of one per centre, so
            # the whole left tail is visible and not only each centre's best.
            "histogram_pairs": self._pool_histograms(strips, "branch_pairs"),
            # The part of it that the cards on screen account for: with no
            # filter this is the whole of it, and with one it is exactly the
            # sum of the strips the reader can see.
            "histogram_pairs_listed": self._pool_histograms(strips_listed, "branch_pairs"),
            "n_centers_listed": int(np.count_nonzero(listed)),
            "kinship": kinship,
            "anchors": anchors,
        }

    def _landscape_cells(self, schemas: np.ndarray) -> List[Dict[str, int]]:
        cells: Dict[Tuple[int, int, int, int], int] = {}
        for q in schemas.tolist():
            key = (int(self.schema_d[q]), int(self.schema_ix[q]), int(self.schema_iy_d[q]), int(self.schema_iy_all[q]))
            cells[key] = cells.get(key, 0) + 1
        return [
            {"d": d, "ix": ix, "iy": iy, "iy_all": iya, "schemas": c}
            for (d, ix, iy, iya), c in sorted(cells.items())
        ]

    def group_detail(
        self, threshold: float, group: int, limit: int = 50, offset: int = 0,
        kinship: Kinship = "all",
    ) -> Dict[str, object]:
        """
        The members of one group (by its representative's set id): every
        centre, most similar to the representative first, with both
        one-sided shares, the similarity expected by chance for two random
        row sets of the same sizes (min(n_a, n_b) / N) with the observed
        similarity rescaled against it, (s - s0) / (1 - s0) - the share of
        the possible excess over chance that is realised, in the manner of
        Cohen's kappa - and what each side holds that the other does not.
        `kinship` restricts the members to those whose schema is nested
        with the representative's ("related") or is not ("unrelated"); the
        unfiltered split is always reported as `kinship_counts`.
        """
        if limit < 1 or offset < 0:
            raise ValueError("limit must be >= 1 and offset >= 0")
        if kinship not in ("all", "related", "unrelated"):
            raise ValueError(f"kinship must be 'all', 'related' or 'unrelated', got {kinship!r}")
        g = self.group(threshold)
        leader = int(group)
        if not (0 <= leader < self.n_sets) or int(g.leader_of_set[leader]) != leader:
            raise ValueError(f"{group} is not a group representative at threshold {g.threshold}")
        members = self._group_members(g, leader)
        rep = int(self.set_best_center[leader])
        members = members[members != rep]
        sims = g.sim_to_leader[self.center_set[members]]
        order = np.lexsort((
            self.set_rank[self.center_set[members]],
            self.schema_d[self.center_schema[members]],
            -sims,
        ))
        members = members[order]
        rep_features = self.schema_features[int(self.center_schema[rep])]
        counts = self._kinship_counts(members, leader, rep_features)
        members = members[self._kinship_mask(members, rep_features, kinship)]
        page = members[int(offset): int(offset) + int(limit)]
        out_members = [self._member_entry(int(c), leader, rep_features) for c in page.tolist()]
        summary = self._group_summary(g, leader, self._group_members(g, leader))
        summary.update({
            "threshold": g.threshold,
            "kinship": kinship,
            "kinship_counts": counts,
            "total": int(members.shape[0]),
            "offset": int(offset),
            "limit": int(limit),
            "members": out_members,
        })
        return summary


def _select_cells(
    codes: np.ndarray, n_cells: int, z: np.ndarray, spec: CenterSpec
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    (centre mask, n per cell, k per cell, per-cell level) of one search
    partition - the selection `vsf.avr._exhaustive_search` applies, shared
    by `collect_centers` and `CenterCatalog.branch_view` so the two can never
    disagree about which cells are centres.
    """
    table = np.bincount(codes * 2 + z, minlength=n_cells * 2).reshape(n_cells, 2)
    n_cell = table.sum(axis=1)
    k_cell = table[:, 1]
    occupied = int(np.count_nonzero(n_cell > 0))
    alpha_eff = spec.effective_alpha(occupied)
    if occupied == 0:
        mask = np.zeros(n_cell.shape, dtype=bool)
    else:
        mask = (k_cell >= min_successes_to_select(n_cell, spec, alpha_eff)) & (n_cell > 0)
    return mask, n_cell, k_cell, float(alpha_eff)


def collect_centers(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    max_d: int = MAX_BRANCH_D,
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    direction: Direction = "presence",
    min_rows: int = 1,
    prune_dependent: bool = False,
) -> CenterCatalog:
    """
    The `CenterCatalog` of the search `discover_branches` would run with
    these arguments: one pass over the same candidate family, selecting
    centres with the same rule as `vsf.avr._exhaustive_search` (and filling
    a `Landscape` from the same pass), keeping those with at least
    `min_rows` rows, then the similarity graph. `prune_dependent` skips the
    candidates that are renamings of a smaller one, exactly as the search
    does, so the catalogue stays the landscape's centre set. Raises ValueError for the
    same unresolvable targets and void thresholds as `compute_landscape`,
    and when the number of distinct centre row sets exceeds
    `MAX_DISTINCT_SETS`.
    """
    if max_d < 1 or max_d > MAX_BRANCH_D:
        raise ValueError(f"max_d must be in [1, {MAX_BRANCH_D}], got {max_d}")
    if int(min_rows) < 1:
        raise ValueError(f"min_rows must be >= 1, got {min_rows}")
    spec = center_spec if center_spec is not None else CenterSpec()
    raw = np.asarray(X, dtype=object)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    prepared = _prepare_search(X, Z, feature_names, positive_class, spec, direction, prune_dependent)
    if prepared is None:
        raise ValueError("no feature columns to search")
    factory, z_binary, names = prepared
    spec = resolve_center_spec(factory, spec, min(max_d, factory.n_features))
    z = z_binary.astype(np.int64)
    N = factory.n_samples
    landscape = Landscape(N, int(z.sum()), direction, names)

    schema_features: List[Tuple[int, ...]] = []
    schema_alpha: List[float] = []
    schema_li: List[int] = []
    c_schema: List[int] = []
    c_cell: List[int] = []
    c_set: List[int] = []
    set_index: Dict[bytes, int] = {}
    set_n: List[int] = []
    set_k: List[int] = []
    set_bits: List[np.ndarray] = []

    for combo, codes, n_cells in factory.iter_candidates(min(max_d, factory.n_features)):
        if n_cells == 0:
            landscape.record(combo, 0, 0, 0)
            continue
        mask, n_cell, k_cell, alpha_eff = _select_cells(codes, n_cells, z, spec)
        li = len(landscape)
        landscape.record(combo, int(k_cell[mask].sum()), int(mask.sum()), int(n_cell[mask].sum()))
        cells = np.nonzero(mask & (n_cell >= int(min_rows)))[0]
        if cells.size == 0:
            continue
        q = len(schema_features)
        schema_features.append(tuple(int(j) for j in combo))
        schema_alpha.append(float(alpha_eff))
        schema_li.append(li)
        order = np.argsort(codes, kind="stable")
        starts = np.concatenate(([0], np.cumsum(n_cell)))
        for cc in cells.tolist():
            member = np.zeros(N, dtype=bool)
            member[order[starts[cc]:starts[cc + 1]]] = True
            packed = np.packbits(member)
            key = packed.tobytes()
            u = set_index.get(key)
            if u is None:
                u = len(set_n)
                if u >= MAX_DISTINCT_SETS:
                    raise ValueError(
                        f"more than {MAX_DISTINCT_SETS} distinct centre row sets: the pairwise "
                        "comparison is quadratic in this number. Raise min_rows (or min_samples) "
                        "to restrict the comparison to larger centres."
                    )
                set_index[key] = u
                set_n.append(int(n_cell[cc]))
                set_k.append(int(k_cell[cc]))
                set_bits.append(packed)
            c_schema.append(q)
            c_cell.append(int(cc))
            c_set.append(u)

    n_bytes = (N + 7) // 8
    return CenterCatalog(
        factory=factory,
        raw_X=raw,
        z=z,
        spec=spec,
        direction=direction,
        feature_names=names,
        min_rows=int(min_rows),
        landscape=landscape.freeze(),
        schema_features=schema_features,
        schema_alpha=np.asarray(schema_alpha, dtype=np.float64),
        schema_landscape_index=np.asarray(schema_li, dtype=np.int64),
        center_schema=np.asarray(c_schema, dtype=np.int64),
        center_cell=np.asarray(c_cell, dtype=np.int64),
        center_set=np.asarray(c_set, dtype=np.int64),
        set_n=np.asarray(set_n, dtype=np.int64),
        set_k=np.asarray(set_k, dtype=np.int64),
        set_bits=(np.vstack(set_bits) if set_bits else np.zeros((0, n_bytes), dtype=np.uint8)),
        set_index=set_index,
    )
