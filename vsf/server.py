"""
vsf.server: `vsf.serve(df)` — an installable, in-process interactive VSF web
application for an ARBITRARY pandas DataFrame with one call, the way
`streamlit.run`/`gradio.Interface.launch` do.

(see Project_Master_Document.md for the full
revision history): the legacy top-level `server.py` demo script (a
single-CSV, single-dataset script duplicating this module's request
dispatch for the bundled mushroom dataset) has been deleted outright rather
than updated — it could not survive the `AVREngine` -> `BranchEngine` API
change without a rewrite indistinguishable from this module, so keeping both
was pure duplication. `vsf.serve(df)` is now the ONLY live-server entry
point; `tests/test_server.py` (which asserted on the legacy script's
internals) was deleted alongside it.

Removed relative to v1.0 (all four are the same product decisions recorded
in Project_Master_Document.md, applied here to the live-server
surface):
  - `/api/mine_center` (dirty-center conjunctive-filter mining) — its sole
    implementation, `vsf.mining.mine_dirty_center`, no longer exists.
  - `/api/top_columns` (Auto-Discovery / Universal Propositional Screening)
    — its implementation, `vsf.mining.compute_top_insights`, no longer
    exists; the user now picks the target column and criterion directly
    from the dropdown populated by `/api/columns`, never from a "here's
    what's interesting" catalog.
  - `/api/graph_inference` and `/api/mine_graph_links` (Graph Inference /
    Knowledge-Base chain mining) and the `graph.html` /
    `static/{css,js}/graph_reasoning.*` assets that rendered them — the
    reasoning-graph feature is gone, not just its route.
  - `composite_target` support inside `/api/analyze` (the composite AND
    -filter target builder) — replaced in 2026-09 by `also` (see the end
    of this docstring), which does what the old builder did not: remove
    the target's columns from the feature space and anchor the certificate
    to the conjunction's base rate.

`/api/analyze` itself changed shape, not just scope: it used to fit ONE
`AVRResult` (a single adaptively-chosen `d_star`) per request. It now runs
Independent Branch Discovery (`vsf.avr.discover_branches`,
Project_Master_Document.md Section 4) and returns UP TO `MAX_BRANCH_D`
independently-found branches in one response — one visualization payload
per dimensionality — so the frontend's branch selector can switch between
them instantly, client-side, with no further request (mirroring how `vsf.dashboard.
export_full_dashboard` already bakes all branches into its static export).

Per-instance isolation: this server keeps its dataframe, translations
table, default target, and analyze-response cache on the constructed HTTP
server instance (`_VSFServer.df` / `.translations` / `.default_target` /
`.cache_lock` / `.last_params` / `.last_res`), not as module-level globals,
so two concurrently-running `serve()` calls in one process (e.g. sequential
notebook cells with different dataframes) never share state.

Static assets are served from the PACKAGED `vsf/webapp/` resources via
`importlib.resources` (mirroring `vsf/dashboard.py`'s `_read_template`
pattern for `vsf/templates/`) rather than cwd-relative file serving, so this
works from an installed `vsf` package invoked from any caller directory.

Dataset-agnosticism: `default_target` is `"class"` if the dataframe has a
column literally named "class" (matching the bundled mushroom demo's
convention), otherwise the dataframe's FIRST column. This is only the
column shown when the page first loads — the frontend's catalog lets the
user click any other column afterward (`/api/columns` reports
`default_target` and `vsf/webapp/static/js/app.js` reads it instead of
hardcoding "class"; see that file's `init()`).

Global Pattern Scan (`/api/scan/start` / `/api/scan/status` /
`/api/scan/cancel`, added after v2.0): scans every column of `df`, treats
each of its observed values as a One-vs-Rest binary criterion (exactly
`/api/analyze`'s `criterion` shape), runs Independent Branch Discovery
against every OTHER column for it, and keeps the (column, value) pair only
if the MAXIMUM certified COVERAGE across its up-to-4 branches is at least a
caller-chosen `coverage_threshold` AND the pair survives Benjamini-Hochberg
FDR control across every pair scanned in that run (see `_run_dataset_scan`'s
docstring for why coverage is the filter: an association statistic can read
high on a target with zero certified centres, which is precisely what this
scan must not select for).
A sweep of k targets at a nominal alpha yields ~alpha*k spurious "patterns"
by construction, so the FDR control is not optional; `fdr_q` sets the rate.

This is still NOT a return of v1.0's removed Auto-Discovery/Universal
Propositional Screening (see the master doc for why that was
cut): v1.0 used permutation tests to GATE greedy feature selection inside a
single analysis, whereas the scan tests already-selected, exhaustively
searched branches and corrects only for the multiplicity the scan itself
creates. That looping makes it genuinely expensive
— one exhaustive 1D-4D search per COLUMN (`discover_branches_by_value`
scores every value of the column in the same enumeration, since the
candidate partitions do not depend on the target; the result per
(column, value) pair is identical to a per-pair `discover_branches` call),
plus a per-pair reporting stage — so it runs in a background thread
(`_run_dataset_scan`), polled via `/api/scan/status` rather than returned
synchronously, with `/api/scan/cancel` to stop early.
Only one scan may run at a time per server instance; state lives on
`_VSFServer.scan_job`/`.scan_lock`/`.scan_cancel_event`, so — like
everything else in this module — two concurrently-running `serve()` calls
never share a scan.

Analyze-response cache and sibling prefetch (2026-09): `/api/analyze`
responses are cached serialised, several at a time (`_VSFServer.analyze_cache`,
LRU keyed by target, value and certificate parameters), identical concurrent
requests are computed once (`_VSFServer.inflight`), and after a
criterion-mode analysis the other values of the same column are computed in
a background thread (`_prefetch_sibling_values`, one shared exhaustive search
via `vsf.avr.iter_branches_by_value`) so the user's next clicks in that
column are served from the cache. `serve(prefetch=False)` disables the
prefetch; responses are byte-identical either way.

Solution landscape (Section 4.9): `/api/landscape` (the 10 x 10 count
lattice of every scored schema at the request's certificate),
`/api/landscape/cell` (the schemas of one lattice cell, paged),
`/api/landscape/curves` (the per-d envelope over the purity floor -
`vsf.avr.compute_tau_curves`; cached per parameters WITHOUT tau, which is
the abscissa) and `/api/landscape/at` (the schemas of one dimensionality
at a given floor within one ten-percent coverage category - the
click-through of a curve point; computes the landscape at that floor).
`/api/analyze` with `features=[...]` opens one schema (`report_schema`).

Dataset screen (Section 4.12): `/api/screen` - one profile per column and
every pair of columns one of which (nearly) determines the other, computed
on the feature columns alone so it can be read BEFORE a target is chosen,
plus the target leakage report once a target is named. The reader's
exclusions travel as `drop=[column, ...]` on every analysis endpoint and
`prune=true` skips candidates that are renamings of a smaller schema; both
are part of the analysis cache key and are echoed in `/api/analyze`.

Redundant centres (Section 4.11): `/api/centers/groups` (the centres of
every scored schema grouped by mutual containment at a user threshold -
one page of representatives, the run summary and the nearest-neighbour
histogram; `vsf.redundancy`) and `/api/centers/group` (the members of one
group, paged) and `/api/centers/branch` (the centres of one schema, each
with every centre of another schema that holds almost the same rows). The centre catalogue and its similarity graph are computed
once per (landscape key, min_rows) and cached; the threshold only regroups.

Rules (Section 4.15): `/api/rules` lists every occupied cell of the given
schemas (the winning 1D-4D schemas of the current analysis) as a
conjunctive rule with its rows, purity, certificate status and
generalisations (`vsf.rules.enumerate_rules`); cached per parameters,
schemas and `min_rows`. The filtering, the purity bars and the cards are
client-side.

Composite targets (Section 4.10): `/api/analyze` and the landscape family
accept `also=[[column, value], ...]` - up to two further (column, value)
pairs conjoined with the primary (target, criterion). The indicator
searched is the AND of all pairs; every column of the target is removed
from the feature space (`_Target`, `_resolve_target`, `_target_arrays`).
`/api/target` reports a target's rows, share and remaining features before
any search. This supersedes the v1.0 `composite_target` builder removed
above: that one neither excluded the target's columns from the features
nor anchored the certificate to the conjunction's base rate.

Validation (Section 4.14): `/api/validate` checks the analysis the page
shows with the post-selection machinery of `vsf.selective` - a certificate
that holds for the reported schemas (`method`: "split" or
"family_bonferroni") and the nested cross-validated coverage with schema
stability (`repeats` x 5 folds). It costs several full searches, so it runs
in a background thread: the first request starts the job and every request
with the same body returns its current state (`status`, `progress`, and
`result` once done). One validation runs at a time per server; a request
for a different analysis while one runs gets 409. Finished results are
kept per request key.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
import urllib.parse
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass
from importlib import resources
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as _np
import pandas as pd

from . import vis as _vis
from .avr import (
    DEFAULT_N_PERMUTATIONS,
    MAX_BRANCH_D,
    discover_branches,
    compute_landscape,
    compute_tau_curves,
    discover_branches_by_value,
    iter_branches_by_value,
    report_schema,
    select_branch_dimensionality,
)
from .centers import CenterSpec
from .metrics import benjamini_hochberg
from .screen import DEFAULT_MIN_STRENGTH, screen_dataset, target_report
from .selective import (
    certification_passes,
    certify_discovery,
    nested_crossvalidation,
    nested_passes,
)
from .avr import _prepare_search
from .redundancy import DEFAULT_GROUP_THRESHOLD, CenterCatalog, collect_centers
from .rules import enumerate_rules, union_coverage
from .vis import Translations, catalog_from_dataframe, prepare_visualization_payload

__all__ = ["serve"]

# Branch Discovery ceiling for the live `/api/analyze` endpoint — matches
# `vsf.avr.MAX_BRANCH_D`, the hard ceiling this display's spatial encoding
# supports (3 coordinate axes + 1 time/frame axis; see `vsf.avr`'s module
# docstring). `discover_branches` is fully deterministic (no permutation
# testing, no randomness at all — see that module's docstring), so unlike
#: Permutation replicates per (column, value) pair in a Global Pattern Scan.
#: 199 rather than `vsf.avr.DEFAULT_N_PERMUTATIONS` because the scan pays this
#: cost once per pair over a family that is routinely hundreds of pairs wide;
#: it floors the per-pair p-value at 0.005, still an order of magnitude below
#: any FDR rate a user would set.
_SCAN_N_PERMUTATIONS = 199
#: Fixed so a scan is reproducible run-to-run on unchanged data.
_SCAN_RANDOM_STATE = 0

# v1.0 there is no alpha/vir_threshold to pin
# here.
_MAX_D = MAX_BRANCH_D

#: Analyze-response cache: number of serialised `/api/analyze` responses
#: kept per server instance, and a ceiling on their total size. A response
#: is 10-20 MB of JSON for a 10 000-point render of four branches; the
#: cache holds the SERIALISED bytes, not the Python payload (which is
#: several times larger in memory), so a user stepping back and forth
#: between the values of one column pays the computation once per value.
_ANALYZE_CACHE_MAX_ENTRIES = 16
_ANALYZE_CACHE_MAX_BYTES = 512 * 1024 * 1024
#: Solution landscapes kept per server (each holds every scored candidate
#: of one search: a few MB at M ~ 50).
_LANDSCAPE_CACHE_MAX_ENTRIES = 8
#: Tau-curves (`vsf.avr.compute_tau_curves`) kept per server, keyed by the
#: landscape parameters WITHOUT tau (the curve is the dependence on tau).
_CURVES_CACHE_MAX_ENTRIES = 8
#: Centre catalogues (`vsf.redundancy.CenterCatalog`) kept per server: each
#: holds the bit-packed rows of every distinct centre and its similarity
#: graph (tens of MB at S ~ 20 000 distinct centres, N ~ 8 000 rows).
_CENTERS_CACHE_MAX_ENTRIES = 4
#: Dataset screens kept per server (one per set of screened columns).
_SCREEN_CACHE_MAX_ENTRIES = 8
#: Finished `/api/validate` results kept per server.
_VALIDATE_CACHE_MAX_ENTRIES = 16
#: Repetitions of the 5-fold split `/api/validate` accepts (each is five
#: full searches).
_VALIDATE_MAX_REPEATS = 10
_VALIDATE_FOLDS = 5
#: Certified cells listed per branch in a validation response.
_VALIDATE_MAX_CELLS = 50
#: Page sizes the redundant-centres endpoints accept at most.
_CENTERS_MAX_LIMIT = 500
#: Rule listings (`vsf.rules.enumerate_rules`) kept per server, keyed by the
#: analyze parameters plus the schemas listed and `min_rows`.
_RULES_CACHE_MAX_ENTRIES = 16
#: Largest number of schemas a tau-curve request may exclude (Rules view).
_CURVES_MAX_EXCLUDE = 200
#: Largest number of schemas `/api/rules` lists at once: the schemas on the
#: envelope (one per floor and dimensionality, tens on a real dataset).
_RULES_MAX_SCHEMAS = 400
#: Largest rule set `/api/rules/union` combines.
_RULES_UNION_MAX = 20000
#: Grid step of the tau-curves, in percent of purity.
_CURVES_STEP_PCT = 1.0
#: Largest number of (column, value) conjuncts a composite target may have:
#: the primary (target, criterion) plus up to two `also` pairs. Base rates
#: fall multiplicatively with every conjunct, and beyond three the certified
#: cells are single objects on every dataset the framework is evaluated on.
_MAX_TARGET_CONJUNCTS = 3


@dataclass(frozen=True)
class _Target:
    """
    A resolved analysis target: the primary column with its optional value,
    plus the extra (column, value) conjuncts of a COMPOSITE target
    (`also`, sorted, possibly empty). The indicator searched is the
    conjunction of all pairs - one more 0/1 column, which is all the search
    ever sees - and every column named in the conjunction is removed from
    the feature space, otherwise the search would trivially "find" the
    target's own columns as its schema.
    """
    target_col: str
    criterion: Optional[str]
    also: Tuple[Tuple[str, str], ...] = ()

    @property
    def columns(self) -> List[str]:
        return [self.target_col] + [c for c, _ in self.also]

    @property
    def pairs(self) -> List[Tuple[str, str]]:
        assert self.criterion is not None
        return [(self.target_col, self.criterion)] + list(self.also)

    def params(self) -> Dict[str, Any]:
        """The target's part of an analyze cache key."""
        out: Dict[str, Any] = {"target_col": self.target_col, "criterion": self.criterion}
        if self.also:
            out["also"] = self.also
        return out


def _resolve_target(df: pd.DataFrame, req: Dict[str, Any], default_target: str) -> Tuple[Optional[_Target], Optional[str]]:
    """
    Reads `target`, `criterion` and `also` (a list of [column, value]
    pairs) from a request body and validates them against `df`: every
    column must exist, the conjunct columns must be distinct from each
    other and from the target, every value must be one the column actually
    takes, and a conjunction needs an explicit `criterion`. Returns
    (target, None) or (None, error message).
    """
    target_col = req.get("target", default_target)
    if target_col not in df.columns:
        return None, f"target {target_col!r} is not a column of this dataset"
    criterion = req.get("criterion", None)
    criterion = None if criterion is None else str(criterion)
    raw_also = req.get("also", None) or []
    if not isinstance(raw_also, list):
        return None, "also must be a list of [column, value] pairs"
    also: List[Tuple[str, str]] = []
    for item in raw_also:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None, "also must be a list of [column, value] pairs"
        col, val = str(item[0]), str(item[1])
        if col not in df.columns:
            return None, f"also: {col!r} is not a column of this dataset"
        if col == target_col or any(col == c for c, _ in also):
            return None, f"also: column {col!r} appears twice in the target"
        if not (df[col].astype(str) == val).any():
            return None, f"also: {col!r} never takes the value {val!r}"
        also.append((col, val))
    if also and criterion is None:
        return None, "a composite target needs an explicit criterion for the primary column"
    if len(also) + 1 > _MAX_TARGET_CONJUNCTS:
        return None, f"a target may have at most {_MAX_TARGET_CONJUNCTS} (column, value) conjuncts"
    return _Target(target_col, criterion, tuple(sorted(also))), None


def _default_multiplicity(rule: str) -> str:
    """
    The multiplicity a request gets when it names none: a certificate
    (`rule="certified"`) is corrected over the whole search family, because
    the per-schema correction does not hold for a schema the search chose
    (Project_Master_Document.md Sections 4.5 and 4.14). The per-schema
    correction must be asked for explicitly (`multiplicity="bonferroni"`).
    `rule="purity"` certifies nothing; its per-cell intervals keep the
    per-schema level.
    """
    return "family" if rule == "certified" else "bonferroni"


def _parse_certificate(
    req: Dict[str, Any], min_samples_key: str = "min_samples",
) -> Tuple[Optional[CenterSpec], Optional[str]]:
    """
    The `CenterSpec` of a request - `tau`, `alpha`, `rule`, `multiplicity`
    and `min_samples` - or (None, error). The spec is returned unresolved;
    every library entry point resolves the family size itself (cached).
    """
    try:
        tau = float(req.get("tau", 0.90))
        alpha = float(req.get("alpha", 0.05))
        min_samples = int(req.get(min_samples_key, 1))
    except (TypeError, ValueError):
        return None, "tau, alpha and min_samples must be numbers"
    rule = req.get("rule", "purity")
    if rule not in ("purity", "certified"):
        return None, f"rule must be 'purity' or 'certified', got {rule!r}"
    multiplicity = req.get("multiplicity", None)
    if multiplicity is None:
        multiplicity = _default_multiplicity(rule)
    if multiplicity not in ("family", "bonferroni"):
        return None, f"multiplicity must be 'family' or 'bonferroni', got {multiplicity!r}"
    if multiplicity == "family" and rule != "certified":
        return None, "multiplicity='family' needs rule='certified'"
    try:
        return CenterSpec(
            tau=tau, alpha=alpha, rule=rule, min_samples=min_samples,
            multiplicity=multiplicity,
        ), None
    except ValueError as exc:
        return None, str(exc)


def _certificate_params(spec: CenterSpec) -> Dict[str, Any]:
    """The certificate's part of an analysis cache key (resolution excluded)."""
    return {
        "tau": spec.tau, "alpha": spec.alpha, "rule": spec.rule,
        "min_samples": spec.min_samples, "multiplicity": spec.multiplicity,
    }


def _certificate_payload(spec: CenterSpec) -> Dict[str, Any]:
    """
    What a response says about the certificate it was computed under. For a
    resolved family spec this includes the family size, the per-cell level
    and the smallest cell that can be certified at all (a fully pure one).
    """
    out: Dict[str, Any] = {
        "tau": spec.tau,
        "alpha": spec.alpha,
        "rule": spec.rule,
        "min_samples": spec.min_samples,
        "method": spec.method,
        "multiplicity": spec.multiplicity,
        "family_tests": spec.family_tests,
        "per_cell_level": None,
        "min_certifiable_rows": None,
        "valid_after_search": spec.rule == "certified" and spec.multiplicity == "family",
    }
    if spec.multiplicity == "family" and spec.family_tests is not None:
        level = spec.effective_alpha(1)
        out["per_cell_level"] = level
        # A cell of n rows, all of the value, has p = tau ** n.
        out["min_certifiable_rows"] = int(_np.ceil(_np.log(level) / _np.log(spec.tau) - 1e-12))
    return out


def _parse_screen_options(
    df: pd.DataFrame, req: Dict[str, Any], target: Optional["_Target"] = None,
) -> Tuple[Optional[Tuple[str, ...]], bool, Optional[str]]:
    """
    (`drop`, `prune`, error) of a request: the columns the reader excluded in
    the dataset screen (Section 4.12) and whether the search skips candidates
    that are renamings of a smaller one. `drop` is returned sorted and
    de-duplicated so that two requests naming the same columns share a cache
    entry; a column of the target itself is accepted and ignored (it is
    already out of the feature space), an unknown column is an error, and
    excluding EVERY feature is an error rather than an empty search.
    """
    raw = req.get("drop", None)
    prune = bool(req.get("prune", False))
    if raw is None:
        return (), prune, None
    if not isinstance(raw, (list, tuple)):
        return None, prune, "drop must be a list of column names"
    names: List[str] = []
    for item in raw:
        col = str(item)
        if col not in df.columns:
            return None, prune, f"drop: {col!r} is not a column of this dataset"
        names.append(col)
    drop = tuple(sorted(dict.fromkeys(names)))
    if target is not None:
        remaining = [c for c in df.columns if c not in target.columns and c not in set(drop)]
        if not remaining:
            return None, prune, "drop: every feature column would be excluded"
    return drop, prune, None


def _target_arrays(
    df: pd.DataFrame, target: _Target, drop: Sequence[str] = (),
) -> Tuple[pd.DataFrame, "_np.ndarray", "_np.ndarray"]:
    """
    (X_df, Z, sort_Z) for a target: the feature frame with every target
    column removed - and with every column of `drop` removed as well, the
    reader's own exclusions from the dataset screen (Section 4.12) - the
    indicator (0/1 conjunction of all pairs, or the raw column when no
    criterion is given) and the raw primary column used for stable ordering
    of the display.
    """
    if target.criterion is None:
        Z = df[target.target_col].values
    else:
        mask = _np.ones(len(df), dtype=bool)
        for col, val in target.pairs:
            mask &= (df[col].astype(str) == val).values
        Z = mask.astype(int)
    X_df = df.drop(columns=target.columns)
    extra = [c for c in dict.fromkeys(drop) if c in X_df.columns]
    if extra:
        X_df = X_df.drop(columns=extra)
    return X_df, Z, df[target.target_col].values


def _target_display(target: _Target, translations: Optional[Translations]) -> str:
    """`col = value` for a single pair; `col = value ∧ col2 = value2` for a conjunction."""
    if target.criterion is None:
        return target.target_col
    return " \u2227 ".join(
        f"{_vis.humanize_col(c, translations)} = {_vis.humanize_val(c, v, translations)}"
        for c, v in target.pairs
    )

# Static asset content-types served from the packaged `vsf.webapp` resources.
_STATIC_ROUTES: Dict[str, tuple] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/static/css/styles.css": ("static/css/styles.css", "text/css; charset=utf-8"),
    "/static/js/app.js": ("static/js/app.js", "application/javascript; charset=utf-8"),
}


def _read_webapp_asset(relative_path: str) -> str:
    """
    Reads a packaged `vsf/webapp/...` asset via `importlib.resources`, the
    same mechanism `vsf/dashboard.py`'s `_read_template` uses for
    `vsf/templates/` — works whether `vsf` is installed as a wheel/sdist or
    run from an editable checkout, unlike cwd-relative file reads.
    """
    resource = resources.files("vsf.webapp")
    for part in relative_path.split("/"):
        resource = resource.joinpath(part)
    return resource.read_text(encoding="utf-8")


def _default_target(df: pd.DataFrame) -> str:
    """
    `"class"` if present (matching the bundled mushroom demo's convention),
    else the dataframe's first column. This is only the page's INITIAL
    target; the frontend catalog lets the user click any other column
    afterward.
    """
    return "class" if "class" in df.columns else str(df.columns[0])


class _VSFServer(http.server.ThreadingHTTPServer):
    """
    A `ThreadingHTTPServer` carrying its OWN per-instance dataframe,
    translations table, default target, and analyze-response cache — see
    the module docstring's isolation note. `VSFRequestHandler` below reads
    all of this via `self.server.*` rather than module-level globals, so
    multiple `serve()` calls in one process never share state.
    """

    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        df: pd.DataFrame,
        translations: Optional[Translations],
        prefetch: bool = True,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.df = df
        self.translations = translations
        self.default_target = _default_target(df)
        self.cache_lock = threading.Lock()
        self.last_params: Optional[Dict[str, Any]] = None
        self.last_branches_payload: Optional[Dict[str, Any]] = None
        # Multi-entry LRU of serialised responses keyed by the full request
        # key (target, criterion, certificate parameters) - see
        # `_ANALYZE_CACHE_MAX_ENTRIES`. `last_params`/`last_branches_payload`
        # above still mirror the most recent one for callers that inspect
        # it (tests, `vsf.dashboard`-style consumers).
        self.analyze_cache: "OrderedDict[Tuple[Any, ...], bytes]" = OrderedDict()
        self.analyze_cache_bytes = 0
        # Solution landscapes (`vsf.avr.Landscape`) keyed like the analyze
        # cache minus `features`; computed on first request (the ranking
        # pass alone), kept for the lattice and its cell listings.
        self.landscape_cache: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
        # Tau-curves per (target, criterion, rule, alpha, min_samples,
        # direction) - the same analyze key with tau removed.
        self.curves_cache: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
        # Centre catalogues per (landscape key, min_rows). Built under
        # `centers_lock`, so two requests for one key (e.g. the group list
        # and a member list opened at once) compute it once.
        self.centers_cache: "OrderedDict[Tuple[Any, ...], CenterCatalog]" = OrderedDict()
        # Rule listings of the winning schemas (`/api/rules`).
        self.rules_cache: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
        self.centers_lock = threading.Lock()
        # Dataset screens (`vsf.screen.screen_dataset`) per (columns, min_strength).
        self.screen_cache: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
        # Requests being computed right now, so two clicks on the same key
        # (or a click racing the background prefetch of that key) compute it
        # once: the second waits on the first's Event and reads the cache.
        self.inflight: Dict[Tuple[Any, ...], threading.Event] = {}
        # Background prefetch of the OTHER values of the column the user just
        # analysed (`_prefetch_sibling_values`). One worker at a time; a new
        # analyze request for a different column cancels the running one.
        self.prefetch_enabled = prefetch
        self.prefetch_lock = threading.Lock()
        self.prefetch_thread: Optional[threading.Thread] = None
        self.prefetch_cancel: Optional[threading.Event] = None
        # Global Pattern Scan state (see module docstring). `scan_job` is
        # None until the first scan starts; `scan_cancel_event` is a fresh
        # threading.Event() per scan, set by `/api/scan/cancel` and polled
        # by `_run_dataset_scan`'s background thread between pairs.
        self.scan_lock = threading.Lock()
        self.scan_job: Optional[Dict[str, Any]] = None
        self.scan_cancel_event: Optional[threading.Event] = None
        # `/api/validate` jobs by request key (running or finished, LRU over
        # finished ones) and the key of the one running now, if any.
        self.validate_lock = threading.Lock()
        self.validate_jobs: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
        self.validate_running: Optional[Tuple[Any, ...]] = None

    # -- analyze-response cache ---------------------------------------------
    def cache_get(self, key: Tuple[Any, ...]) -> Optional[bytes]:
        """Serialised response for `key`, marking it most recently used."""
        with self.cache_lock:
            body = self.analyze_cache.get(key)
            if body is not None:
                self.analyze_cache.move_to_end(key)
            return body

    def cache_put(self, key: Tuple[Any, ...], params: Dict[str, Any],
                  payload: Dict[str, Any], body: bytes) -> None:
        with self.cache_lock:
            old = self.analyze_cache.pop(key, None)
            if old is not None:
                self.analyze_cache_bytes -= len(old)
            self.analyze_cache[key] = body
            self.analyze_cache_bytes += len(body)
            while self.analyze_cache and (
                len(self.analyze_cache) > _ANALYZE_CACHE_MAX_ENTRIES
                or self.analyze_cache_bytes > _ANALYZE_CACHE_MAX_BYTES
            ):
                _, evicted = self.analyze_cache.popitem(last=False)
                self.analyze_cache_bytes -= len(evicted)
            self.last_params = params
            self.last_branches_payload = payload

    def claim(self, key: Tuple[Any, ...]) -> Tuple[Optional[bytes], Optional[threading.Event], bool]:
        """
        Atomically: a cached body if present; otherwise the Event of a
        computation already in flight for `key`; otherwise a fresh Event
        registered for `key` with `owner=True`, meaning the caller must
        compute it and call `release`.
        """
        with self.cache_lock:
            body = self.analyze_cache.get(key)
            if body is not None:
                self.analyze_cache.move_to_end(key)
                return body, None, False
            event = self.inflight.get(key)
            if event is not None:
                return None, event, False
            event = threading.Event()
            self.inflight[key] = event
            return None, event, True

    def release(self, key: Tuple[Any, ...], event: threading.Event) -> None:
        with self.cache_lock:
            if self.inflight.get(key) is event:
                del self.inflight[key]
        event.set()

    def get_landscape(self, params: Dict[str, Any], center_spec: CenterSpec, target: _Target):
        """The `Landscape` for an analyze parameter set, computed once."""
        key = _analyze_key(params)
        with self.cache_lock:
            hit = self.landscape_cache.get(key)
            if hit is not None:
                self.landscape_cache.move_to_end(key)
                return hit
        X_df, Z, _ = _target_arrays(self.df, target, params.get("drop", ()))
        landscape = compute_landscape(
            X_df.values, Z, feature_names=list(X_df.columns), max_d=_MAX_D,
            positive_class=(1 if target.criterion is not None else None),
            center_spec=center_spec, direction=params["direction"],
            prune_dependent=bool(params.get("prune", False)),
        )
        with self.cache_lock:
            self.landscape_cache[key] = landscape
            while len(self.landscape_cache) > _LANDSCAPE_CACHE_MAX_ENTRIES:
                self.landscape_cache.popitem(last=False)
        return landscape


    def get_screen(self, columns: Tuple[str, ...], min_strength: float):
        """The `DatasetScreen` of these columns of `df`, computed once per (columns, min_strength)."""
        key = (columns, round(float(min_strength), 6))
        with self.cache_lock:
            hit = self.screen_cache.get(key)
            if hit is not None:
                self.screen_cache.move_to_end(key)
                return hit
        sub = self.df[list(columns)]
        screen = screen_dataset(sub.values, list(sub.columns), min_strength=float(min_strength))
        with self.cache_lock:
            self.screen_cache[key] = screen
            while len(self.screen_cache) > _SCREEN_CACHE_MAX_ENTRIES:
                self.screen_cache.popitem(last=False)
        return screen

    def get_center_catalog(
        self, params: Dict[str, Any], center_spec: CenterSpec, target: _Target, min_rows: int,
    ) -> CenterCatalog:
        """The `CenterCatalog` for a landscape parameter set and `min_rows`, computed once."""
        key = _analyze_key(params) + (("min_rows", repr(int(min_rows))),)
        with self.centers_lock:
            hit = self.centers_cache.get(key)
            if hit is not None:
                self.centers_cache.move_to_end(key)
                return hit
            X_df, Z, _ = _target_arrays(self.df, target, params.get("drop", ()))
            catalog = collect_centers(
                X_df.values, Z, feature_names=list(X_df.columns), max_d=_MAX_D,
                positive_class=(1 if target.criterion is not None else None),
                center_spec=center_spec, direction=params["direction"], min_rows=int(min_rows),
                prune_dependent=bool(params.get("prune", False)),
            )
            self.centers_cache[key] = catalog
            while len(self.centers_cache) > _CENTERS_CACHE_MAX_ENTRIES:
                self.centers_cache.popitem(last=False)
            return catalog

    def get_rules(
        self, params: Dict[str, Any], center_spec: CenterSpec, target: _Target,
        schemas: List[List[int]], min_rows: int,
    ) -> Dict[str, Any]:
        """The rules of `schemas` (`enumerate_rules`) for a parameter set, computed once."""
        key = _analyze_key(params) + (("schemas", repr(schemas)), ("min_rows", repr(int(min_rows))))
        with self.cache_lock:
            hit = self.rules_cache.get(key)
            if hit is not None:
                self.rules_cache.move_to_end(key)
                return hit
        X_df, Z, _ = _target_arrays(self.df, target, params.get("drop", ()))
        if target.criterion is None:
            raise ValueError("the rules view needs an explicit target value (criterion)")
        Z = _np.asarray(Z).astype(int)
        if params["direction"] == "absence":
            Z = 1 - Z
        out = enumerate_rules(
            X_df.values, Z, schemas, list(X_df.columns), center_spec,
            translations=self.translations, min_rows=int(min_rows),
        )
        out["feature_names"] = list(X_df.columns)
        with self.cache_lock:
            self.rules_cache[key] = out
            while len(self.rules_cache) > _RULES_CACHE_MAX_ENTRIES:
                self.rules_cache.popitem(last=False)
        return out

    def get_tau_curves(
        self, params: Dict[str, Any], center_spec: CenterSpec, target: _Target,
        exclude: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        The tau-curves (`compute_tau_curves`) for an analyze parameter set,
        computed once per parameters-without-tau: `center_spec.tau` does not
        enter the key, the curve is the dependence on it. `exclude` (schemas
        left out of the family, Rules view) is part of the key.
        """
        exclude = sorted({tuple(sorted(int(j) for j in sc)) for sc in (exclude or [])})
        key = _analyze_key({k: v for k, v in params.items() if k != "tau"})  # `_target` is dropped by `_analyze_key`
        if exclude:
            key = key + (("exclude", repr(exclude)),)
        with self.cache_lock:
            hit = self.curves_cache.get(key)
            if hit is not None:
                self.curves_cache.move_to_end(key)
                return hit
        X_df, Z, _ = _target_arrays(self.df, target, params.get("drop", ()))
        curves = compute_tau_curves(
            X_df.values, Z, feature_names=list(X_df.columns), max_d=_MAX_D,
            positive_class=(1 if target.criterion is not None else None),
            center_spec=center_spec, direction=params["direction"],
            step_pct=_CURVES_STEP_PCT,
            prune_dependent=bool(params.get("prune", False)),
            exclude=[list(sc) for sc in exclude] or None,
        )
        curves["exclude"] = [list(sc) for sc in exclude]
        with self.cache_lock:
            self.curves_cache[key] = curves
            while len(self.curves_cache) > _CURVES_CACHE_MAX_ENTRIES:
                self.curves_cache.popitem(last=False)
        return curves

    # -- background prefetch ------------------------------------------------
    def start_prefetch(
        self, target_col: str, criterion: str, center_spec: CenterSpec,
        direction: str = "presence", drop: Sequence[str] = (), prune: bool = False,
    ) -> None:
        """
        After a criterion-mode analysis of (`target_col`, `criterion`),
        compute and cache the responses for the column's other values in a
        background thread, so the user's next clicks in the same column are
        served from the cache. Superseded (cancelled between values) by the
        next analysis of a different column; skipped entirely while a
        Global Pattern Scan is running, which would otherwise share the CPU
        with it.
        """
        if not self.prefetch_enabled:
            return
        with self.scan_lock:
            scanning = self.scan_job is not None and self.scan_job["status"] == "running"
        if scanning:
            return
        with self.prefetch_lock:
            if self.prefetch_cancel is not None:
                self.prefetch_cancel.set()
            cancel = threading.Event()
            self.prefetch_cancel = cancel
            thread = threading.Thread(
                target=_prefetch_sibling_values,
                args=(self, target_col, criterion, center_spec, direction, cancel, tuple(drop), bool(prune)),
                daemon=True,
                name="vsf-prefetch",
            )
            self.prefetch_thread = thread
            thread.start()

    def cancel_prefetch(self) -> None:
        with self.prefetch_lock:
            if self.prefetch_cancel is not None:
                self.prefetch_cancel.set()


class VSFRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Request handler for `vsf.serve()`. Serves the packaged `vsf.webapp`
    assets via `_STATIC_ROUTES` / `_read_webapp_asset` — there is no
    cwd-relative directory to delegate to once `vsf` is installed as a real
    package.
    """

    server: _VSFServer  # narrows the inherited `self.server` type for readers

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # Quiet by default (matches a typical Streamlit/Gradio-style launch);
        # errors still surface via `_send_json_response`'s 500 payloads and
        # via `BaseHTTPRequestHandler.log_error` on genuine protocol errors.
        pass

    # -- dispatch -----------------------------------------------------

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/columns":
            self._handle_columns_api()
        elif path == "/api/scan/status":
            self._handle_scan_status_api()
        elif path in _STATIC_ROUTES:
            self._serve_static(path)
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/analyze":
            self._handle_analyze_api()
        elif path == "/api/scan/start":
            self._handle_scan_start_api()
        elif path == "/api/scan/cancel":
            self._handle_scan_cancel_api()
        elif path == "/api/landscape":
            self._handle_landscape_api(cell=False)
        elif path == "/api/landscape/cell":
            self._handle_landscape_api(cell=True)
        elif path == "/api/landscape/frontier":
            self._handle_frontier_api()
        elif path == "/api/landscape/curves":
            self._handle_curves_api(at=False)
        elif path == "/api/landscape/at":
            self._handle_curves_api(at=True)
        elif path == "/api/target":
            self._handle_target_api()
        elif path == "/api/rules":
            self._handle_rules_api()
        elif path == "/api/rules/union":
            self._handle_rules_union_api()
        elif path == "/api/centers/groups":
            self._handle_centers_api(detail=False)
        elif path == "/api/centers/group":
            self._handle_centers_api(detail=True)
        elif path == "/api/centers/branch":
            self._handle_centers_api(detail=False, branch=True)
        elif path == "/api/screen":
            self._handle_screen_api()
        elif path == "/api/validate":
            self._handle_validate_api()
        else:
            self._read_json_body()
            self.send_error(404, "Endpoint not found")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    # -- response helpers -----------------------------------------------

    def _send_json_response(self, status_code: int, payload: Dict[str, Any]) -> None:
        """
        No `Access-Control-Allow-Origin` header — same-origin only. A
        wildcard CORS header would let ANY page open in the user's browser
        read this dataset and every analysis result over
        `http://127.0.0.1:<port>`, not just the served frontend.
        """
        self._send_json_bytes(status_code, json.dumps(payload).encode("utf-8"))

    def _send_json_bytes(self, status_code: int, body: bytes) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path: str) -> None:
        relative_path, content_type = _STATIC_ROUTES[path]
        try:
            body = _read_webapp_asset(relative_path)
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            self.send_error(404, f"Asset not found: {exc}")
            return
        if relative_path == "index.html":
            # Lazy import: vsf/__init__.py sets __version__ *after* importing
            # .server, so a module-level `from . import __version__` here
            # would raise ImportError on a partially-initialized package.
            from . import __version__ as _vsf_version
            body = body.replace("{{VSF_VERSION}}", _vsf_version)
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        return json.loads(post_data.decode("utf-8")) if post_data else {}

    # -- API endpoints ----------------------------------------------------

    def _handle_columns_api(self) -> None:
        """
        `default_target` is the SERVER INSTANCE's actual resolved default
        (`self.server.default_target`, computed once in `_VSFServer.
        __init__` from the caller's `df` — see `_default_target`), never a
        hardcoded "class".
        """
        try:
            df = self.server.df
            translations = self.server.translations
            columns = catalog_from_dataframe(df, translations)
            self._send_json_response(
                200,
                {
                    "columns": columns,
                    "default_target": self.server.default_target,
                    "total_rows": len(df),
                },
            )
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_analyze_api(self) -> None:
        """
        Runs Independent Branch Discovery (`vsf.avr.discover_branches`) for
        the requested target/criterion and returns UP TO `MAX_BRANCH_D`
        independently-found branches in one response — one visualization
        payload per dimensionality, keyed by dimensionality as a string
        (matching `vsf.dashboard.export_full_dashboard`'s `BRANCHES_DATA`
        shape) — plus `branch_dims`/`default_branch` so the frontend's
        branch selector can render and
        preselect without guessing. Switching branches afterward is a pure
        client-side re-render against this same response — no further
        request.

        `target_col = req.get("target", self.server.default_target)` falls
        back to THIS dataset's actual default target, never the literal
        string "class". An invalid/missing `target_col` (not a column of
        `df`) raises a 400 rather than silently substituting a fallback
        that would analyze the wrong column.
        """
        try:
            req = self._read_json_body()
            df = self.server.df
            translations = self.server.translations

            target, err = _resolve_target(df, req, self.server.default_target)
            if target is None:
                self._send_json_response(400, {"error": err})
                return
            target_col, criterion = target.target_col, target.criterion

            # Features are `df` minus EVERY column of the target (a
            # composite target's own columns would otherwise be found as
            # its schema, tautologically).
            drop, prune, drop_err = _parse_screen_options(df, req, target)
            if drop is None:
                self._send_json_response(400, {"error": drop_err})
                return
            X_df, Z, sort_Z = _target_arrays(df, target, drop)
            feature_names = list(X_df.columns)
            X = X_df.values

            # Certificate parameters are part of the cache key: changing tau
            # or alpha changes every centre, every colour and every reported
            # number, so a cached payload computed at a different tau must not
            # be served.
            center_spec, cert_err = _parse_certificate(req)
            if center_spec is None:
                self._send_json_response(400, {"error": cert_err})
                return
            direction = req.get("direction", "presence")
            if direction not in ("presence", "absence"):
                self._send_json_response(
                    400,
                    {"error": f"direction must be 'presence' or 'absence', got {direction!r}"},
                )
                return
            if direction == "absence" and criterion is None:
                self._send_json_response(
                    400,
                    {"error": "an absence search needs an explicit target value (criterion) whose absence to certify"},
                )
                return
            # Explicit schema (`features`: indices into the feature columns,
            # i.e. `df` minus the target, in order) - opened from the
            # landscape rather than found by the search.
            features = req.get("features", None)
            if features is not None:
                try:
                    features = [int(j) for j in features]
                except (TypeError, ValueError):
                    self._send_json_response(400, {"error": "features must be a list of column indices"})
                    return
                if (not features or len(set(features)) != len(features)
                        or len(features) > _MAX_D
                        or any(j < 0 or j >= len(feature_names) for j in features)):
                    self._send_json_response(400, {
                        "error": f"features must be 1 to {_MAX_D} distinct indices in [0, {len(feature_names)})",
                    })
                    return

            cache_key = dict(target.params())
            cache_key.update(_certificate_params(center_spec))
            cache_key.update({
                "direction": direction,
                "drop": drop,
                "prune": prune,
            })
            if features is not None:
                cache_key["features"] = tuple(features)
            key = _analyze_key(cache_key)

            # A request for a different column supersedes any prefetch of
            # the previous column's values (checked between values there).
            if criterion is None or self.server.last_params is None or (
                self.server.last_params.get("target_col") != target_col
            ):
                self.server.cancel_prefetch()

            body, event, owner = self.server.claim(key)
            if body is None and event is not None and not owner:
                # Someone else (another request, or the prefetch worker) is
                # computing exactly this response: wait for it rather than
                # computing it twice.
                event.wait()
                body = self.server.cache_get(key)
            if body is None:
                if not owner:
                    # The other computation failed before caching; own it.
                    body, event, owner = self.server.claim(key)
                    if body is None and event is not None and not owner:
                        event.wait()
                        body = self.server.cache_get(key)
            if body is None:
                assert event is not None
                try:
                    # v2.3: `discover_branches` always ranks by coverage and
                    # always requires a resolvable positive class (see
                    # `vsf.avr`'s module docstring) -- `positive_class=1` in
                    # criterion mode, where Z is literally 0/1; without a
                    # criterion, only a naturally two-valued target resolves
                    # one automatically, and a K>2-valued target raises
                    # ValueError, caught below and reported as 400 rather
                    # than crashing the request.
                    positive_class = 1 if criterion is not None else None
                    if features is not None:
                        # No uncorrected permutation p-value for a schema
                        # picked from the landscape (see `report_schema`).
                        branches = report_schema(
                            X,
                            Z,
                            features,
                            feature_names=feature_names,
                            random_state=_SCAN_RANDOM_STATE,
                            positive_class=positive_class,
                            center_spec=center_spec,
                            n_permutations_centers=0,
                            direction=direction,
                            family_max_d=_MAX_D,
                            prune_dependent=prune,
                        )
                    else:
                        branches = discover_branches(
                            X,
                            Z,
                            feature_names=feature_names,
                            max_d=_MAX_D,
                            random_state=_SCAN_RANDOM_STATE,
                            positive_class=positive_class,
                            center_spec=center_spec,
                            n_permutations_centers=DEFAULT_N_PERMUTATIONS,
                            direction=direction,
                            prune_dependent=prune,
                        )
                    response_payload = _build_analyze_response(
                        self.server, target, center_spec, branches,
                        direction=direction, features=features,
                        drop=drop, prune=prune,
                    )
                    body = json.dumps(response_payload).encode("utf-8")
                    self.server.cache_put(key, cache_key, response_payload, body)
                finally:
                    self.server.release(key, event)
                # Sibling prefetch is per column value; a composite target has
                # no siblings in that sense.
                if criterion is not None and features is None and not target.also:
                    self.server.start_prefetch(
                        target_col, str(criterion), center_spec, direction, drop, prune
                    )

            self._send_json_bytes(200, body)
        except ValueError as exc:
            # A resolvable-positive-class failure (see the discover_branches
            # call above) or a bad certificate value -- both are client
            # input problems, not server faults.
            self._send_json_response(400, {"error": str(exc)})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _parse_landscape_request(
        self, req: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], CenterSpec, Optional[int]]]:
        """
        The (params, center_spec, d) of a landscape-family request, or None
        after a 400 has been sent. `params` is the analyze key's material:
        target, criterion, also (composite targets), tau, alpha, rule,
        min_samples, direction - plus the resolved `_Target` under
        `"_target"`, which `_analyze_key` ignores.
        """
        df = self.server.df
        target, err = _resolve_target(df, req, self.server.default_target)
        if target is None:
            self._send_json_response(400, {"error": err})
            return None
        target_col, criterion = target.target_col, target.criterion
        center_spec, cert_err = _parse_certificate(req)
        if center_spec is None:
            self._send_json_response(400, {"error": cert_err})
            return None
        direction = req.get("direction", "presence")
        if direction not in ("presence", "absence"):
            self._send_json_response(400, {"error": "invalid rule or direction"})
            return None
        if direction == "absence" and criterion is None:
            self._send_json_response(400, {"error": "an absence landscape needs an explicit criterion"})
            return None
        d = req.get("d", None)
        if d is not None:
            try:
                d = int(d)
            except (TypeError, ValueError):
                self._send_json_response(400, {"error": "d must be an integer or null"})
                return None
            if d < 1 or d > _MAX_D:
                self._send_json_response(400, {"error": f"d must be in [1, {_MAX_D}]"})
                return None
        drop, prune, err = _parse_screen_options(df, req, target)
        if drop is None:
            self._send_json_response(400, {"error": err})
            return None
        params = dict(target.params())
        params.update(_certificate_params(center_spec))
        params.update({"direction": direction, "drop": drop, "prune": prune})
        params["_target"] = target  # not part of the key (see `_analyze_key`)
        return params, center_spec, d

    def _handle_landscape_api(self, cell: bool) -> None:
        """
        `/api/landscape`: the 10 x 10 count lattice of every candidate the
        search scored for the given target/criterion/certificate/direction
        (`vsf.avr.Landscape.bins`), for one dimensionality (`d`) or the whole
        family (`d` absent/null). `/api/landscape/cell`: the schemas of one
        lattice cell (`ix`, `iy`, `limit`, `offset`; `Landscape.cell`).
        Both compute the landscape on first use and cache it.
        """
        try:
            req = self._read_json_body()
            parsed = self._parse_landscape_request(req)
            if parsed is None:
                return
            params, center_spec, d = parsed
            target_col, criterion, direction = params["target_col"], params["criterion"], params["direction"]
            landscape = self.server.get_landscape(params, center_spec, params["_target"])
            if not cell:
                out = landscape.bins(d)
                out.update({
                    "target": target_col, "criterion": criterion, "direction": direction,
                    "also": [list(p) for p in params["_target"].also],
                    "n_candidates": len(landscape),
                    "feature_names": landscape.feature_names,
                })
                self._send_json_response(200, out)
                return
            try:
                ix, iy = int(req["ix"]), int(req["iy"])
                limit = int(req.get("limit", 100))
                offset = int(req.get("offset", 0))
            except (KeyError, TypeError, ValueError):
                self._send_json_response(400, {"error": "ix and iy are required integers; limit/offset optional integers"})
                return
            n = landscape.N_BINS
            if not (0 <= ix < n and 0 <= iy < n) or limit < 1 or offset < 0:
                self._send_json_response(400, {"error": f"ix, iy must be in [0, {n}); limit >= 1; offset >= 0"})
                return
            out = landscape.cell(d, ix, iy, limit=min(limit, 1000), offset=offset)
            out.update({"d": d, "ix": ix, "iy": iy, "target": target_col, "criterion": criterion, "direction": direction})
            self._send_json_response(200, out)
        except ValueError as exc:
            self._send_json_response(400, {"error": str(exc)})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_frontier_api(self) -> None:
        """
        `/api/landscape/frontier`: the cost-coverage Pareto staircase of
        every dimensionality at the request's certificate
        (`vsf.avr.Landscape.frontier`). `cost` is "centers" (default) or
        "conditions" (`d` x centres - what the description actually spells
        out). Reads the cached landscape, so it costs a sort, not a search.
        """
        try:
            req = self._read_json_body()
            parsed = self._parse_landscape_request(req)
            if parsed is None:
                return
            params, center_spec, _ = parsed
            cost = req.get("cost", "centers")
            if cost not in ("centers", "conditions"):
                self._send_json_response(400, {"error": "cost must be 'centers' or 'conditions'"})
                return
            landscape = self.server.get_landscape(params, center_spec, params["_target"])
            out = landscape.frontier(cost)
            out.update({
                "target": params["target_col"], "criterion": params["criterion"],
                "direction": params["direction"], "tau": params["tau"], "rule": params["rule"],
                "also": [list(p) for p in params["_target"].also],
                "n_candidates": len(landscape),
                "feature_names": landscape.feature_names,
            })
            self._send_json_response(200, out)
        except ValueError as exc:
            self._send_json_response(400, {"error": str(exc)})
        except Exception as e:  # pragma: no cover - defensive
            self._send_json_response(500, {"error": str(e)})

    def _handle_screen_api(self) -> None:
        """
        `/api/screen`: the dataset screen of Section 4.12 - one profile per
        column (categories, largest category, singleton categories, missing
        rows, flags) and every pair of columns one of which (nearly)
        determines the other, with the exact number of exception rows.
        Computed on the FEATURE columns alone and cached per server, so it
        can be read before any target is chosen; `min_strength` moves the
        reporting floor of the inexact pairs.

        With a resolvable target in the request the response also carries
        `target_report` - how well each remaining column determines the
        target indicator - which is the leakage check and is the only part
        that depends on the target. `drop` is echoed back and removes the
        named columns from BOTH parts, so the reader sees the screen of the
        feature space the search will actually run on.
        """
        try:
            req = self._read_json_body()
            df = self.server.df
            try:
                min_strength = float(req.get("min_strength", DEFAULT_MIN_STRENGTH))
            except (TypeError, ValueError):
                self._send_json_response(400, {"error": "min_strength must be a number"})
                return
            if not (0.0 <= min_strength <= 1.0):
                self._send_json_response(400, {"error": "min_strength must be in [0, 1]"})
                return
            target: Optional[_Target] = None
            if req.get("target") is not None:
                target, err = _resolve_target(df, req, self.server.default_target)
                if target is None:
                    self._send_json_response(400, {"error": err})
                    return
            drop, prune, err = _parse_screen_options(df, req, target)
            if drop is None:
                self._send_json_response(400, {"error": err})
                return
            target_columns = list(target.columns) if target is not None else []
            keep = [c for c in df.columns if c not in set(target_columns) and c not in set(drop)]
            screen = self.server.get_screen(tuple(keep), min_strength)
            out = screen.to_dict()
            out.update({
                "columns_screened": keep,
                "dropped_columns": list(drop),
                "target_columns": target_columns,
                "target": None if target is None else target.target_col,
                "criterion": None if target is None else target.criterion,
                "prune": prune,
                "target_report": None,
            })
            if target is not None and target.criterion is not None and keep:
                X_df, Z, _ = _target_arrays(df, target, drop)
                out["target_report"] = target_report(X_df.values, Z, list(X_df.columns))
            self._send_json_response(200, out)
        except ValueError as exc:
            self._send_json_response(400, {"error": str(exc)})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_rules_api(self) -> None:
        """
        `/api/rules`: every occupied cell of the given `schemas` (feature-index
        lists - the schemas on the tau-curves' envelope, the winners among them)
        as a conjunctive rule with its purity, rows, certificate status and
        generalisations (`vsf.rules.enumerate_rules`, Section 4.15). The
        certificate parameters and target are those of an analyze request;
        `min_rows` (default 1) drops smaller cells. Cached.
        """
        try:
            req = self._read_json_body()
            parsed = self._parse_landscape_request(req)
            if parsed is None:
                return
            params, center_spec, _ = parsed
            schemas = req.get("schemas")
            if not isinstance(schemas, list) or not schemas or len(schemas) > _RULES_MAX_SCHEMAS:
                self._send_json_response(400, {"error": f"schemas must be a non-empty list of at most {_RULES_MAX_SCHEMAS} feature-index lists"})
                return
            try:
                schemas_i = [[int(j) for j in sc] for sc in schemas]
                min_rows = int(req.get("min_rows", 1))
            except (TypeError, ValueError):
                self._send_json_response(400, {"error": "schemas must hold integer indices; min_rows an integer"})
                return
            if min_rows < 1 or any(not sc or len(sc) > _MAX_D or len(set(sc)) != len(sc) for sc in schemas_i):
                self._send_json_response(400, {"error": f"each schema must be 1 to {_MAX_D} distinct indices; min_rows >= 1"})
                return
            n_features = int(_target_arrays(self.server.df, params["_target"], params.get("drop", ()))[0].shape[1])
            if any(j < 0 or j >= n_features for sc in schemas_i for j in sc):
                self._send_json_response(400, {"error": f"schema indices must lie in [0, {n_features})"})
                return
            out = dict(self.server.get_rules(params, center_spec, params["_target"], schemas_i, min_rows))
            out.update({
                "target": params["target_col"], "criterion": params["criterion"],
                "direction": params["direction"], "tau": params["tau"], "rule": params["rule"],
                "also": [list(p) for p in params["_target"].also],
            })
            self._send_json_response(200, out)
        except ValueError as exc:
            self._send_json_response(400, {"error": str(exc)})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_rules_union_api(self) -> None:
        """
        `/api/rules/union`: what a set of rules covers together
        (`vsf.rules.union_coverage`): the union of their rows, the value
        rows among them (coverage = recall of "the value is where a rule
        fires"), and the precision of that prediction. `rules` is a list of
        {features, values}; `certified` (optional, a list of booleans of
        the same length) asks for the same figures for the certified subset
        as well. Target, direction and exclusions as for `/api/rules`.
        """
        try:
            req = self._read_json_body()
            parsed = self._parse_landscape_request(req)
            if parsed is None:
                return
            params, _, _ = parsed
            rules = req.get("rules")
            if not isinstance(rules, list) or len(rules) > _RULES_UNION_MAX:
                self._send_json_response(400, {"error": f"rules must be a list of at most {_RULES_UNION_MAX} {{features, values}} objects"})
                return
            cert = req.get("certified", None)
            if cert is not None and (not isinstance(cert, list) or len(cert) != len(rules)):
                self._send_json_response(400, {"error": "certified must be a list of booleans, one per rule"})
                return
            target = params["_target"]
            if target.criterion is None:
                self._send_json_response(400, {"error": "the rules view needs an explicit target value (criterion)"})
                return
            X_df, Z, _ = _target_arrays(self.server.df, target, params.get("drop", ()))
            Z = _np.asarray(Z).astype(int)
            if params["direction"] == "absence":
                Z = 1 - Z
            out: Dict[str, Any] = {"all": union_coverage(X_df.values, Z, rules)}
            if cert is not None:
                out["certified"] = union_coverage(X_df.values, Z, [r for r, c in zip(rules, cert) if c])
            out.update({"target": target.target_col, "criterion": target.criterion, "direction": params["direction"], "tau": params["tau"]})
            self._send_json_response(200, out)
        except (ValueError, KeyError, TypeError) as exc:
            self._send_json_response(400, {"error": str(exc)})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_centers_api(self, detail: bool, branch: bool = False) -> None:
        """
        `/api/centers/groups`: the centres of every schema the search scored
        at the request's certificate, grouped by mutual containment at
        `threshold` (default 0.8, range [0.5, 1]); `min_rows` (default 1)
        drops centres with fewer rows before grouping; `sort` is
        "coverage" (default) or "members"; `filter_d` keeps groups with a
        member of that dimensionality; `cell` = {d, ix, iy} keeps groups
        with a member in that landscape lattice cell (d null: the
        all-d lattice); `limit`/`offset` page the representatives.
        `/api/centers/group`: the members of the group whose representative
        is `group` (a distinct-set id from the list), paged.
        `/api/centers/branch`: the centres of ONE schema (`features`, the
        selected branch or an opened schema) and, per centre, every centre
        of another schema with mutual containment >= `threshold` to it;
        `anchor` (a cell code) pages one centre's list with
        `limit`/`offset` (`CenterCatalog.branch_view`).
        `kinship` ("all", "related", "unrelated"; both detail endpoints)
        keeps only the alternatives whose schema is nested with the
        reference's column set, or only those whose schema is not; the
        unfiltered split comes back as `kinship_counts` either way.
        """
        try:
            req = self._read_json_body()
            parsed = self._parse_landscape_request(req)
            if parsed is None:
                return
            params, center_spec, _ = parsed
            try:
                threshold = float(req.get("threshold", DEFAULT_GROUP_THRESHOLD))
                min_rows = int(req.get("min_rows", 1))
                limit = int(req.get("limit", 20 if not detail else 50))
                offset = int(req.get("offset", 0))
            except (TypeError, ValueError):
                self._send_json_response(400, {"error": "threshold must be a number; min_rows, limit, offset integers"})
                return
            if min_rows < 1 or limit < 1 or offset < 0:
                self._send_json_response(400, {"error": "min_rows >= 1, limit >= 1, offset >= 0 are required"})
                return
            limit = min(limit, _CENTERS_MAX_LIMIT)
            kinship = req.get("kinship", "all")
            if kinship not in ("all", "related", "unrelated"):
                self._send_json_response(400, {"error": "kinship must be 'all', 'related' or 'unrelated'"})
                return
            catalog = self.server.get_center_catalog(params, center_spec, params["_target"], min_rows)
            echo = {
                "target": params["target_col"], "criterion": params["criterion"],
                "direction": params["direction"], "tau": params["tau"], "rule": params["rule"],
                "alpha": params["alpha"], "min_samples": params["min_samples"],
                "also": [list(p) for p in params["_target"].also],
            }
            if branch:
                feats = req.get("features")
                if not isinstance(feats, list) or not feats:
                    self._send_json_response(400, {"error": "features must be a non-empty list of column indices"})
                    return
                try:
                    feats_i = [int(j) for j in feats]
                    anchor = req.get("anchor", None)
                    anchor = None if anchor is None else int(anchor)
                except (TypeError, ValueError):
                    self._send_json_response(400, {"error": "features and anchor must be integers"})
                    return
                out = catalog.branch_view(feats_i, threshold, limit=limit, offset=offset, anchor=anchor, kinship=kinship)
            elif detail:
                try:
                    group = int(req["group"])
                except (KeyError, TypeError, ValueError):
                    self._send_json_response(400, {"error": "group is a required integer"})
                    return
                out = catalog.group_detail(threshold, group, limit=limit, offset=offset, kinship=kinship)
            else:
                sort = req.get("sort", "coverage")
                filter_d = req.get("filter_d", None)
                if filter_d is not None:
                    try:
                        filter_d = int(filter_d)
                    except (TypeError, ValueError):
                        filter_d = 0
                    if not (1 <= filter_d <= _MAX_D):
                        self._send_json_response(400, {"error": f"filter_d must be an integer in [1, {_MAX_D}]"})
                        return
                cell_req = req.get("cell", None)
                cell = None
                if cell_req is not None:
                    n_bins = catalog.landscape.N_BINS
                    try:
                        cd = cell_req.get("d", None)
                        cell = (None if cd is None else int(cd), int(cell_req["ix"]), int(cell_req["iy"]))
                    except (AttributeError, KeyError, TypeError, ValueError):
                        cell = None
                    if cell is None or not (0 <= cell[1] < n_bins and 0 <= cell[2] < n_bins) or (
                        cell[0] is not None and not (1 <= cell[0] <= _MAX_D)
                    ):
                        self._send_json_response(400, {"error": f"cell must be {{d: 1..{_MAX_D} or null, ix: 0..{n_bins - 1}, iy: 0..{n_bins - 1}}}"})
                        return
                out = catalog.groups_page(
                    threshold, sort=sort, d=filter_d, cell=cell, limit=limit, offset=offset,
                    kinship=kinship,
                )
                out["filter_d"] = filter_d
                out["cell"] = None if cell is None else {"d": cell[0], "ix": cell[1], "iy": cell[2]}
            out.update(echo)
            self._send_json_response(200, out)
        except ValueError as exc:
            self._send_json_response(400, {"error": str(exc)})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_target_api(self) -> None:
        """
        `/api/target`: what a (possibly composite) target IS before any
        search runs - its display name, the number and share of rows it
        selects (the base rate the certificate scale is anchored to), the
        columns it removes from the feature space and how many features
        remain. Cheap (one boolean mask), so the frontend can call it on
        every change of the conjunction.
        """
        try:
            req = self._read_json_body()
            df = self.server.df
            target, err = _resolve_target(df, req, self.server.default_target)
            if target is None:
                self._send_json_response(400, {"error": err})
                return
            drop, _prune, drop_err = _parse_screen_options(df, req, target)
            if drop is None:
                self._send_json_response(400, {"error": drop_err})
                return
            X_df, Z, _ = _target_arrays(df, target, drop)
            n = int(len(df))
            n_pos = int((Z == 1).sum()) if target.criterion is not None else None
            self._send_json_response(200, {
                "target": target.target_col, "criterion": target.criterion,
                "also": [list(p) for p in target.also],
                "target_display": _target_display(target, self.server.translations),
                "target_columns": target.columns,
                "n_samples": n,
                "n_positive": n_pos,
                "share": (n_pos / n) if (n_pos is not None and n) else None,
                "n_features": int(X_df.shape[1]),
                "feature_names": list(X_df.columns),
                "dropped_columns": list(drop),
            })
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_curves_api(self, at: bool) -> None:
        """
        `/api/landscape/curves`: the per-d envelope of the landscape over
        the purity floor (`vsf.avr.compute_tau_curves`) - coverage (mass
        under absence) of the best d-subset at every whole-percent tau from
        the base rate to 100 %, with that subset and its centre count. The
        request's `tau` is ignored (the curve is the dependence on tau);
        rule/alpha/min_samples/direction are honoured. Cached per those.

        `/api/landscape/at`: the schemas of dimensionality `d` whose
        x-fraction at the purity floor `tau` falls in the same ten-percent
        category (`ix`, 0..9, left-open right-closed) - the click-through
        of a curve point. Computes (and caches) the landscape at that tau.
        """
        try:
            req = self._read_json_body()
            parsed = self._parse_landscape_request(req)
            if parsed is None:
                return
            params, center_spec, d = parsed
            target_col, criterion, direction = params["target_col"], params["criterion"], params["direction"]
            if not at:
                exclude = req.get("exclude", None) or []
                if not isinstance(exclude, list) or len(exclude) > _CURVES_MAX_EXCLUDE:
                    self._send_json_response(400, {"error": f"exclude must be a list of at most {_CURVES_MAX_EXCLUDE} feature-index lists"})
                    return
                try:
                    exclude_i = [[int(j) for j in sc] for sc in exclude]
                except (TypeError, ValueError):
                    self._send_json_response(400, {"error": "exclude must hold integer feature indices"})
                    return
                if any(not sc or len(sc) > _MAX_D or len(set(sc)) != len(sc) for sc in exclude_i):
                    self._send_json_response(400, {"error": f"each excluded schema must be 1 to {_MAX_D} distinct indices"})
                    return
                curves = self.server.get_tau_curves(params, center_spec, params["_target"], exclude_i)
                out = dict(curves)
                out.update({"target": target_col, "criterion": criterion, "direction": direction,
                            "also": [list(p) for p in params["_target"].also], "rule": params["rule"]})
                self._send_json_response(200, out)
                return
            if d is None:
                self._send_json_response(400, {"error": "d is required for /api/landscape/at"})
                return
            try:
                ix = int(req["ix"])
                limit = int(req.get("limit", 100))
                offset = int(req.get("offset", 0))
            except (KeyError, TypeError, ValueError):
                self._send_json_response(400, {"error": "ix is a required integer; limit/offset optional integers"})
                return
            landscape = self.server.get_landscape(params, center_spec, params["_target"])
            n = landscape.N_BINS
            if not (0 <= ix < n) or limit < 1 or offset < 0:
                self._send_json_response(400, {"error": f"ix must be in [0, {n}); limit >= 1; offset >= 0"})
                return
            out = landscape.by_x_category(d, ix, limit=min(limit, 1000), offset=offset)
            out.update({"d": d, "ix": ix, "tau": params["tau"], "x": ("mass" if direction == "absence" else "coverage"),
                        "target": target_col, "criterion": criterion, "direction": direction})
            self._send_json_response(200, out)
        except ValueError as exc:
            self._send_json_response(400, {"error": str(exc)})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_scan_start_api(self) -> None:
        """
        Starts a Global Pattern Scan (see module docstring) in a background
        thread and returns immediately — the caller polls
        `/api/scan/status` for progress/results. Rejects with 409 if a scan
        is already running on this server instance (only one at a time;
        cancel it first via `/api/scan/cancel` to start a different one).
        """
        try:
            req = self._read_json_body()
            if "nmi_threshold" in req:
                self._send_json_response(400, {
                    "error": (
                        "nmi_threshold no longer exists: this scan filters on "
                        "certified-centre coverage, not on an association "
                        "statistic. Use coverage_threshold (share of the target "
                        "value captured by certified centres, percent)."
                    )
                })
                return
            if "u_adj_threshold" in req and "coverage_threshold" not in req:
                self._send_json_response(400, {
                    "error": (
                        "u_adj_threshold no longer exists: it measured whether an "
                        "association EXISTS, while this scan looks for (column, "
                        "value) pairs that produce certified discrete centres - the "
                        "two select different pairs. Pass coverage_threshold "
                        "(percent), and optionally tau and alpha."
                    )
                })
                return
            try:
                threshold_pct = float(req.get("coverage_threshold", 20))
                scan_tau = float(req.get("tau", 0.90))
                scan_alpha = float(req.get("alpha", 0.05))
                scan_min_samples = int(req.get("min_samples", 1))
            except (TypeError, ValueError):
                self._send_json_response(
                    400, {
                        "error": (
                            "coverage_threshold, tau, alpha and min_samples "
                            "must be numbers"
                        )
                    },
                )
                return
            if not (0.0 <= threshold_pct <= 100.0):
                self._send_json_response(
                    400, {"error": "coverage_threshold must be in [0, 100]"}
                )
                return
            # `rule` (Strict mode) is shared with the Centres & Colour panel --
            # the scan is certified under the same certificate the viewport
            # would show. `min_samples` is NOT shared: it has its own
            # "Min. objects" field in the Scan panel (scanMinSamples in
            # app.js), independent of Centres & Colour's, because a sensible
            # per-cell floor for mining across hundreds of pairs is not
            # necessarily the one you'd set while looking at a single
            # branch. (Both used to silently fall back to CenterSpec's
            # defaults -- rule="purity", min_samples=1 -- regardless of what
            # either UI value was; that bug is what introduced this parsing.)
            # Validated the same way /api/analyze does above, so both
            # endpoints agree on what a bad `rule` looks like.
            scan_rule = req.get("rule", "purity")
            if scan_rule not in ("purity", "certified"):
                self._send_json_response(
                    400,
                    {"error": f"rule must be 'purity' or 'certified', got {scan_rule!r}"},
                )
                return
            scan_multiplicity = req.get("multiplicity", None) or _default_multiplicity(scan_rule)
            if scan_multiplicity not in ("family", "bonferroni") or (
                scan_multiplicity == "family" and scan_rule != "certified"
            ):
                self._send_json_response(
                    400,
                    {"error": "multiplicity must be 'family' (with rule='certified') or 'bonferroni'"},
                )
                return
            scan_direction = req.get("direction", "presence")
            if scan_direction not in ("presence", "absence"):
                self._send_json_response(
                    400,
                    {"error": f"direction must be 'presence' or 'absence', got {scan_direction!r}"},
                )
                return
            try:
                scan_spec = CenterSpec(
                    tau=scan_tau, alpha=scan_alpha, rule=scan_rule,
                    min_samples=scan_min_samples, multiplicity=scan_multiplicity,
                )
            except ValueError as exc:
                self._send_json_response(400, {"error": str(exc)})
                return
            try:
                fdr_q = float(req.get("fdr_q", 0.05))
            except (TypeError, ValueError):
                self._send_json_response(400, {"error": "fdr_q must be a number"})
                return
            if not (0.0 < fdr_q <= 1.0):
                self._send_json_response(400, {"error": "fdr_q must be in (0, 1]"})
                return
            try:
                n_perm = int(req.get("n_permutations", _SCAN_N_PERMUTATIONS))
                n_perm_fw = int(req.get("n_permutations_familywise", 0))
            except (TypeError, ValueError):
                self._send_json_response(400, {"error": "permutation counts must be integers"})
                return
            if n_perm < 0 or n_perm_fw < 0:
                self._send_json_response(400, {"error": "permutation counts must be non-negative"})
                return
            if n_perm == 0 and n_perm_fw == 0:
                self._send_json_response(400, {
                    "error": (
                        "the scan requires a permutation null: with no p-values there is "
                        "nothing for Benjamini-Hochberg to control across the swept family."
                    )
                })
                return

            with self.server.scan_lock:
                if self.server.scan_job is not None and self.server.scan_job["status"] == "running":
                    self._send_json_response(409, {"error": "A scan is already in progress on this server."})
                    return
                # The scan needs the CPU; a running prefetch yields to it.
                self.server.cancel_prefetch()
                cancel_event = threading.Event()
                self.server.scan_cancel_event = cancel_event
                self.server.scan_job = {
                    "status": "running",
                    "threshold_pct": threshold_pct,
                    "tau": scan_spec.tau,
                    "alpha": scan_spec.alpha,
                    "rule": scan_spec.rule,
                    "multiplicity": scan_spec.multiplicity,
                    "min_samples": scan_spec.min_samples,
                    "direction": scan_direction,
                    "fdr_q": fdr_q,
                    "n_permutations": n_perm,
                    "n_permutations_familywise": n_perm_fw,
                    "started_at": time.time(),
                    "progress": {"current": 0, "total": 0, "label": ""},
                    "results": None,
                    "error": None,
                }

            scan_drop, scan_prune, drop_err = _parse_screen_options(self.server.df, req)
            if scan_drop is None:
                self._send_json_response(400, {"error": drop_err})
                return
            thread = threading.Thread(
                target=_run_dataset_scan,
                args=(
                    self.server, threshold_pct, fdr_q, n_perm, n_perm_fw,
                    cancel_event, scan_spec, scan_direction, scan_drop, scan_prune,
                ),
                daemon=True,
            )
            thread.start()
            self._send_json_response(200, {"status": "started"})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_validate_api(self) -> None:
        """
        `/api/validate`: start, or report, the post-selection check of one
        analysis (see the module docstring). The body is an analysis body
        (target, criterion, also, tau, alpha, rule, min_samples, direction,
        drop, prune) plus `method` ("split" default, or
        "family_bonferroni") and `repeats` (1..10, default 5). A failed job
        is returned as such until a request carries `retry: true`.
        """
        try:
            req = self._read_json_body()
            parsed = self._parse_landscape_request(req)
            if parsed is None:
                return
            params, center_spec, _ = parsed
            target = params["_target"]
            if target.criterion is None:
                self._send_json_response(400, {"error": "validation needs an explicit criterion (a named value)"})
                return
            method = req.get("method", "split")
            if method not in ("split", "family_bonferroni"):
                self._send_json_response(400, {"error": "method must be 'split' or 'family_bonferroni'"})
                return
            try:
                repeats = int(req.get("repeats", 5))
            except (TypeError, ValueError):
                self._send_json_response(400, {"error": "repeats must be an integer"})
                return
            if not (1 <= repeats <= _VALIDATE_MAX_REPEATS):
                self._send_json_response(400, {"error": f"repeats must be in [1, {_VALIDATE_MAX_REPEATS}]"})
                return
            if not (0.0 < center_spec.tau < 1.0):
                self._send_json_response(400, {"error": "a certificate needs a purity floor below 100 %"})
                return
            key = _analyze_key(params) + (("method", method), ("repeats", repeats))
            server = self.server
            with server.validate_lock:
                job = server.validate_jobs.get(key)
                if job is not None and job["status"] == "error" and bool(req.get("retry", False)):
                    # A failed job is reported to every poll; only an explicit
                    # retry (the user pressing the button again) restarts it,
                    # so a polling page cannot loop on a deterministic failure.
                    del server.validate_jobs[key]
                    job = None
                if job is None:
                    if server.validate_running is not None:
                        running = server.validate_jobs.get(server.validate_running, {})
                        self._send_json_response(409, {
                            "error": "another validation is running; wait for it to finish",
                            "running": running.get("request"),
                        })
                        return
                    total = certification_passes(method) + nested_passes(_VALIDATE_FOLDS, repeats)
                    job = {
                        "status": "running",
                        "progress": {"done": 0, "total": total},
                        "request": {
                            "target": target.target_col, "criterion": target.criterion,
                            "also": [list(pair) for pair in target.also],
                            "tau": center_spec.tau, "alpha": center_spec.alpha,
                            "rule": center_spec.rule, "multiplicity": center_spec.multiplicity,
                            "min_samples": center_spec.min_samples,
                            "direction": params["direction"], "drop": list(params["drop"]),
                            "prune": params["prune"], "method": method, "repeats": repeats,
                        },
                        "result": None,
                        "error": None,
                        "started": time.time(),
                        "elapsed": 0.0,
                    }
                    server.validate_jobs[key] = job
                    server.validate_running = key
                    thread = threading.Thread(
                        target=_run_validation,
                        args=(server, key, params, center_spec, target, method, repeats),
                        daemon=True,
                    )
                    thread.start()
                else:
                    server.validate_jobs.move_to_end(key)
                snapshot = dict(job)
                if snapshot["status"] == "running":
                    snapshot["elapsed"] = time.time() - snapshot["started"]
            self._send_json_response(200, snapshot)
        except ValueError as exc:
            self._send_json_response(400, {"error": str(exc)})
        except Exception as e:  # pragma: no cover - defensive
            self._send_json_response(500, {"error": str(e)})

    def _handle_scan_status_api(self) -> None:
        """
        Reports the current/last Global Pattern Scan's status. `{"status":
        "idle"}` before any scan has ever run on this server instance;
        otherwise the full job dict (`status` in {"running", "done",
        "cancelled", "error"}, `progress`, `results`, `error`) — see
        `_run_dataset_scan`. Safe to poll repeatedly; a page reload mid-scan
        just resumes polling the same in-progress job.
        """
        with self.server.scan_lock:
            job = dict(self.server.scan_job) if self.server.scan_job is not None else None
        if job is None:
            self._send_json_response(200, {"status": "idle"})
            return
        self._send_json_response(200, job)

    def _handle_scan_cancel_api(self) -> None:
        """
        Signals the running scan's background thread to stop after its
        CURRENT column finishes — the per-column search
        (`discover_branches_by_value`) isn't interrupted mid-search, so
        cancellation lands within one column's worth of time, not instantly. A no-op (200) if no scan is
        currently running.
        """
        with self.server.scan_lock:
            event = self.server.scan_cancel_event
            job = self.server.scan_job
            if event is None or job is None or job["status"] != "running":
                self._send_json_response(200, {"status": "not_running"})
                return
            event.set()
        self._send_json_response(200, {"status": "cancelling"})


def _analyze_key(params: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Hashable cache key of an `/api/analyze` request's parameters. Keys
    starting with an underscore are carriers (the resolved `_Target`), not
    key material.
    """
    return tuple(sorted((str(k), repr(v)) for k, v in params.items() if not str(k).startswith("_")))


def _cell_description(
    X_df: pd.DataFrame,
    feature_names: Sequence[str],
    features: Sequence[int],
    rows: "_np.ndarray",
    translations: Optional[Translations],
) -> List[Dict[str, Any]]:
    """
    The (column, values) conditions of one cell, read off its rows. A column
    the capacity rule coarsened contributes several values ("column in
    {a, b}"), which is exactly what the cell holds.
    """
    out: List[Dict[str, Any]] = []
    for j in features:
        col = feature_names[j]
        raw = pd.unique(X_df[col].values[rows])
        values = sorted((str(v) for v in raw), key=str)
        out.append({
            "column": _vis.humanize_col(col, translations),
            "values": [_vis.humanize_val(col, v, translations) for v in values],
        })
    return out


def _cv_dict(cv: Any) -> Dict[str, Any]:
    return {
        "mean": float(cv.mean), "se": float(cv.se), "purity": float(cv.purity_mean),
        "n_splits": int(cv.n_splits), "n_repeats": int(cv.n_repeats),
    }


def _run_validation(
    server: "_VSFServer",
    key: Tuple[Any, ...],
    params: Dict[str, Any],
    center_spec: CenterSpec,
    target: _Target,
    method: str,
    repeats: int,
) -> None:
    """Background worker of `/api/validate`; writes into `server.validate_jobs[key]`."""
    cert_total = certification_passes(method)  # type: ignore[arg-type]
    total = cert_total + nested_passes(_VALIDATE_FOLDS, repeats)

    def report(done: int) -> None:
        with server.validate_lock:
            job = server.validate_jobs.get(key)
            if job is not None:
                job["progress"] = {"done": int(done), "total": total}

    try:
        X_df, Z, _ = _target_arrays(server.df, target, params.get("drop", ()))
        names = list(X_df.columns)
        X = X_df.values
        direction = params["direction"]
        prune = bool(params.get("prune", False))
        cert = certify_discovery(
            X, Z, names, positive_class=1, center_spec=center_spec,
            method=method, max_d=_MAX_D,  # type: ignore[arg-type]
            random_state=_SCAN_RANDOM_STATE, direction=direction,
            prune_dependent=prune,
            progress=lambda done, _t: report(done),
        )
        nested = nested_crossvalidation(
            X, Z, names, positive_class=1, center_spec=center_spec, max_d=_MAX_D,
            n_splits=_VALIDATE_FOLDS, n_repeats=repeats,
            random_state=_SCAN_RANDOM_STATE, direction=direction,
            prune_dependent=prune,
            progress=lambda done, _t: report(cert_total + done),
        )
        prepared = _prepare_search(X, Z, names, 1, center_spec, direction, prune)
        assert prepared is not None
        factory = prepared[0]
        translations = server.translations

        cert_branches: Dict[str, Any] = {}
        for d, br in cert.branches.items():
            codes, _ = factory.codes(br.features)
            certified = br.certified_cells
            cells = []
            for c in certified[:_VALIDATE_MAX_CELLS]:
                cells.append({
                    "conditions": _cell_description(X_df, names, br.features, codes == c.cell, translations),
                    "n": c.n, "k": c.k,
                    "purity": (c.k / c.n) if c.n else 0.0,
                    "purity_lower": c.purity_lower,
                    "p_value": c.p_value,
                    "p_adjusted": c.p_adjusted,
                })
            cert_branches[str(d)] = {
                "features": [_vis.humanize_col(names[j], translations) for j in br.features],
                "n_tested": len(br.cells),
                "n_certified": len(certified),
                "cells": cells,
                "cells_truncated": len(certified) > _VALIDATE_MAX_CELLS,
                "coverage_eval": br.coverage_eval,
                "coverage_all_rows": br.coverage_all_rows,
                "mass_eval": br.mass_eval,
                "n_eval": br.n_eval,
                "n_eval_positive": br.n_eval_positive,
            }

        nested_branches: Dict[str, Any] = {}
        for d, nb in nested.branches.items():
            st = nb.stability
            nested_branches[str(d)] = {
                "winner": [_vis.humanize_col(n, translations) for n in nb.full_data_winner_names],
                "nested": _cv_dict(nb.nested),
                "fixed_schema": _cv_dict(nb.fixed_schema),
                "stability": {
                    "share_same": st.share_equal_to_reference,
                    "n_resamples": st.n_resamples,
                    "n_distinct": st.n_distinct,
                    "modal": [_vis.humanize_col(names[j], translations) for j in st.modal_schema],
                    "modal_share": st.modal_share,
                    "mean_pairwise_jaccard": st.mean_pairwise_jaccard,
                },
            }
        d_star = nested.select_dimensionality()
        result = {
            "certificate": {
                "method": cert.method,
                "guarantee": cert.guarantee,
                "n_tests": {str(d): t for d, t in cert.n_tests.items()},
                "per_cell_level": {str(d): lv for d, lv in cert.per_cell_level.items()},
                "search_rows": cert.search_rows,
                "eval_rows": cert.eval_rows,
                "branches": cert_branches,
            },
            "nested": {
                "undetermined_reason": nested.undetermined_reason,
                "n_splits": nested.n_splits,
                "n_repeats": nested.n_repeats,
                "d_star": d_star,
                "branches": nested_branches,
            },
        }
        with server.validate_lock:
            job = server.validate_jobs.get(key)
            if job is not None:
                job.update(status="done", result=result, progress={"done": total, "total": total},
                           elapsed=time.time() - job["started"])
    except Exception as exc:  # reported to the client, not raised in a daemon thread
        with server.validate_lock:
            job = server.validate_jobs.get(key)
            if job is not None:
                job.update(status="error", error=str(exc), elapsed=time.time() - job["started"])
    finally:
        with server.validate_lock:
            if server.validate_running == key:
                server.validate_running = None
            finished = [k for k, j in server.validate_jobs.items() if j["status"] != "running"]
            while len(finished) > _VALIDATE_CACHE_MAX_ENTRIES:
                server.validate_jobs.pop(finished.pop(0), None)


def _build_analyze_response(
    server: "_VSFServer",
    target: _Target,
    center_spec: CenterSpec,
    branches: Dict[int, Any],
    direction: str = "presence",
    features: Optional[List[int]] = None,
    drop: Sequence[str] = (),
    prune: bool = False,
) -> Dict[str, Any]:
    """
    The `/api/analyze` response body for discovered `branches`: one
    visualisation payload per dimensionality plus the branch selector's
    metadata. Shared by the request handler and the background prefetch so
    the two can never disagree on a field.
    """
    df = server.df
    translations = server.translations
    target_col, criterion = target.target_col, target.criterion
    X_df, Z, sort_Z = _target_arrays(df, target, drop)
    indicator_labels = None
    display_target_name = _target_display(target, translations)
    if criterion is not None and direction == "absence":
        # The renderer's "positive" indicator is the complement: every
        # cell purity, bound and certificate it computes must refer to
        # rows WITHOUT the value, exactly as the search did. The class
        # labels keep naming the value, so a point's hover still reads
        # "Column = value" / "not Column = value".
        Z = 1 - Z
        indicator_labels = (display_target_name, f"not {display_target_name}")
        display_target_name = (
            f"not ({display_target_name})" if target.also
            else f"{_vis.humanize_col(target_col, translations)} \u2260 {_vis.humanize_val(target_col, criterion, translations)}"
        )
    feature_names = list(X_df.columns)
    X = X_df.values
    n_positive = int((Z == 1).sum()) if criterion is not None else None
    # The display certifies with the spec the search resolved (for a family
    # certificate: the family size of this search); every branch carries it.
    if branches:
        center_spec = next(iter(branches.values())).centers.spec

    branches_data: Dict[str, Any] = {}
    for d, branch in branches.items():
        branches_data[str(d)] = prepare_visualization_payload(
            branch,
            X,
            Z,
            feature_names=feature_names,
            target_name=display_target_name,
            # Z is a One-vs-Rest 0/1 vector in criterion mode, so its class
            # labels must read as the criterion and its negation, not as "0"
            # and "1".
            target_is_indicator=criterion is not None,
            sort_Z=sort_Z,
            translations=translations,
            positive_value=(
                1 if criterion is not None
                else (_np.unique(Z)[-1] if len(_np.unique(Z)) else None)
            ),
            center_spec=center_spec,
            indicator_labels=indicator_labels,
        )
    selected_d = select_branch_dimensionality(branches)
    return {
        "target": target_col,
        "criterion": criterion,
        # The extra (column, value) conjuncts of a composite target (empty
        # for a plain one), the display name of the whole conjunction, the
        # columns removed from the feature space, and the size of the
        # indicator searched (after the absence inversion, if any).
        "also": [list(p) for p in target.also],
        "target_display": display_target_name,
        "target_columns": target.columns,
        # The reader's exclusions from the dataset screen (Section 4.12) and
        # whether renaming candidates were pruned: both change the candidate
        # family, so both are part of the analysis identity and are echoed
        # here for the legend and the export.
        "dropped_columns": [str(c) for c in drop],
        "prune_dependent": bool(prune),
        "n_features": len(feature_names),
        "n_positive": n_positive,
        "n_samples": int(len(df)),
        # "presence" or "absence" (see `vsf.avr.Direction`). Under
        # "absence" every coverage/purity/centre figure in `branches`
        # refers to the complement of the criterion, and the frontend
        # draws certified cells red ("certified free of the value") with
        # the mass of those cells as the headline.
        "direction": direction,
        # Set when the response is for an explicitly chosen schema (opened
        # from the landscape) rather than the search's winners: the branch
        # carries no uncorrected p-value, by design (`vsf.avr.report_schema`).
        "schema": (
            None if features is None else {
                "features": [int(j) for j in features],
                "feature_names": [feature_names[j] for j in features],
                "selected_from_landscape": True,
            }
        ),
        "branches": branches_data,
        "branch_dims": sorted(branches.keys()),
        # v2.2: the branch opened first is the SMALLEST SUFFICIENT one, not
        # the widest available. The product goal is the fewest cells that
        # capture the target value, and `max(branches)` is the opposite of
        # that: on `relationship = Husband` the 4-D branch certifies 3
        # centres for 98.6 % coverage where the 2-D branch certifies 1 for
        # 99.9 %. Falls back to the widest branch only when no
        # dimensionality certifies anything, since there is then nothing to
        # prefer.
        "default_branch": (
            str(selected_d) if selected_d in branches
            else (str(max(branches.keys())) if branches else None)
        ),
        # v2.2: the answer to "how many characteristics does it take to
        # describe this value" -- the smallest sufficient dimensionality by
        # out-of-sample certified coverage, or None when no dimensionality
        # certifies anything. The frontend must render None as "none",
        # never as 1.
        "sufficient_d": None if selected_d is None else int(selected_d),
        "certificate": _certificate_payload(center_spec),
    }


def _prefetch_sibling_values(
    server: "_VSFServer",
    target_col: str,
    criterion: str,
    center_spec: CenterSpec,
    direction: str,
    cancel: threading.Event,
    drop: Sequence[str] = (),
    prune: bool = False,
) -> None:
    """
    Background worker of `_VSFServer.start_prefetch`: computes and caches
    the `/api/analyze` response for every other observed value of
    `target_col` (same certificate parameters), one value at a time through
    `vsf.avr.iter_branches_by_value`, so the exhaustive search over the
    column's feature set runs once for all of them. Each cached body is
    byte-identical to what a direct request for that value would produce:
    the branches are the same objects `discover_branches` returns for the
    0/1 indicator (pinned in `tests/test_fastpaths.py`), and the response is
    built by the same `_build_analyze_response`.

    Stops between values as soon as `cancel` is set. Values already cached
    or already being computed by a foreground request are skipped; a value
    this worker is computing is registered in `server.inflight`, so a
    foreground click on it waits for this result instead of duplicating it.
    """
    df = server.df
    try:
        values = [
            str(v) for v in df[target_col].dropna().unique().tolist()
            if str(v) != criterion
        ]
        params_of: Dict[str, Dict[str, Any]] = {}
        keys_of: Dict[str, Tuple[Any, ...]] = {}
        todo: List[str] = []
        for v in values:
            params = {"target_col": target_col, "criterion": v}
            params.update(_certificate_params(center_spec))
            params.update({"direction": direction, "drop": tuple(drop), "prune": bool(prune)})
            key = _analyze_key(params)
            if server.cache_get(key) is not None:
                continue
            params_of[v] = params
            keys_of[v] = key
            todo.append(v)
        if not todo or cancel.is_set():
            return
        X_df = df.drop(columns=[target_col])
        extra = [c for c in dict.fromkeys(drop) if c in X_df.columns]
        if extra:
            X_df = X_df.drop(columns=extra)
        feature_names = list(X_df.columns)
        X = X_df.values
        for v, branches in iter_branches_by_value(
            X,
            df[target_col].values,
            todo,
            feature_names=feature_names,
            max_d=_MAX_D,
            random_state=_SCAN_RANDOM_STATE,
            center_spec=center_spec,
            n_permutations_centers=DEFAULT_N_PERMUTATIONS,
            direction=direction,
            prune_dependent=prune,
        ):
            if cancel.is_set():
                return
            key = keys_of[v]
            body, event, owner = server.claim(key)
            if not owner:
                continue  # cached meanwhile, or a foreground request owns it
            assert event is not None
            try:
                if branches:
                    payload = _build_analyze_response(
                        server, _Target(target_col, v), center_spec, branches,
                        direction=direction, drop=drop, prune=prune,
                    )
                    server.cache_put(key, params_of[v], payload, json.dumps(payload).encode("utf-8"))
            finally:
                server.release(key, event)
    except Exception:  # pragma: no cover - a prefetch failure must never surface
        return


def _append_scan_record(
    scanned: List[Dict[str, Any]], col: str, val: object, branches: Dict[int, Any],
    direction: str = "presence",
) -> None:
    """
    One scan row for a (column, value) pair from its discovered branches.
    Under `direction="absence"` the coverage/purity/mass fields refer to the
    complement of the value (see `vsf.avr.Direction`); the row says so.
    """
    # Winner within the pair: highest coverage, ties broken toward
    # FEWER certified centres and then toward the LOWER
    # dimensionality — the same order as
    # `vsf.centers.coverage_score`, extended with a preference for
    # the simpler display when two branches are otherwise identical.
    best_d, best = max(
        branches.items(),
        key=lambda kv: (
            kv[1].centers.coverage if kv[1].centers else 0.0,
            -(kv[1].centers.n_centers if kv[1].centers else 0),
            -kv[0],
        ),
    )
    centers = best.centers
    p_used = (
        best.coverage_p_value_familywise
        if best.coverage_p_value_familywise is not None
        else (centers.coverage_p_value if centers else None)
    )
    cv = centers.coverage_cv if centers else None
    scanned.append({
        "column": col,
        "value": str(val),
        "direction": direction,
        "coverage": float(centers.coverage) if centers else 0.0,
        "n_centers": int(centers.n_centers) if centers else 0,
        "purity_pooled": float(centers.purity_pooled) if centers else 0.0,
        "mass": float(centers.mass) if centers else 0.0,
        "lift": float(centers.lift) if centers else 0.0,
        "n_positive": int(centers.n_positive) if centers else 0,
        "coverage_cv": None if cv is None else float(cv.mean),
        "coverage_cv_se": None if cv is None else float(cv.se),
        "undetermined": bool(centers.is_undetermined) if centers else True,
        "best_d": best_d,
        "best_features": best.selected_feature_names,
        "p_value": (
            None if centers is None or centers.coverage_p_value is None
            else float(centers.coverage_p_value)
        ),
        "p_value_familywise": (
            None if best.coverage_p_value_familywise is None
            else float(best.coverage_p_value_familywise)
        ),
        # 1.0 rather than None so a pair that produced no p-value is
        # carried through BH as an automatic non-rejection instead of
        # silently shrinking the family size m and inflating everyone
        # else's critical value.
        "_p": 1.0 if p_used is None else float(p_used),
    })


def _run_dataset_scan(
    server: "_VSFServer",
    threshold_pct: float,
    fdr_q: float,
    n_permutations: int,
    n_permutations_familywise: int,
    cancel_event: threading.Event,
    spec: CenterSpec,
    direction: str = "presence",
    drop: Sequence[str] = (),
    prune: bool = False,
) -> None:
    """
    Global Pattern Scan background worker (see module docstring). For every
    column of `server.df`, for every one of its observed values, runs
    `discover_branches` with that (column, value) as a One-vs-Rest binary
    target against every OTHER column as features — an honest exhaustive
    1D-4D search, identical in kind to a single `/api/analyze` call, just
    looped over every (column, value) pair in the dataset.

    Selection is a conjunction of two independent conditions, and both are
    load-bearing:

    1.  EFFECT SIZE. The maximum certified COVERAGE across the pair's
        branches must be at least `threshold_pct / 100` (inclusive — a pair
        certifying exactly 100% coverage clears a 100% threshold) — the
        share of the pair's target-value samples that fall inside cells
        certified to be at least `spec.tau` pure at simultaneous level
        `1 - spec.alpha`.

        v2.2 changed this from `u_adj` (an association statistic, "does
        knowing this pair's features tell you anything about the target");
        v2.3 removed `u_adj` from the codebase entirely (see `vsf.avr`'s
        module docstring). The scan's product is a list of (column, value)
        pairs worth DISPLAYING as discrete centres, and association is not
        that: on the UCI Adult / Census Income dataset, income = ">50K" used to read
        u_adj = 34.0% with zero certified centres at tau = 0.90, and
        occupation = "Armed-Forces" read 44.4% while the highest cell purity
        anywhere in its best branch was 2.42%. A scan filtered on
        association would have returned both; filtering on coverage returns
        neither, which is the correct behaviour for the question being
        asked.

    2.  SIGNIFICANCE, FDR-controlled over the whole swept family. Every pair
        contributes its winning branch's COVERAGE permutation p-value to one
        Benjamini-Hochberg procedure at rate `fdr_q`. Critically, BH runs over
        EVERY pair scanned — not only those clearing condition 1 — because
        filtering first and correcting afterwards is itself a selection effect
        and voids the guarantee. The p-value is drawn from the multivariate
        hypergeometric null of the cell counts given both margins, which is
        the exact permutation null of the coverage statistic.

    `n_permutations_familywise > 0` upgrades each pair's p-value from the
    uncorrected per-branch value to one corrected for the look-elsewhere
    effect of that pair's own C(M,1..4) subset scan. Without it, FDR is
    controlled across TARGETS but not across each target's internal search,
    which leaves the reported p-values anti-conservative; with it, both levels
    are covered, at roughly `n_permutations_familywise` times the per-pair
    search cost. Any published scan result must set it.

    The reader's column exclusions (`drop`, Section 4.12) are removed from
    every pair's feature space, and `prune` skips renaming candidates, so a
    scan searches the same feature space as the analyses beside it.

    Runs entirely in a background thread started by `_handle_scan_start_api`;
    progress is written to `server.scan_job` under `server.scan_lock` before
    every column so `/api/scan/status` always reflects the latest state.

    Search cost: the candidate partitions of a column's feature set do not
    depend on which of its values is the positive class, so all values of one
    column are searched in ONE enumeration of the family
    (`vsf.avr.discover_branches_by_value`) - one `bincount` per candidate
    yields the (cells x values) table every value's ranking key is read
    from. Each (column, value) pair's result is exactly what the per-pair
    `discover_branches` call produced; only the reporting stage (the
    per-branch permutation null, cross-validation and, when requested, the
    familywise null) remains per pair. Consequently `cancel_event` is checked
    between COLUMNS, not between pairs: a cancelled scan reports the columns
    it did complete, with BH applied to that completed family only.
    """
    df = server.df
    pairs: List[tuple] = []
    values_by_col: Dict[str, List[object]] = {}
    for col in df.columns:
        vals = df[col].dropna().unique().tolist()
        values_by_col[col] = vals
        for val in vals:
            pairs.append((col, val))
    total = len(pairs)

    scanned: List[Dict[str, Any]] = []
    # (column, value) pairs not searched because tau is not above the base
    # rate of the indicator (`vsf.avr.base_rate_reason`) - reported, not
    # silently dropped, since "this value is too common to localise at this
    # tau" is information the analyst needs.
    skipped_pairs: List[Dict[str, str]] = []
    try:
        done = 0
        for col in df.columns:
            vals = values_by_col[col]
            if not vals:
                continue
            if cancel_event.is_set():
                break

            with server.scan_lock:
                server.scan_job["progress"] = {
                    "current": done, "total": total, "label": f"{col} = {vals[0]}",
                }

            X_df = df.drop(columns=[col])
            extra = [c for c in dict.fromkeys(drop) if c in X_df.columns]
            if extra:
                X_df = X_df.drop(columns=extra)
            feature_names = list(X_df.columns)
            X = X_df.values

            col_skipped: Dict[str, str] = {}
            by_value = discover_branches_by_value(
                X,
                df[col].values,
                vals,
                feature_names=feature_names,
                max_d=_MAX_D,
                random_state=_SCAN_RANDOM_STATE,
                center_spec=spec,
                n_permutations_centers=n_permutations,
                n_permutations_familywise_coverage=n_permutations_familywise,
                # The scan record reads no per-cell interval (see
                # `_append_scan_record`); skipping them is the difference
                # between a scan bounded by the search and one bounded by
                # continued-fraction inversions it throws away.
                cell_bounds=False,
                direction=direction,
                skipped=col_skipped,
                prune_dependent=prune,
            )
            done += len(vals)
            for val in vals:
                if str(val) in col_skipped:
                    skipped_pairs.append({
                        "column": col, "value": str(val), "reason": col_skipped[str(val)],
                    })
                    continue
                branches = by_value.get(str(val), {})
                if not branches:
                    continue
                _append_scan_record(scanned, col, val, branches, direction)

        # BH over the ENTIRE completed family, before any effect-size filter.
        rejected = benjamini_hochberg([r["_p"] for r in scanned], q=fdr_q)
        results: List[Dict[str, Any]] = []
        for record, keep in zip(scanned, rejected):
            record["fdr_significant"] = bool(keep)
            record.pop("_p", None)
            if keep and record["coverage"] >= threshold_pct / 100.0:
                results.append(record)

        results.sort(
            key=lambda r: (r["coverage"], -r["n_centers"]), reverse=True
        )
        cancelled = cancel_event.is_set()

        with server.scan_lock:
            server.scan_job["status"] = "cancelled" if cancelled else "done"
            final_current = server.scan_job["progress"]["current"] if cancelled else total
            server.scan_job["progress"] = {"current": final_current, "total": total, "label": ""}
            server.scan_job["results"] = results
            server.scan_job["n_tested"] = len(scanned)
            server.scan_job["n_fdr_significant"] = int(sum(rejected))
            server.scan_job["skipped"] = skipped_pairs
            server.scan_job["n_skipped"] = len(skipped_pairs)
    except Exception as e:
        with server.scan_lock:
            server.scan_job["status"] = "error"
            server.scan_job["error"] = str(e)


def _build_server(
    df: pd.DataFrame,
    host: str = "127.0.0.1",
    port: int = 8000,
    translations: Optional[Translations] = None,
    prefetch: bool = True,
) -> _VSFServer:
    """
    Constructs (binds + listens, does NOT `serve_forever()`) the
    `_VSFServer` instance `serve()` runs. Split out as a private,
    independently-testable seam — mirrors `vsf/dashboard.py`'s internal
    build pattern — so tests can construct a real server, drive it with
    real HTTP requests on an ephemeral port, and tear it down explicitly,
    without blocking on `serve_forever()`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("df must not be empty")
    if len(df.columns) == 0:
        raise ValueError("df must have at least one column")

    return _VSFServer(
        (host, port), VSFRequestHandler, df=df, translations=translations, prefetch=prefetch
    )


def serve(
    df: pd.DataFrame,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    translations: Optional[Dict] = None,
    prefetch: bool = True,
) -> None:
    """
    Starts a local HTTP server and opens the browser with the full
    interactive VSF visualizer for the provided dataset.

    Unlike `vsf.export_full_dashboard()` (a static, `file://`-openable HTML
    export with precomputed branches baked in), `vsf.serve(df)` runs the
    LIVE application against `df` in-process: Independent Branch Discovery
    is computed on demand by this process for whatever target/criterion the
    user selects.

    The page's initial target column is `"class"` if `df` has a column by
    that name, else `df`'s first column (see `_default_target`); the
    catalog panel lets the user click any other column afterward, so this
    function takes no separate `target` argument.

    Binds to `host` (default `127.0.0.1`, loopback-only — binding
    `""`/`0.0.0.0` would expose this dataset and the analyze endpoint,
    which performs no authentication, to the rest of the local network).
    Pass an explicit `host="0.0.0.0"` only if you specifically intend that
    exposure.

    Blocks the calling thread in `serve_forever()` until interrupted
    (Ctrl+C / `KeyboardInterrupt`), then shuts the server down cleanly —
    matching the `streamlit run`/`gradio.launch()` foreground-process UX
    this function is modeled on.

    Args:
        df: the dataset to visualize. Must be a non-empty
            `pandas.DataFrame` with at least one column.
        host: interface to bind. Default `127.0.0.1` (loopback only).
        port: TCP port to bind. Default 8000. Raises `OSError` if already
            in use — pass a different port or free the existing one.
        open_browser: if `True` (default), opens the default system
            browser to `http://<host>:<port>/` once the server is bound
            and listening (before `serve_forever()` — the OS queues any
            connection that arrives in the meantime, so no race is
            possible here).
        translations: optional dataset-specific display table (see
            `vsf.vis.Translations`) — column/value human-readable labels
            for the catalog and analysis output. `None` (the default)
            falls back to raw column/value strings, same as every other
            `vsf` function.
        prefetch: if `True` (default), after each criterion-mode analysis
            the server computes the responses for the column's other values
            in a background thread (`_prefetch_sibling_values`) so that the
            next clicks in that column are served from the response cache.
            Pass `False` on a machine whose CPU must stay free between
            requests; results are identical either way.
    """
    httpd = _build_server(df, host=host, port=port, translations=translations, prefetch=prefetch)
    url = f"http://{host}:{port}/"
    try:
        print("=" * 70)
        print(" VSF Interactive Visual Dashboard Server Running!")
        print(f" Local URL: {url}")
        print(f" Default target column: {httpd.default_target!r}")
        print(" Press Ctrl+C to stop.")
        print("=" * 70)

        if open_browser:
            webbrowser.open(url)

        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down VSF server...")
    finally:
        httpd.server_close()
