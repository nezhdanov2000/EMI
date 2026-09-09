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
    -filter target builder) — a target is now always a single column
    (+ optional single criterion value), matching the webapp's plain
    dropdown.

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
"""

from __future__ import annotations

import http.server
import json
import threading
import time
import urllib.parse
import webbrowser
from collections import OrderedDict
from importlib import resources
from typing import Any, Dict, List, Optional, Tuple

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
#: Grid step of the tau-curves, in percent of purity.
_CURVES_STEP_PCT = 1.0

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

    def get_landscape(self, params: Dict[str, Any], center_spec: CenterSpec):
        """The `Landscape` for an analyze parameter set, computed once."""
        key = _analyze_key(params)
        with self.cache_lock:
            hit = self.landscape_cache.get(key)
            if hit is not None:
                self.landscape_cache.move_to_end(key)
                return hit
        X_df, Z = self._target_arrays(params["target_col"], params["criterion"])
        criterion = params["criterion"]
        landscape = compute_landscape(
            X_df.values, Z, feature_names=list(X_df.columns), max_d=_MAX_D,
            positive_class=(1 if criterion is not None else None),
            center_spec=center_spec, direction=params["direction"],
        )
        with self.cache_lock:
            self.landscape_cache[key] = landscape
            while len(self.landscape_cache) > _LANDSCAPE_CACHE_MAX_ENTRIES:
                self.landscape_cache.popitem(last=False)
        return landscape

    def _target_arrays(self, target_col: str, criterion: Optional[object]):
        X_df = self.df.drop(columns=[target_col])
        Z = (
            (self.df[target_col].astype(str) == str(criterion)).astype(int).values
            if criterion is not None else self.df[target_col].values
        )
        return X_df, Z

    def get_tau_curves(self, params: Dict[str, Any], center_spec: CenterSpec) -> Dict[str, Any]:
        """
        The tau-curves (`compute_tau_curves`) for an analyze parameter set,
        computed once per parameters-without-tau: `center_spec.tau` does not
        enter the key, the curve is the dependence on it.
        """
        key = _analyze_key({k: v for k, v in params.items() if k != "tau"})
        with self.cache_lock:
            hit = self.curves_cache.get(key)
            if hit is not None:
                self.curves_cache.move_to_end(key)
                return hit
        X_df, Z = self._target_arrays(params["target_col"], params["criterion"])
        curves = compute_tau_curves(
            X_df.values, Z, feature_names=list(X_df.columns), max_d=_MAX_D,
            positive_class=(1 if params["criterion"] is not None else None),
            center_spec=center_spec, direction=params["direction"],
            step_pct=_CURVES_STEP_PCT,
        )
        with self.cache_lock:
            self.curves_cache[key] = curves
            while len(self.curves_cache) > _CURVES_CACHE_MAX_ENTRIES:
                self.curves_cache.popitem(last=False)
        return curves

    # -- background prefetch ------------------------------------------------
    def start_prefetch(
        self, target_col: str, criterion: str, center_spec: CenterSpec,
        direction: str = "presence",
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
                args=(self, target_col, criterion, center_spec, direction, cancel),
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
        elif path == "/api/landscape/curves":
            self._handle_curves_api(at=False)
        elif path == "/api/landscape/at":
            self._handle_curves_api(at=True)
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

            target_col = req.get("target", self.server.default_target)
            criterion = req.get("criterion", None)

            if target_col not in df.columns:
                self._send_json_response(
                    400,
                    {"error": f"target {target_col!r} is not a column of this dataset"},
                )
                return

            sort_Z = df[target_col].values

            if criterion is not None:
                Z = (df[target_col].astype(str) == str(criterion)).astype(int).values
                human_criterion = _vis.humanize_val(target_col, str(criterion), translations)
                human_col = _vis.humanize_col(target_col, translations)
                display_target_name = f"{human_col} = {human_criterion}"
            else:
                Z = df[target_col].values
                display_target_name = target_col

            X_df = df.drop(columns=[target_col])
            feature_names = list(X_df.columns)
            X = X_df.values

            # Certificate parameters are part of the cache key: changing tau
            # or alpha changes every centre, every colour and every reported
            # number, so a cached payload computed at a different tau must not
            # be served.
            try:
                tau = float(req.get("tau", 0.90))
                alpha = float(req.get("alpha", 0.05))
                min_samples = int(req.get("min_samples", 1))
            except (TypeError, ValueError):
                self._send_json_response(
                    400, {"error": "tau, alpha and min_samples must be numbers"}
                )
                return
            rule = req.get("rule", "purity")
            if rule not in ("purity", "certified"):
                self._send_json_response(
                    400,
                    {"error": f"rule must be 'purity' or 'certified', got {rule!r}"},
                )
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
            try:
                center_spec = CenterSpec(
                    tau=tau, alpha=alpha, rule=rule, min_samples=min_samples
                )
            except ValueError as exc:
                self._send_json_response(400, {"error": str(exc)})
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

            cache_key = {
                "target_col": target_col,
                "criterion": criterion,
                "tau": tau,
                "alpha": alpha,
                "rule": rule,
                "min_samples": min_samples,
                "direction": direction,
            }
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
                        )
                    response_payload = _build_analyze_response(
                        self.server, target_col, criterion, center_spec, branches,
                        direction=direction, features=features,
                    )
                    body = json.dumps(response_payload).encode("utf-8")
                    self.server.cache_put(key, cache_key, response_payload, body)
                finally:
                    self.server.release(key, event)
                if criterion is not None and features is None:
                    self.server.start_prefetch(
                        target_col, str(criterion), center_spec, direction
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
        target, criterion, tau, alpha, rule, min_samples, direction.
        """
        df = self.server.df
        target_col = req.get("target", self.server.default_target)
        criterion = req.get("criterion", None)
        if target_col not in df.columns:
            self._send_json_response(400, {"error": f"target {target_col!r} is not a column of this dataset"})
            return None
        try:
            tau = float(req.get("tau", 0.90))
            alpha = float(req.get("alpha", 0.05))
            min_samples = int(req.get("min_samples", 1))
        except (TypeError, ValueError):
            self._send_json_response(400, {"error": "tau, alpha and min_samples must be numbers"})
            return None
        rule = req.get("rule", "purity")
        direction = req.get("direction", "presence")
        if rule not in ("purity", "certified") or direction not in ("presence", "absence"):
            self._send_json_response(400, {"error": "invalid rule or direction"})
            return None
        if direction == "absence" and criterion is None:
            self._send_json_response(400, {"error": "an absence landscape needs an explicit criterion"})
            return None
        try:
            center_spec = CenterSpec(tau=tau, alpha=alpha, rule=rule, min_samples=min_samples)
        except ValueError as exc:
            self._send_json_response(400, {"error": str(exc)})
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
        params = {
            "target_col": target_col, "criterion": criterion, "tau": tau,
            "alpha": alpha, "rule": rule, "min_samples": min_samples,
            "direction": direction,
        }
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
            landscape = self.server.get_landscape(params, center_spec)
            if not cell:
                out = landscape.bins(d)
                out.update({
                    "target": target_col, "criterion": criterion, "direction": direction,
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
                curves = self.server.get_tau_curves(params, center_spec)
                out = dict(curves)
                out.update({"target": target_col, "criterion": criterion, "direction": direction,
                            "rule": params["rule"]})
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
            landscape = self.server.get_landscape(params, center_spec)
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
                    min_samples=scan_min_samples,
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

            thread = threading.Thread(
                target=_run_dataset_scan,
                args=(
                    self.server, threshold_pct, fdr_q, n_perm, n_perm_fw,
                    cancel_event, scan_spec, scan_direction,
                ),
                daemon=True,
            )
            thread.start()
            self._send_json_response(200, {"status": "started"})
        except Exception as e:
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
    """Hashable cache key of an `/api/analyze` request's parameters."""
    return tuple(sorted((str(k), repr(v)) for k, v in params.items()))


def _build_analyze_response(
    server: "_VSFServer",
    target_col: str,
    criterion: Optional[object],
    center_spec: CenterSpec,
    branches: Dict[int, Any],
    direction: str = "presence",
    features: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    The `/api/analyze` response body for discovered `branches`: one
    visualisation payload per dimensionality plus the branch selector's
    metadata. Shared by the request handler and the background prefetch so
    the two can never disagree on a field.
    """
    df = server.df
    translations = server.translations
    sort_Z = df[target_col].values
    indicator_labels = None
    if criterion is not None:
        Z = (df[target_col].astype(str) == str(criterion)).astype(int).values
        human_criterion = _vis.humanize_val(target_col, str(criterion), translations)
        human_col = _vis.humanize_col(target_col, translations)
        display_target_name = f"{human_col} = {human_criterion}"
        if direction == "absence":
            # The renderer's "positive" indicator is the complement: every
            # cell purity, bound and certificate it computes must refer to
            # rows WITHOUT the value, exactly as the search did. The class
            # labels keep naming the value, so a point's hover still reads
            # "Column = value" / "not Column = value".
            Z = 1 - Z
            indicator_labels = (display_target_name, f"not {display_target_name}")
            display_target_name = f"{human_col} \u2260 {human_criterion}"
    else:
        Z = df[target_col].values
        display_target_name = target_col
    X_df = df.drop(columns=[target_col])
    feature_names = list(X_df.columns)
    X = X_df.values

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
        "certificate": {
            "tau": center_spec.tau,
            "alpha": center_spec.alpha,
            "rule": center_spec.rule,
            "min_samples": center_spec.min_samples,
            "method": center_spec.method,
            "multiplicity": center_spec.multiplicity,
        },
    }


def _prefetch_sibling_values(
    server: "_VSFServer",
    target_col: str,
    criterion: str,
    center_spec: CenterSpec,
    direction: str,
    cancel: threading.Event,
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
            params = {
                "target_col": target_col, "criterion": v,
                "tau": center_spec.tau, "alpha": center_spec.alpha,
                "rule": center_spec.rule, "min_samples": center_spec.min_samples,
                "direction": direction,
            }
            key = _analyze_key(params)
            if server.cache_get(key) is not None:
                continue
            params_of[v] = params
            keys_of[v] = key
            todo.append(v)
        if not todo or cancel.is_set():
            return
        X_df = df.drop(columns=[target_col])
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
                        server, target_col, v, center_spec, branches, direction=direction
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
