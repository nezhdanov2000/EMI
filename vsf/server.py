"""
vsf.server: `vsf.serve(df)` — an installable, in-process interactive VSF web
application for an ARBITRARY pandas DataFrame with one call, the way
`streamlit.run`/`gradio.Interface.launch` do.

v2.0 "Clean Core" (see Project_Master_Document.md Section 0 for the full
revision history): the legacy top-level `server.py` demo script (a
single-CSV, single-dataset script duplicating this module's request
dispatch for the bundled mushroom dataset) has been deleted outright rather
than updated — it could not survive the `AVREngine` -> `BranchEngine` API
change without a rewrite indistinguishable from this module, so keeping both
was pure duplication. `vsf.serve(df)` is now the ONLY live-server entry
point; `tests/test_server.py` (which asserted on the legacy script's
internals) was deleted alongside it.

Removed relative to v1.0 (all four are the same product decisions recorded
in Project_Master_Document.md Section 0, applied here to the live-server
surface):
  - `/api/mine_center` (dirty-center conjunctive-filter mining) — its sole
    implementation, `vsf.mining.mine_dirty_center`, no longer exists.
  - `/api/top_columns` (Auto-Discovery / Universal Propositional Screening)
    — its implementation, `vsf.mining.compute_top_insights`, no longer
    exists; the user now picks the target column and criterion directly
    from the dropdown populated by `/api/columns` (UI_Functional_Spec.md
    Section 2), never from a "here's what's interesting" catalog.
  - `/api/graph_inference` and `/api/mine_graph_links` (Graph Inference /
    Knowledge-Base chain mining) and the `graph.html` /
    `static/{css,js}/graph_reasoning.*` assets that rendered them — the
    reasoning-graph feature is gone, not just its route.
  - `composite_target` support inside `/api/analyze` (the composite AND
    -filter target builder) — a target is now always a single column
    (+ optional single criterion value), matching `UI_Functional_Spec.md`
    Section 2's plain dropdown.

`/api/analyze` itself changed shape, not just scope: it used to fit ONE
`AVRResult` (a single adaptively-chosen `d_star`) per request. It now runs
Independent Branch Discovery (`vsf.avr.discover_branches`,
Project_Master_Document.md Section 4) and returns UP TO `MAX_BRANCH_D`
independently-found branches in one response — one visualization payload
per dimensionality — so the frontend's branch selector
(UI_Functional_Spec.md Section 3) can switch between them instantly,
client-side, with no further request (mirroring how `vsf.dashboard.
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
if the MAXIMUM `u_adj` across its up-to-4 branches exceeds a caller-chosen
threshold AND the pair survives Benjamini-Hochberg FDR control across every
pair scanned in that run.

Two things changed here in v2.1, both mandatory rather than cosmetic. The
filter used to be `max NMI > threshold`, and NMI_min is precisely the metric
that saturates toward 100 % on a rare One-vs-Rest criterion with no signal
(see `vsf.math.normalized_mutual_information`) — a scan that enumerates every
(column, value) pair generates rare criteria by the hundred, so the old scan
selected FOR the artifact it was most exposed to. And a sweep of k targets at
a nominal alpha yields ~alpha*k spurious "patterns" by construction, so FDR
control across the swept family is not optional; `metric_fdr_q` sets the rate.

This is still NOT a return of v1.0's removed Auto-Discovery/Universal
Propositional Screening (see Section 0 of the master doc for why that was
cut): v1.0 used permutation tests to GATE greedy feature selection inside a
single analysis, whereas the scan tests already-selected, exhaustively
searched branches and corrects only for the multiplicity the scan itself
creates. That looping makes it genuinely expensive
— one `discover_branches` call per pair, each itself an exhaustive 1D-4D
search (see `vsf.avr`'s module docstring on cost) — so it runs in a
background thread (`_run_dataset_scan`), polled via `/api/scan/status`
rather than returned synchronously, with `/api/scan/cancel` to stop early.
Only one scan may run at a time per server instance; state lives on
`_VSFServer.scan_job`/`.scan_lock`/`.scan_cancel_event`, so — like
everything else in this module — two concurrently-running `serve()` calls
never share a scan.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
import urllib.parse
import webbrowser
from importlib import resources
from typing import Any, Dict, List, Optional

import numpy as _np
import pandas as pd

from . import vis as _vis
from .avr import (
    DEFAULT_N_PERMUTATIONS,
    MAX_BRANCH_D,
    discover_branches,
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

    def __init__(self, server_address, RequestHandlerClass, df: pd.DataFrame, translations: Optional[Translations]):
        super().__init__(server_address, RequestHandlerClass)
        self.df = df
        self.translations = translations
        self.default_target = _default_target(df)
        self.cache_lock = threading.Lock()
        self.last_params: Optional[Dict[str, Any]] = None
        self.last_branches_payload: Optional[Dict[str, Any]] = None
        # Global Pattern Scan state (see module docstring). `scan_job` is
        # None until the first scan starts; `scan_cancel_event` is a fresh
        # threading.Event() per scan, set by `/api/scan/cancel` and polled
        # by `_run_dataset_scan`'s background thread between pairs.
        self.scan_lock = threading.Lock()
        self.scan_job: Optional[Dict[str, Any]] = None
        self.scan_cancel_event: Optional[threading.Event] = None


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
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _serve_static(self, path: str) -> None:
        relative_path, content_type = _STATIC_ROUTES[path]
        try:
            body = _read_webapp_asset(relative_path)
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            self.send_error(404, f"Asset not found: {exc}")
            return
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
        branch selector (UI_Functional_Spec.md Section 3) can render and
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
            try:
                center_spec = CenterSpec(
                    tau=tau, alpha=alpha, rule=rule, min_samples=min_samples
                )
            except ValueError as exc:
                self._send_json_response(400, {"error": str(exc)})
                return

            cache_key = {
                "target_col": target_col,
                "criterion": criterion,
                "tau": tau,
                "alpha": alpha,
                "rule": rule,
                "min_samples": min_samples,
            }

            # Same rationale as before: holding `cache_lock` across the
            # check-and-fit means two concurrent requests for different
            # targets never interleave a read of one target's half-written
            # cache with another's write — the second request just waits
            # instead of racing.
            with self.server.cache_lock:
                if self.server.last_params == cache_key and self.server.last_branches_payload is not None:
                    response_payload = self.server.last_branches_payload
                else:
                    # `n_permutations` costs ~0.25 s per branch at B = 999 on
                    # a 32k-row dataset (the null is sampled directly from the
                    # multiple hypergeometric law, not by shuffling labels), so
                    # the interactive path can afford the p-value that the HUD
                    # needs to tell a real association from a bias artifact.
                    # `n_permutations_familywise` is NOT set here: it costs B
                    # times the whole search and belongs to an offline run, not
                    # to a click. The HUD labels the p-value it shows
                    # accordingly.
                    # v2.2: `positive_class` is the value the certified
                    # centres are measured against. In criterion mode Z is
                    # literally 0/1 and the positive value is 1. Without a
                    # criterion the target may have K > 2 values, which have
                    # no single purity; `discover_branches` then returns
                    # `centers=None` and the panel says so rather than
                    # certifying an arbitrary class.
                    positive_class = 1 if criterion is not None else None
                    branches = discover_branches(
                        X,
                        Z,
                        feature_names=feature_names,
                        max_d=_MAX_D,
                        n_permutations=DEFAULT_N_PERMUTATIONS,
                        random_state=_SCAN_RANDOM_STATE,
                        positive_class=positive_class,
                        center_spec=center_spec,
                        n_permutations_centers=DEFAULT_N_PERMUTATIONS,
                    )
                    # `discover_branches` is called with objective="auto"
                    # (the default -- never overridden here), which resolves
                    # to "coverage" exactly when a positive class was
                    # resolvable and to "mi_adj" otherwise (see
                    # `avr._resolve_positive_indicator`'s docstring). Rather
                    # than re-deriving that condition here and risking it
                    # drifting from what the search actually did, read it
                    # off the same signal `discover_branches` itself gates
                    # `BranchResult.centers` on: centers is not None on some
                    # branch iff a positive class was resolved for this call.
                    objective_used = (
                        "coverage" if any(b.centers is not None for b in branches.values())
                        else "mi_adj"
                    )
                    branches_data: Dict[str, Any] = {}
                    for d, branch in branches.items():
                        branches_data[str(d)] = prepare_visualization_payload(
                            branch,
                            X,
                            Z,
                            feature_names=feature_names,
                            target_name=display_target_name,
                            # Z is a One-vs-Rest 0/1 vector in criterion mode,
                            # so its class labels must read as the criterion
                            # and its negation, not as "0" and "1".
                            target_is_indicator=criterion is not None,
                            sort_Z=sort_Z,
                            translations=translations,
                            positive_value=(
                                1 if criterion is not None
                                else (
                                    _np.unique(Z)[-1] if len(_np.unique(Z)) else None
                                )
                            ),
                            center_spec=center_spec,
                        )
                    selected_d = select_branch_dimensionality(branches)
                    response_payload = {
                        "target": target_col,
                        "criterion": criterion,
                        "branches": branches_data,
                        "branch_dims": sorted(branches.keys()),
                        # v2.2: the branch opened first is the SMALLEST
                        # SUFFICIENT one, not the widest available. The
                        # product goal is the fewest cells that capture the
                        # target value, and `max(branches)` is the opposite
                        # of that: on `relationship = Husband` the 4-D branch
                        # certifies 3 centres for 98.6 % coverage where the
                        # 2-D branch certifies 1 for 99.9 %. Falls back to the
                        # widest branch only when no dimensionality certifies
                        # anything, since there is then nothing to prefer.
                        "default_branch": (
                            str(selected_d) if selected_d in branches
                            else (str(max(branches.keys())) if branches else None)
                        ),
                        # Which objective the search actually maximised for
                        # THIS response (see the comment above where this is
                        # computed) -- consumed by the frontend so the
                        # branch-list caption describes reality instead of
                        # being a static, version-drifting string.
                        "objective_used": objective_used,
                        # v2.2: the answer to "how many characteristics does
                        # it take to describe this value" -- the smallest
                        # sufficient dimensionality by out-of-sample certified
                        # coverage, or None when no dimensionality certifies
                        # anything. The frontend must render None as "none",
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
                    self.server.last_params = cache_key
                    self.server.last_branches_payload = response_payload

            self._send_json_response(200, response_payload)
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
                        "nmi_threshold was removed in v2.1. NMI_min saturates toward "
                        "100% on rare One-vs-Rest criteria with no signal, which is "
                        "exactly what a dataset-wide scan produces in bulk. Use "
                        "coverage_threshold (share of the target value captured by "
                        "certified centres, same [0, 100) percent scale)."
                    )
                })
                return
            if "u_adj_threshold" in req and "coverage_threshold" not in req:
                self._send_json_response(400, {
                    "error": (
                        "u_adj_threshold was replaced in v2.2 by coverage_threshold. "
                        "u_adj measures whether an association EXISTS; the scan's "
                        "purpose is to find (column, value) pairs that produce "
                        "certified discrete centres, and the two select different "
                        "pairs. On data/adult_census.csv, income='>50K' reaches "
                        "u_adj = 34.0% with zero certified centres at tau = 0.90, "
                        "and occupation='Armed-Forces' reaches 44.4% with a highest "
                        "cell purity of 2.42%. Pass coverage_threshold (percent), "
                        "and optionally tau and alpha."
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
                cancel_event = threading.Event()
                self.server.scan_cancel_event = cancel_event
                self.server.scan_job = {
                    "status": "running",
                    "threshold_pct": threshold_pct,
                    "tau": scan_spec.tau,
                    "alpha": scan_spec.alpha,
                    "rule": scan_spec.rule,
                    "min_samples": scan_spec.min_samples,
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
                    cancel_event, scan_spec,
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
        CURRENT (column, value) pair finishes — `discover_branches` itself
        isn't interrupted mid-search, so cancellation lands within one
        pair's worth of time, not instantly. A no-op (200) if no scan is
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


def _run_dataset_scan(
    server: "_VSFServer",
    threshold_pct: float,
    fdr_q: float,
    n_permutations: int,
    n_permutations_familywise: int,
    cancel_event: threading.Event,
    spec: CenterSpec,
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

        v2.2 changed this from `u_adj`, and the search objective with it, for
        one reason: the scan's product is a list of (column, value) pairs
        worth DISPLAYING as discrete centres, and u_adj does not measure
        that. On `data/adult_census.csv`, income = ">50K" reaches
        u_adj = 34.0% with zero certified centres at tau = 0.90, and
        occupation = "Armed-Forces" reaches 44.4% while the highest cell
        purity anywhere in its best branch is 2.42%. A scan filtered on
        u_adj returns both; a scan filtered on coverage returns neither, and
        that is the correct behaviour for the question being asked. u_adj is
        still recorded per pair as a diagnostic, so the divergence between
        "an association exists" and "a centre exists" stays visible in the
        results rather than being decided silently.

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
    progress is written to `server.scan_job` under `server.scan_lock` after
    every pair so `/api/scan/status` always reflects the latest state. Checks
    `cancel_event` between pairs (not mid-`discover_branches` — see
    `_handle_scan_cancel_api`'s docstring); a cancelled scan reports the pairs
    it did complete, with BH applied to that completed family only.
    """
    df = server.df
    pairs: List[tuple] = []
    for col in df.columns:
        for val in df[col].dropna().unique().tolist():
            pairs.append((col, val))
    total = len(pairs)

    scanned: List[Dict[str, Any]] = []
    try:
        for i, (col, val) in enumerate(pairs):
            if cancel_event.is_set():
                break

            with server.scan_lock:
                server.scan_job["progress"] = {"current": i, "total": total, "label": f"{col} = {val}"}

            Z = (df[col].astype(str) == str(val)).astype(int).values
            X_df = df.drop(columns=[col])
            feature_names = list(X_df.columns)
            X = X_df.values

            branches = discover_branches(
                X,
                Z,
                feature_names=feature_names,
                max_d=_MAX_D,
                n_permutations=n_permutations,
                n_permutations_familywise=n_permutations_familywise,
                random_state=_SCAN_RANDOM_STATE,
                objective="coverage",
                positive_class=1,
                center_spec=spec,
                n_permutations_centers=n_permutations,
                n_permutations_familywise_coverage=n_permutations_familywise,
            )
            if not branches:
                continue

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
                "coverage": float(centers.coverage) if centers else 0.0,
                "n_centers": int(centers.n_centers) if centers else 0,
                "purity_pooled": float(centers.purity_pooled) if centers else 0.0,
                "mass": float(centers.mass) if centers else 0.0,
                "lift": float(centers.lift) if centers else 0.0,
                "n_positive": int(centers.n_positive) if centers else 0,
                "coverage_cv": None if cv is None else float(cv.mean),
                "coverage_cv_se": None if cv is None else float(cv.se),
                "undetermined": bool(centers.is_undetermined) if centers else True,
                # Diagnostic only: retained so that the divergence between
                # "an association exists" and "a centre exists" is visible in
                # the scan output instead of being resolved silently.
                "max_u_adj": float(best.u_adj),
                "best_d": best_d,
                "best_mi": float(best.mi),
                "best_mi_null": float(best.mi_null),
                "best_mi_adj": float(best.mi_adj),
                "best_features": best.selected_feature_names,
                "p_value": (
                    None if centers is None or centers.coverage_p_value is None
                    else float(centers.coverage_p_value)
                ),
                "p_value_familywise": (
                    None if best.coverage_p_value_familywise is None
                    else float(best.coverage_p_value_familywise)
                ),
                "p_value_mi": None if best.p_value is None else float(best.p_value),
                # 1.0 rather than None so a pair that produced no p-value is
                # carried through BH as an automatic non-rejection instead of
                # silently shrinking the family size m and inflating everyone
                # else's critical value.
                "_p": 1.0 if p_used is None else float(p_used),
            })

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
    except Exception as e:
        with server.scan_lock:
            server.scan_job["status"] = "error"
            server.scan_job["error"] = str(e)


def _build_server(
    df: pd.DataFrame,
    host: str = "127.0.0.1",
    port: int = 8000,
    translations: Optional[Translations] = None,
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

    return _VSFServer((host, port), VSFRequestHandler, df=df, translations=translations)


def serve(
    df: pd.DataFrame,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    translations: Optional[Dict] = None,
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
    """
    httpd = _build_server(df, host=host, port=port, translations=translations)
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
