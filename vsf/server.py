"""
vsf.server: `vsf.serve(df)` — an installable, in-process equivalent of the
repository's standalone `server.py` demo script, for launching the full
interactive VSF web application on an ARBITRARY pandas DataFrame with one
call, the way `streamlit.run`/`gradio.Interface.launch` do.

Architectural relationship to the top-level `server.py` (READ THIS FIRST):
this module is a deliberately INDEPENDENT reimplementation, not a refactor
of `server.py` into a shared core. Three reasons, in order of importance:

  1. `server.py` is a single-CSV, single-dataset demo script: it hardcodes
     `DATASET_PATH` (the bundled `data/mushrooms.csv`) and
     `MUSHROOM_TRANSLATIONS`, does `pd.read_csv(DATASET_PATH)` fresh inside
     every request handler, and keeps its request-scoped cache
     (`_last_params`/`_last_res`/`_top_columns_cache`) as MODULE-LEVEL
     globals. That's safe for a script that starts once per process against
     one fixed dataset. It is NOT safe for `vsf.serve(df)`, which must
     support being called more than once in the same process (e.g. from a
     notebook, once per dataset a user wants to look at) without one
     server's cache or dataframe leaking into another's. This module moves
     every piece of that state onto the constructed HTTP server instance
     (`_VSFServer.df` / `.translations` / `.default_target` / `.cache_lock`
     / `.last_params` / `.last_res` / `.top_columns_cache`), so two
     concurrently-running `serve()` calls in one process are fully
     isolated from each other.
  2. `server.py`'s frontend assets (`index.html`, `graph.html`,
     `static/css/*`, `static/js/*`) live at the repository root, served via
     `http.server.SimpleHTTPRequestHandler`'s cwd-relative file serving
     (`main()` does `os.chdir(os.path.dirname(__file__))` before binding).
     That only works from a checkout run in-place; it cannot work for an
     installed `vsf` package invoked from an arbitrary caller directory.
     This module instead serves a PACKAGED COPY of those assets
     (`vsf/webapp/`) via `importlib.resources`, mirroring the exact pattern
     `vsf/dashboard.py` already established for `vsf/templates/`.
  3. `tests/test_server.py` asserts directly on `server.py`'s internals
     (`srv._cache_lock`, `srv.MUSHROOM_TRANSLATIONS` via `assertIs`, the
     hardcoded `"default_target": "class"` response, and the `/api/mushroom`
     endpoint contract). Refactoring `server.py` to delegate to this module
     would either break that suite or force it to special-case the mushroom
     dataset — defeating the point of a dataset-agnostic `serve()`. Keeping
     the two implementations separate costs some duplication (the request
     dispatch and target-preparation logic below intentionally parallels
     `server.py`'s) in exchange for zero risk to already-passing tests and a
     genuinely generic `vsf.serve()`.

Dataset-agnosticism: `default_target` is `"class"` if the dataframe has a
column literally named "class" (matching the bundled mushroom demo's
convention), otherwise the dataframe's FIRST column. This is only the
column shown when the page first loads — the frontend's catalog lets the
user click any other column afterward (`/api/columns` reports
`default_target` and the packaged `vsf/webapp/static/js/app.js` reads it
instead of hardcoding "class"; see that file's `init()`). `serve()` itself
takes no `target` parameter for exactly that reason.

Not ported (same scope cut as `vsf.dashboard.export_full_dashboard`, for
the same reasons — see that module's docstring): nothing is cut here,
actually — unlike the static export, a live server CAN service the
composite AND-filter builder and the Graph Inference / Knowledge-Base
chain-mining endpoints (`/api/graph_inference`, `/api/mine_graph_links`)
because they run against the live `df`, not a precomputed payload. Both
are implemented below, dataset-agnostically (no hardcoded target/criterion
fallbacks).
"""

from __future__ import annotations

import http.server
import json
import threading
import urllib.parse
import webbrowser
from importlib import resources
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import vis as _vis
from .avr import AVREngine
from .mining import compute_top_insights, mine_dirty_center
from .vis import Translations, catalog_from_dataframe, prepare_visualization_payload

__all__ = ["serve"]

# AVR fit parameters for the live `/api/analyze`-equivalent endpoint — match
# `server.py`'s hardcoded values exactly, so `vsf.serve(df)` reproduces the
# same fit the standalone demo script would for the same data.
_ALPHA = 0.01
_VIR_THRESHOLD = 0.85
_MAX_D = 7
_N_PERMUTATIONS = 100
_RANDOM_STATE = 42

# Static asset content-types served from the packaged `vsf.webapp` resources.
_STATIC_ROUTES: Dict[str, tuple] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/graph.html": ("graph.html", "text/html; charset=utf-8"),
    "/static/css/styles.css": ("static/css/styles.css", "text/css; charset=utf-8"),
    "/static/css/graph_reasoning.css": (
        "static/css/graph_reasoning.css",
        "text/css; charset=utf-8",
    ),
    "/static/js/app.js": ("static/js/app.js", "application/javascript; charset=utf-8"),
    "/static/js/graph_reasoning.js": (
        "static/js/graph_reasoning.js",
        "application/javascript; charset=utf-8",
    ),
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
    `"class"` if present (matching the bundled mushroom demo's convention
    and `server.py`'s hardcoded behavior), else the dataframe's first
    column. This is only the page's INITIAL target; the frontend catalog
    lets the user click any other column afterward.
    """
    return "class" if "class" in df.columns else str(df.columns[0])


class _VSFServer(http.server.ThreadingHTTPServer):
    """
    A `ThreadingHTTPServer` carrying its OWN per-instance dataframe,
    translations table, default target, and request cache — see the module
    docstring's point (1). `VSFRequestHandler` below reads all of this via
    `self.server.*` rather than module-level globals, so multiple `serve()`
    calls in one process (e.g. sequential notebook cells with different
    dataframes) never share state.
    """

    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, df: pd.DataFrame, translations: Optional[Translations]):
        super().__init__(server_address, RequestHandlerClass)
        self.df = df
        self.translations = translations
        self.default_target = _default_target(df)
        # Mirrors `server.py`'s `_cache_lock`/`_last_params`/`_last_res`/
        # `_top_columns_cache` globals, but scoped to this instance — see
        # the module docstring.
        self.cache_lock = threading.Lock()
        self.last_params: Optional[Dict[str, Any]] = None
        self.last_res = None
        self.top_columns_cache: Optional[Dict[str, Any]] = None


class VSFRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Request handler for `vsf.serve()`. Unlike `server.py`'s
    `VSFRequestHandler` (which extends `SimpleHTTPRequestHandler` for
    cwd-relative static file serving), this extends `BaseHTTPRequestHandler`
    directly and serves the packaged `vsf.webapp` assets itself via
    `_STATIC_ROUTES` / `_read_webapp_asset` — there is no cwd-relative
    directory to delegate to once `vsf` is installed as a real package.
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
        elif path == "/api/top_columns":
            self._handle_top_columns_api()
        elif path in _STATIC_ROUTES:
            self._serve_static(path)
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/analyze":
            self._handle_analyze_api()
        elif path == "/api/mine_center":
            self._handle_mine_center_api()
        elif path == "/api/graph_inference":
            self._handle_graph_inference_api()
        elif path == "/api/mine_graph_links":
            self._handle_mine_graph_links_api()
        else:
            self.send_error(404, "Endpoint not found")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    # -- response helpers -----------------------------------------------

    def _send_json_response(self, status_code: int, payload: Dict[str, Any]) -> None:
        """
        No `Access-Control-Allow-Origin` header — same-origin only. See
        `server.py`'s `_send_json_response` docstring for the reasoning
        (a wildcard CORS header would let ANY page open in the user's
        browser read this dataset and every mining/inference result over
        `http://127.0.0.1:<port>`, not just the served frontend).
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
        Mirrors `server.py`'s `_handle_columns_api`, except `default_target`
        is the SERVER INSTANCE's actual resolved default (`self.server.
        default_target`, computed once in `_VSFServer.__init__` from the
        caller's `df` — see `_default_target`), never a hardcoded "class".
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

    def _handle_top_columns_api(self) -> None:
        """Mirrors `server.py`'s `_handle_top_columns_api`, cached per-instance."""
        try:
            with self.server.cache_lock:
                cached = self.server.top_columns_cache
            if cached is not None:
                self._send_json_response(200, cached)
                return

            result = compute_top_insights(
                self.server.df,
                min_nmi=0.75,
                n_permutations=200,
                fdr_q=0.05,
                random_state=_RANDOM_STATE,
                translations=self.server.translations,
            )

            with self.server.cache_lock:
                self.server.top_columns_cache = result

            self._send_json_response(200, result)
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_analyze_api(self) -> None:
        """
        Mirrors `server.py`'s `_handle_analyze_api`. Two behavioral
        deviations, both deliberate fixes rather than parity bugs:

          - `target_col = req.get("target", self.server.default_target)`
            falls back to THIS dataset's actual default target, not the
            literal string "class".
          - An invalid/missing `target_col` (not composite, and not a
            column of `df`) raises a 400 instead of `server.py`'s silent
            `target_col = "class"` override, which would silently analyze
            the wrong column (or crash on datasets with no "class" column
            at all) instead of telling the caller their request was bad.
        """
        try:
            req = self._read_json_body()
            df = self.server.df
            translations = self.server.translations

            composite_target = req.get("composite_target", None)
            target_col = req.get("target", self.server.default_target)
            criterion = req.get("criterion", None)

            drop_cols = []
            if composite_target:
                mask = pd.Series([True] * len(df))
                display_parts = []
                for cond in composite_target:
                    col = cond.get("col")
                    val = str(cond.get("val"))
                    if col in df.columns:
                        mask = mask & (df[col].astype(str) == val)
                        human_col = _vis.humanize_col(col, translations)
                        human_val = _vis.humanize_val(col, val, translations)
                        display_parts.append(f"{human_col}={human_val}")
                        drop_cols.append(col)

                Z = mask.astype(int).values
                sort_Z = Z
                display_target_name = " AND ".join(display_parts) if display_parts else "Composite Filter"
                X_df = df.drop(columns=drop_cols)
            else:
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

            cache_key = {
                "composite_target": req.get("composite_target"),
                "target_col": target_col,
                "criterion": req.get("criterion"),
            }

            # Same rationale as `server.py`'s `_last_params`/`_last_res`
            # locking: holding `cache_lock` across the check-and-fit means
            # two concurrent requests for different targets never interleave
            # a read of one target's half-written cache with another's
            # write — the second request just waits instead of racing.
            with self.server.cache_lock:
                if self.server.last_params == cache_key and self.server.last_res is not None:
                    res, cached_X, cached_Z, cached_features, cached_target_name = self.server.last_res
                    X = cached_X
                    Z = cached_Z
                    feature_names = cached_features
                    display_target_name = cached_target_name
                else:
                    engine = AVREngine(
                        alpha=_ALPHA,
                        vir_threshold=_VIR_THRESHOLD,
                        max_d=_MAX_D,
                        n_permutations=_N_PERMUTATIONS,
                        random_state=_RANDOM_STATE,
                    )
                    res = engine.fit(X, Z, feature_names=feature_names)
                    self.server.last_params = cache_key
                    self.server.last_res = (res, X, Z, feature_names, display_target_name)

            payload = prepare_visualization_payload(
                res,
                X,
                Z,
                feature_names=feature_names,
                target_name=display_target_name,
                sort_Z=sort_Z,
                translations=translations,
            )
            self._send_json_response(200, payload)
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_mine_center_api(self) -> None:
        """
        Mirrors `server.py`'s `_handle_mine_center_api`, with the same
        default-target fallback fix as `_handle_analyze_api` (this
        dataset's actual default, not a hardcoded "class").
        """
        try:
            req = self._read_json_body()
            df = self.server.df
            translations = self.server.translations

            target_col = req.get("target", self.server.default_target)
            criterion = req.get("criterion", None)
            composite_target = req.get("composite_target", None)
            center_coords = req.get("center_coords", {})
            i_z_x_f = float(req.get("i_z_x_f", 1.0))

            drop_cols = []
            if composite_target:
                mask_z = np.ones(len(df), dtype=bool)
                for cond in composite_target:
                    col = cond.get("col")
                    val = str(cond.get("val"))
                    if col in df.columns:
                        mask_z &= df[col].astype(str) == val
                        drop_cols.append(col)
                Z = mask_z.astype(int).values
            else:
                if target_col not in df.columns:
                    self._send_json_response(
                        400,
                        {"error": f"target {target_col!r} is not a column of this dataset"},
                    )
                    return
                if criterion is not None:
                    Z = (df[target_col].astype(str) == str(criterion)).astype(int).values
                    drop_cols.append(target_col)
                else:
                    Z = df[target_col].values
                    if df[target_col].dtype.kind in ("U", "S", "O", "b"):
                        _, Z = np.unique(Z, return_inverse=True)
                    drop_cols.append(target_col)

            mask = np.ones(len(df), dtype=bool)
            for col, val in center_coords.items():
                if col in df.columns:
                    mask &= df[col].astype(str) == str(val)
                    if col not in drop_cols:
                        drop_cols.append(col)

            X_df = df.drop(columns=drop_cols)
            results = mine_dirty_center(X_df, Z, mask, i_z_x_f, translations=translations)
            self._send_json_response(200, {"results": results})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_graph_inference_api(self) -> None:
        """
        Mirrors `server.py`'s `_handle_graph_inference_api`, defaulting to
        this dataset's actual default target rather than a hardcoded
        "class".
        """
        try:
            req = self._read_json_body()
            inputs = req.get("inputs", {})
            target = req.get("target", self.server.default_target)
            target_criterion = req.get("target_criterion", None)
            nmi_threshold = float(req.get("nmi_threshold", 0.1))

            if target not in self.server.df.columns:
                self._send_json_response(
                    400, {"error": f"target {target!r} is not a column of this dataset"}
                )
                return

            from .graph_inference import run_graph_inference

            result = run_graph_inference(
                df=self.server.df,
                inputs=inputs,
                target=target,
                target_criterion=target_criterion,
                nmi_threshold=nmi_threshold,
                translations=self.server.translations,
            )
            self._send_json_response(200, result)
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_mine_graph_links_api(self) -> None:
        """
        Mirrors `server.py`'s `_handle_mine_graph_links_api`, defaulting to
        this dataset's actual default target rather than a hardcoded
        "class".
        """
        try:
            req = self._read_json_body()
            target = req.get("target", self.server.default_target)
            min_nmi = float(req.get("min_nmi", 0.20))
            max_depth = int(req.get("max_depth", 3))

            if target not in self.server.df.columns:
                self._send_json_response(
                    400, {"error": f"target {target!r} is not a column of this dataset"}
                )
                return

            from .graph_miner import mine_strong_links

            result = mine_strong_links(
                self.server.df,
                target=target,
                min_nmi=min_nmi,
                max_depth=max_depth,
                translations=self.server.translations,
            )
            self._send_json_response(200, result)
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})


def _build_server(
    df: pd.DataFrame,
    host: str = "127.0.0.1",
    port: int = 8000,
    translations: Optional[Translations] = None,
) -> _VSFServer:
    """
    Constructs (binds + listens, does NOT `serve_forever()`) the
    `_VSFServer` instance `serve()` runs. Split out as a private,
    independently-testable seam — mirrors `vsf/dashboard.py`'s
    `_build_scenario` pattern — so tests can construct a real server,
    drive it with real HTTP requests on an ephemeral port, and tear it
    down explicitly, without blocking on `serve_forever()`. See
    `tests/test_server.py`'s analogous pattern against the standalone
    `server.py` script.
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
    export with a fixed set of precomputed scenarios), `vsf.serve(df)` runs
    the LIVE application against `df` in-process: every analysis, dirty-
    center mining, and graph-inference request is computed on demand by
    this process, exactly as the repository's standalone `server.py` demo
    does for the bundled mushroom dataset — but for any dataframe, with no
    running-from-a-checkout requirement.

    The page's initial target column is `"class"` if `df` has a column by
    that name, else `df`'s first column (see `_default_target`); the
    catalog panel lets the user click any other column afterward, so this
    function takes no separate `target` argument.

    Binds to `host` (default `127.0.0.1`, loopback-only — matching
    `server.py`'s security posture; see that module's `main()` docstring
    for why binding `""`/`0.0.0.0` would expose this dataset and every
    mining/inference endpoint, none of which perform authentication, to
    the rest of the local network). Pass an explicit `host="0.0.0.0"` only
    if you specifically intend that exposure.

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
            `vsf` function; there is no dataset-specific default baked
            into this function (unlike the bundled `server.py` demo
            script, which always uses `MUSHROOM_TRANSLATIONS`).
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
