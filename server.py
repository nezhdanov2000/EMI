"""
VSF Local Web Server & API Handler
Hosts index.html and provides /api/analyze endpoints for interactive dataset visual rendering.
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.parse
from datetime import datetime
from typing import Any, Dict

# Global cache to speed up visual-only updates (e.g. blue_feature changes).
# `VSFRequestHandler` runs under `http.server.ThreadingHTTPServer`, i.e. one
# thread per connection: two concurrent requests can read/write these globals
# interleaved (e.g. one thread checking `_last_params == cache_key` while
# another is mid-assignment to `_last_res`), which can serve a request the
# stale/half-written cached tuple from a DIFFERENT target than the one that
# was just requested, or crash on a partially-updated tuple. `_cache_lock`
# serializes every read AND write of `_last_params` / `_last_res` /
# `_top_columns_cache` so a request always observes a consistent snapshot.
_cache_lock = threading.Lock()
_last_params = None
_last_res = None
_top_columns_cache = None

import numpy as np
import pandas as pd

import vsf

# `server.py` is a standalone script, not itself part of an installed
# package, and `main()` later does `os.chdir(os.path.dirname(__file__))` —
# but that chdir happens at server-start time, AFTER module import, so it
# cannot be relied on to make `examples/` importable. Explicitly put this
# script's own directory on `sys.path` (idempotent if already invoked via
# `python /path/to/server.py`, where Python already does this) so
# `examples.mushroom_demo` resolves regardless of the caller's cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from examples.mushroom_demo import MUSHROOM_TRANSLATIONS

PORT = 8050
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "mushrooms.csv")


class VSFRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    HTTP Request handler for serving static files and VSF API endpoints.
    """

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/api/mushroom":
            self._handle_mushroom_api()
        elif self.path == "/api/columns":
            self._handle_columns_api()
        elif self.path == "/api/top_columns":
            self._handle_top_columns_api()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/api/analyze":
            self._handle_analyze_api()
        elif self.path == "/api/mine_center":
            self._handle_mine_center_api()
        elif self.path == "/api/graph_inference":
            self._handle_graph_inference_api()
        elif self.path == "/api/mine_graph_links":
            self._handle_mine_graph_links_api()
        else:
            self.send_error(404, "Endpoint not found")

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def _send_json_response(self, status_code: int, payload: Dict[str, Any]) -> None:
        """
        Helper to send JSON responses.

        No `Access-Control-Allow-Origin` header is sent: this handler serves
        both the static frontend (`index.html`, `graph.html`, `static/*`)
        AND the `/api/*` endpoints from the same origin
        (`http://127.0.0.1:PORT`), so the frontend never needs cross-origin
        access. An earlier version sent `Access-Control-Allow-Origin: *`
        unconditionally, which does nothing for the legitimate same-origin
        frontend but does let ANY third-party site's JavaScript read these
        endpoints' responses (dataset contents, mining/inference results) for
        a browser that has this local server reachable — e.g. via
        `http://127.0.0.1:8050/...` fetches from an unrelated open tab. Add a
        specific, non-wildcard origin back here only if a genuine
        cross-origin frontend is introduced.
        """
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _handle_columns_api(self) -> None:
        """API endpoint to get the list of columns available for analysis."""
        try:
            if not os.path.exists(DATASET_PATH):
                self.send_error(404, "Mushroom dataset not found")
                return
                
            df = pd.read_csv(DATASET_PATH)
            ru_cols = vsf.catalog_from_dataframe(df, MUSHROOM_TRANSLATIONS)

            self._send_json_response(200, {
                "columns": ru_cols,
                "default_target": "class",
                "total_rows": len(df)
            })
            
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_top_columns_api(self) -> None:
        """
        API endpoint to get the list of columns & criteria whose best
        achievable association with a small forward-selected feature subset
        clears both an effect-size floor and a statistical-significance bar.

        Two-stage pipeline (Project_Master_Document.md Section 4.5.3), run
        because exhaustively significance-testing every single (column,
        value) criterion in the dataset against every candidate feature
        subset is too expensive to do directly:

        Stage 1 (fast screen, effect size only): for every (column, value)
            binary criterion Z that clears a minimum support count (the same
            support floor `vsf.mining._min_support_count` uses for
            dirty-center filter mining), run greedy forward NMI selection
            (up to 4 steps) over the OTHER features and keep the best
            Normalized Mutual Information seen and the feature subset S that
            achieved it. Criteria below `min_nmi` are dropped here. This is
            a point-estimate screen — it says nothing about whether the
            association could be a small-sample fluctuation.

        Stage 2 (validation, statistical significance): for the Stage-1
            survivors ONLY, run a marginal permutation test (H0: Z carries
            no information about the joint code of its selected subset S
            beyond chance) and apply Benjamini-Hochberg FDR correction
            JOINTLY across all survivors tested in this call. A criterion is
            only returned if it passes BOTH stages.

        An earlier version of this endpoint returned every criterion whose
        raw NMI cleared `min_nmi`, with NO minimum support requirement and
        NO significance test at all — a criterion with a handful of positive
        examples can trivially reach NMI close to 1.0 by chance with a
        large-enough candidate feature pool, and nothing distinguished that
        from a genuine association.
        """
        global _top_columns_cache
        try:
            if not os.path.exists(DATASET_PATH):
                self.send_error(404, "Mushroom dataset not found")
                return

            with _cache_lock:
                cached = _top_columns_cache
            if cached is not None:
                self._send_json_response(200, cached)
                return

            df = pd.read_csv(DATASET_PATH)
            result = vsf.compute_top_insights(
                df,
                min_nmi=0.75,
                n_permutations=200,
                fdr_q=0.05,
                random_state=42,
                translations=MUSHROOM_TRANSLATIONS,
            )

            with _cache_lock:
                _top_columns_cache = result

            self._send_json_response(200, result)

        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_mushroom_api(self) -> None:
        """API endpoint to run default VSF analysis on the mushroom dataset."""
        try:
            if not os.path.exists(DATASET_PATH):
                self.send_error(404, "Mushroom dataset not found")
                return

            df = pd.read_csv(DATASET_PATH)
            Z = df["class"].values
            X_df = df.drop(columns=["class"])
            feature_names = list(X_df.columns)
            X = X_df.values

            engine = vsf.AVREngine(
                alpha=0.01, vir_threshold=0.85, max_d=7, n_permutations=100, random_state=42
            )
            res = engine.fit(X, Z, feature_names=feature_names)
            payload = vsf.prepare_visualization_payload(
                res, X, Z, feature_names=feature_names, target_name="class", sort_Z=Z,
                translations=MUSHROOM_TRANSLATIONS,
            )

            self._send_json_response(200, payload)
            
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_analyze_api(self) -> None:
        """API endpoint to run VSF analysis with a dynamically selected target."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            composite_target = req.get("composite_target", None)
            target_col = req.get("target", "class")
            criterion = req.get("criterion", None)
            df = pd.read_csv(DATASET_PATH)



            drop_cols = []
            if composite_target:
                mask = pd.Series([True] * len(df))
                display_parts = []
                for cond in composite_target:
                    col = cond.get("col")
                    val = str(cond.get("val"))
                    if col in df.columns:
                        mask = mask & (df[col].astype(str) == val)
                        human_col = vsf.vis.humanize_col(col, MUSHROOM_TRANSLATIONS)
                        human_val = vsf.vis.humanize_val(col, val, MUSHROOM_TRANSLATIONS)
                        display_parts.append(f"{human_col}={human_val}")
                        drop_cols.append(col)
                
                Z = mask.astype(int).values
                sort_Z = Z
                display_target_name = " AND ".join(display_parts) if display_parts else "Composite Filter"
                X_df = df.drop(columns=drop_cols)
            else:
                if target_col not in df.columns:
                    target_col = "class"

                sort_Z = df[target_col].values

                if criterion is not None:
                    Z = (df[target_col].astype(str) == str(criterion)).astype(int).values
                    human_criterion = vsf.vis.humanize_val(target_col, str(criterion), MUSHROOM_TRANSLATIONS)
                    human_col = vsf.vis.humanize_col(target_col, MUSHROOM_TRANSLATIONS)
                    display_target_name = f"{human_col} = {human_criterion}"
                else:
                    Z = df[target_col].values
                    display_target_name = target_col
                
                X_df = df.drop(columns=[target_col])

            feature_names = list(X_df.columns)
            X = X_df.values

            global _last_params, _last_res
            cache_key = {
                "composite_target": req.get("composite_target"),
                "target_col": req.get("target", "class"),
                "criterion": req.get("criterion")
            }

            # `_last_params`/`_last_res` are shared across every connection
            # thread under ThreadingHTTPServer. Holding `_cache_lock` across
            # both the check and the (potentially several-second) fit+update
            # means two concurrent requests for different targets can no
            # longer interleave a read of one target's half-written cache
            # tuple with another's write; the cost is that a second request
            # simply waits for the first engine.fit() to finish rather than
            # running concurrently against stale/inconsistent globals.
            with _cache_lock:
                if _last_params == cache_key and _last_res is not None:
                    res, cached_X, cached_Z, cached_features, cached_target_name = _last_res
                    # Use cached items to prevent redundant delay
                    X = cached_X
                    Z = cached_Z
                    feature_names = cached_features
                    display_target_name = cached_target_name
                else:
                    engine = vsf.AVREngine(
                        alpha=0.01, vir_threshold=0.85, max_d=7, n_permutations=100, random_state=42
                    )
                    res = engine.fit(X, Z, feature_names=feature_names)

                    # Update cache
                    _last_params = cache_key
                    _last_res = (res, X, Z, feature_names, display_target_name)

            payload = vsf.prepare_visualization_payload(
                res, X, Z, feature_names=feature_names, target_name=display_target_name, sort_Z=sort_Z,
                translations=MUSHROOM_TRANSLATIONS,
            )

            self._send_json_response(200, payload)
            
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})


    def _handle_mine_center_api(self) -> None:
        """API endpoint to mine conjunctive filters for a dirty center with full target context."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            target_col = req.get("target", "class")
            criterion = req.get("criterion", None)
            composite_target = req.get("composite_target", None)
            center_coords = req.get("center_coords", {})  # e.g. {"cap-shape": "x", ...}
            i_z_x_f = float(req.get("i_z_x_f", 1.0))
            
            df = pd.read_csv(DATASET_PATH)
            
            # Setup Z target
            drop_cols = []
            if composite_target:
                mask_z = np.ones(len(df), dtype=bool)
                for cond in composite_target:
                    col = cond.get("col")
                    val = str(cond.get("val"))
                    if col in df.columns:
                        mask_z &= (df[col].astype(str) == val)
                        drop_cols.append(col)
                Z = mask_z.astype(int).values
            elif criterion is not None:
                Z = (df[target_col].astype(str) == str(criterion)).astype(int).values
                drop_cols.append(target_col)
            else:
                Z = df[target_col].values
                if df[target_col].dtype.kind in ('U', 'S', 'O', 'b'):
                    _, Z = np.unique(Z, return_inverse=True)
                drop_cols.append(target_col)
            
            # Setup mask for the specific cluster center
            mask = np.ones(len(df), dtype=bool)
            for col, val in center_coords.items():
                if col in df.columns:
                    mask &= (df[col].astype(str) == str(val))
                    if col not in drop_cols:
                        drop_cols.append(col)
                        
            X_df = df.drop(columns=drop_cols)
            
            # Use mining module
            results = vsf.mine_dirty_center(X_df, Z, mask, i_z_x_f, translations=MUSHROOM_TRANSLATIONS)
            
            self._send_json_response(200, {"results": results})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_graph_inference_api(self) -> None:
        """API endpoint for graph inference reasoning."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            inputs = req.get("inputs", {})
            target = req.get("target", "class")
            target_criterion = req.get("target_criterion", None)
            nmi_threshold = float(req.get("nmi_threshold", 0.1))
            
            df = pd.read_csv(DATASET_PATH)
            
            from vsf.graph_inference import run_graph_inference
            result = run_graph_inference(
                df=df,
                inputs=inputs,
                target=target,
                target_criterion=target_criterion,
                nmi_threshold=nmi_threshold,
                translations=MUSHROOM_TRANSLATIONS,
            )
            
            self._send_json_response(200, result)
            
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def _handle_mine_graph_links_api(self) -> None:
        """API endpoint to mine top global logical reasoning chains."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            target = req.get("target", "class")
            min_nmi = float(req.get("min_nmi", 0.20))
            max_depth = int(req.get("max_depth", 3))
            
            df = pd.read_csv(DATASET_PATH)
            
            from vsf.graph_miner import mine_strong_links
            result = mine_strong_links(
                df, target=target, min_nmi=min_nmi, max_depth=max_depth,
                translations=MUSHROOM_TRANSLATIONS,
            )
            
            self._send_json_response(200, result)
            
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

def main() -> None:
    """
    Entry point for the VSF Local Web Server.

    Binds to 127.0.0.1 only. `("", PORT)` (the previous binding) listens on
    ALL network interfaces, exposing this dashboard — including the raw
    mushroom dataset and every mining/inference endpoint, none of which
    perform authentication — to every other host on the local network (and
    to the internet if the machine has a public IP / port-forwarded router).
    This is a local research dashboard with no auth layer; it has no reason
    to accept connections from anywhere but the machine running it.
    """
    os.chdir(os.path.dirname(__file__))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), VSFRequestHandler) as httpd:
        print("=" * 70)
        print(f" VSF Interactive Visual Dashboard Server Running!")
        print(f" Local URL: http://localhost:{PORT}")
        print("=" * 70)
        httpd.serve_forever()


if __name__ == "__main__":
    main()

