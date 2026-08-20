"""
VSF Local Web Server & API Handler
Hosts index.html and provides /api/analyze endpoints for interactive dataset visual rendering.
"""

import http.server
import json
import os
import socketserver
from typing import Any, Dict

import numpy as np
import pandas as pd

import vsf

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
        else:
            super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/api/analyze":
            self._handle_analyze_api()
        else:
            self.send_error(404, "Endpoint not found")

    def _send_json_response(self, status_code: int, payload: Dict[str, Any]) -> None:
        """Helper to send JSON responses."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _handle_columns_api(self) -> None:
        """API endpoint to get the list of columns available for analysis."""
        try:
            if not os.path.exists(DATASET_PATH):
                self.send_error(404, "Mushroom dataset not found")
                return
                
            df = pd.read_csv(DATASET_PATH)
            raw_cols = list(df.columns)
            ru_cols = []
            
            for c in raw_cols:
                ru_title = vsf.vis.MUSHROOM_TRANSLATIONS["columns"].get(c, c)
                display_label = f"{ru_title} ({c})" if ru_title != c else c
                
                unique_vals = df[c].dropna().unique().tolist()
                criteria = []
                for val in unique_vals:
                    val_str = str(val)
                    human_val = vsf.vis.humanize_val(c, val_str)
                    criteria.append({"id": val_str, "label": human_val})
                    
                ru_cols.append({"id": c, "label": display_label, "criteria": criteria})

            self._send_json_response(200, {
                "columns": ru_cols,
                "default_target": "class",
                "total_rows": len(df)
            })
            
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
                res, X, Z, feature_names=feature_names, target_name="class"
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
                        human_col = vsf.vis.humanize_col(col)
                        human_val = vsf.vis.humanize_val(col, val)
                        display_parts.append(f"{human_col}={human_val}")
                        drop_cols.append(col)
                
                Z = mask.astype(int).values
                display_target_name = " AND ".join(display_parts) if display_parts else "Сложный фильтр"
                X_df = df.drop(columns=drop_cols)
            else:
                if target_col not in df.columns:
                    target_col = "class"

                if criterion is not None:
                    Z = (df[target_col].astype(str) == str(criterion)).astype(int).values
                    human_criterion = vsf.vis.humanize_val(target_col, str(criterion))
                    human_col = vsf.vis.humanize_col(target_col)
                    display_target_name = f"{human_col} = {human_criterion}"
                else:
                    Z = df[target_col].values
                    display_target_name = target_col
                
                X_df = df.drop(columns=[target_col])

            feature_names = list(X_df.columns)
            X = X_df.values

            engine = vsf.AVREngine(
                alpha=0.01, vir_threshold=0.85, max_d=7, n_permutations=100, random_state=42
            )
            res = engine.fit(X, Z, feature_names=feature_names)
            payload = vsf.prepare_visualization_payload(
                res, X, Z, feature_names=feature_names, target_name=display_target_name
            )

            self._send_json_response(200, payload)
            
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})


def main() -> None:
    """Entry point for the VSF Local Web Server."""
    os.chdir(os.path.dirname(__file__))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer(("", PORT), VSFRequestHandler) as httpd:
        print("=" * 70)
        print(f" VSF Interactive Visual Dashboard Server Running!")
        print(f" Local URL: http://localhost:{PORT}")
        print("=" * 70)
        httpd.serve_forever()


if __name__ == "__main__":
    main()

