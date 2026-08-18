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
            en_cols = []
            
            for c in raw_cols:
                en_title = vsf.vis.MUSHROOM_TRANSLATIONS["columns"].get(c, c)
                display_label = f"{en_title} ({c})" if en_title != c else c
                en_cols.append({"id": c, "label": display_label})

            self._send_json_response(200, {
                "columns": en_cols,
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

            target_col = req.get("target", "class")
            df = pd.read_csv(DATASET_PATH)

            if target_col not in df.columns:
                target_col = "class"

            Z = df[target_col].values
            X_df = df.drop(columns=[target_col])
            feature_names = list(X_df.columns)
            X = X_df.values

            engine = vsf.AVREngine(
                alpha=0.01, vir_threshold=0.85, max_d=7, n_permutations=100, random_state=42
            )
            res = engine.fit(X, Z, feature_names=feature_names)
            payload = vsf.prepare_visualization_payload(
                res, X, Z, feature_names=feature_names, target_name=target_col
            )

            self._send_json_response(200, payload)
            
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})


def main() -> None:
    """Entry point for the VSF Local Web Server."""
    os.chdir(os.path.dirname(__file__))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), VSFRequestHandler) as httpd:
        print("=" * 70)
        print(f" VSF Interactive Visual Dashboard Server Running!")
        print(f" Local URL: http://localhost:{PORT}")
        print("=" * 70)
        httpd.serve_forever()


if __name__ == "__main__":
    main()

