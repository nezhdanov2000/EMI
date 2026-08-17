"""
VSF Local Web Server & API Handler
Hosts index.html and provides /api/analyze endpoints for interactive dataset visual rendering.
"""

import http.server
import socketserver
import json
import os
import pandas as pd
import numpy as np
import vsf

PORT = 8000
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "mushrooms.csv")


class VSFRequestHandler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/api/mushroom":
            self._handle_mushroom_api()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/analyze":
            self._handle_analyze_api()
        else:
            self.send_error(404, "Endpoint not found")

    def _handle_mushroom_api(self):
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
                alpha=0.01, vir_threshold=0.85, max_d=7, n_permutations=200, random_state=42
            )
            res = engine.fit(X, Z, feature_names=feature_names)
            payload = vsf.prepare_visualization_payload(res, X, Z, feature_names=feature_names)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def _handle_analyze_api(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            dataset_type = req.get("dataset", "synthetic")
            if dataset_type == "synthetic":
                d_true = int(req.get("d_true", 3))
                X, Z, true_idx, names = vsf.generate_synthetic_dataset(
                    n_samples=800, d_true=d_true, n_noise_features=4, random_state=42
                )
                feature_names = names
            else:
                df = pd.read_csv(DATASET_PATH)
                Z = df["class"].values
                X_df = df.drop(columns=["class"])
                feature_names = list(X_df.columns)
                X = X_df.values

            engine = vsf.AVREngine(
                alpha=0.01, vir_threshold=0.85, max_d=7, n_permutations=200, random_state=42
            )
            res = engine.fit(X, Z, feature_names=feature_names)
            payload = vsf.prepare_visualization_payload(res, X, Z, feature_names=feature_names)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


def main():
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
