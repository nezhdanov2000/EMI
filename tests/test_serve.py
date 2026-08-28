"""
Regression tests for vsf.server / vsf.serve(df).

This module deliberately does NOT reuse `tests/test_server.py`'s
`_ServerTestBase` (which spins up the standalone `server.py` script's
`VSFRequestHandler` against the bundled, hardcoded mushroom CSV): `vsf.serve`
is a fully independent implementation (see `vsf/server.py`'s module
docstring for why), and its whole point is to work on an ARBITRARY
dataframe — including ones with no "class" column at all, which
`server.py`'s handler could never be tested against without editing its
hardcoded `DATASET_PATH`.

Covers:
  1. `_default_target`: "class" when present, else the dataframe's first
     column — checked directly and via `/api/columns`' reported value.
  2. Per-instance state isolation: two independently-built `_VSFServer`s
     (different dataframes) never share `df`, cache locks, or cached
     results — the exact bug class module-level globals (as in `server.py`)
     would introduce for a reusable `serve()`.
  3. Thread-safety: concurrent `/api/analyze` requests for two different
     targets on the SAME server instance must not cross-contaminate (same
     regression shape as `tests/test_server.py`'s equivalent test, run here
     against the new implementation).
  4. No wildcard CORS header (same posture as `server.py`).
  5. `/api/analyze` (and `/api/mine_center`, `/api/graph_inference`,
     `/api/mine_graph_links`) default to the SERVER'S resolved
     `default_target` when no `target` is given in the request, never a
     hardcoded "class" — and reject an unknown target with 400 instead of
     `server.py`'s silent fallback-to-"class".
  6. Packaged static assets (`vsf/webapp/index.html`, `graph.html`,
     `static/css/*`, `static/js/*`) are served correctly via
     `importlib.resources`, proving the packaging works independent of
     process cwd (constructed with cwd deliberately left unchanged from
     the test runner's, unlike `server.py`'s `main()` which chdirs).
  7. Dataset-agnosticism smoke test: analysis, dirty-center mining, and
     graph inference all succeed end-to-end against a synthetic dataset
     whose target column is NOT named "class" and whose values are not
     mushroom vocabulary.
  8. `serve()`'s input validation (non-DataFrame / empty DataFrame) and its
     `open_browser` behavior (opens exactly once, with the right URL, only
     when requested) — exercised without blocking the test on
     `serve_forever()` forever, by making `webbrowser.open` raise a
     sentinel exception that unwinds through `serve()`'s try/finally
     (proving `httpd.server_close()` still runs) for the "opens" case, and
     by shutting the constructed server down from another thread for the
     "does not open" case.
"""

import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vsf
import vsf.server as vserve
from vsf.server import VSFRequestHandler, _build_server, _default_target


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _synthetic_df(seed: int = 0, n: int = 400) -> pd.DataFrame:
    """
    A dataset with NO "class" column and no mushroom vocabulary, so any
    test passing against it proves genuine dataset-agnosticism rather than
    an accidental match with the bundled demo dataset's shape.
    """
    rng = np.random.RandomState(seed)
    color = rng.choice(["red", "blue", "green"], n)
    # Make `outcome` deterministically dependent on `color` so NMI-based
    # code paths (top_columns, graph inference/mining) have real signal to
    # find rather than pure noise.
    outcome = np.where(color == "red", "yes", rng.choice(["yes", "no"], n))
    return pd.DataFrame(
        {
            "color": color,
            "size": rng.choice(["small", "medium", "large"], n),
            "shape": rng.choice(["circle", "square"], n),
            "outcome": outcome,
        }
    )


class _LiveServerTestBase(unittest.TestCase):
    """Spins up one `_VSFServer` (new implementation) per test class."""

    df_factory = staticmethod(_synthetic_df)

    @classmethod
    def setUpClass(cls):
        cls.df = cls.df_factory()
        cls.httpd = _build_server(cls.df, host="127.0.0.1", port=0, translations=None)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.thread.join(timeout=5)
        cls.httpd.server_close()

    def _get(self, path, timeout=30):
        return urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=timeout)

    def _post(self, path, payload, timeout=60):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            return e


# ---------------------------------------------------------------------------
# `_default_target`
# ---------------------------------------------------------------------------

class TestDefaultTarget(unittest.TestCase):
    def test_uses_class_when_present(self):
        df = pd.DataFrame({"x": [1, 2], "class": ["a", "b"], "y": [3, 4]})
        self.assertEqual(_default_target(df), "class")

    def test_falls_back_to_first_column_when_absent(self):
        df = _synthetic_df()
        self.assertNotIn("class", df.columns)
        self.assertEqual(_default_target(df), "color")


# ---------------------------------------------------------------------------
# Per-instance isolation (the core fix over server.py's module globals)
# ---------------------------------------------------------------------------

class TestPerInstanceIsolation(unittest.TestCase):
    def test_two_servers_do_not_share_df_or_cache_state(self):
        df_a = pd.DataFrame({"class": ["p", "e"] * 50, "x": list(range(100))})
        df_b = pd.DataFrame({"target": ["y", "n"] * 60, "z": list(range(120))})

        srv_a = _build_server(df_a, port=0)
        srv_b = _build_server(df_b, port=0)
        try:
            self.assertIsNot(srv_a.df, srv_b.df)
            self.assertIsNot(srv_a.cache_lock, srv_b.cache_lock)
            self.assertEqual(srv_a.default_target, "class")
            self.assertEqual(srv_b.default_target, "target")

            ta = threading.Thread(target=srv_a.serve_forever, daemon=True)
            tb = threading.Thread(target=srv_b.serve_forever, daemon=True)
            ta.start()
            tb.start()
            time.sleep(0.2)

            port_a = srv_a.server_address[1]
            port_b = srv_b.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port_a}/api/columns") as r:
                data_a = json.loads(r.read())
            with urllib.request.urlopen(f"http://127.0.0.1:{port_b}/api/columns") as r:
                data_b = json.loads(r.read())

            self.assertEqual(data_a["default_target"], "class")
            self.assertEqual(data_a["total_rows"], 100)
            self.assertEqual(data_b["default_target"], "target")
            self.assertEqual(data_b["total_rows"], 120)

            # Populating server A's analyze cache must never appear on B.
            req_a = urllib.request.Request(
                f"http://127.0.0.1:{port_a}/api/analyze",
                data=json.dumps({"target": "class"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_a, timeout=30) as r:
                json.loads(r.read())
            self.assertIsNotNone(srv_a.last_res)
            self.assertIsNone(srv_b.last_res)
        finally:
            srv_a.shutdown()
            srv_a.server_close()
            srv_b.shutdown()
            srv_b.server_close()


# ---------------------------------------------------------------------------
# Thread-safety on a single instance (mirrors tests/test_server.py)
# ---------------------------------------------------------------------------

class TestThreadSafetyAndCORS(_LiveServerTestBase):
    def test_cache_lock_exists_and_guards_shared_state(self):
        self.assertIsInstance(self.httpd.cache_lock, type(threading.Lock()))

    def test_no_wildcard_cors_header(self):
        resp = self._get("/api/columns")
        self.assertEqual(resp.status, 200)
        self.assertIsNone(resp.headers.get("Access-Control-Allow-Origin"))

    def test_concurrent_analyze_requests_for_different_targets_do_not_cross_contaminate(self):
        targets = ["color", "outcome"]
        results = [None, None]
        errors = [None, None]

        def call(idx):
            try:
                resp = self._post("/api/analyze", {"target": targets[idx]})
                results[idx] = json.loads(resp.read())
            except Exception as e:  # pragma: no cover - failure path
                errors[idx] = e

        threads = [threading.Thread(target=call, args=(i,)) for i in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=90)

        self.assertIsNone(errors[0], msg=errors[0])
        self.assertIsNone(errors[1], msg=errors[1])
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])
        self.assertNotEqual(results[0]["target_name"], results[1]["target_name"])


# ---------------------------------------------------------------------------
# Dataset-agnostic default-target fallback (the audited fix over server.py)
# ---------------------------------------------------------------------------

class TestDatasetAgnosticDefaultTargetFallback(_LiveServerTestBase):
    def test_columns_endpoint_reports_real_default_not_hardcoded_class(self):
        resp = self._get("/api/columns")
        data = json.loads(resp.read())
        self.assertNotIn("class", self.df.columns)
        self.assertEqual(data["default_target"], "color")

    def test_analyze_without_explicit_target_uses_server_default(self):
        resp = self._post("/api/analyze", {})
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read())
        # target_name falls back to the raw column name with no criterion.
        self.assertEqual(data["target_name"], "color")

    def test_analyze_rejects_unknown_target_instead_of_silently_falling_back(self):
        # server.py silently rewrites an unknown target to "class" (which
        # may not even exist in an arbitrary df); this implementation must
        # reject it instead of analyzing the wrong column.
        resp = self._post("/api/analyze", {"target": "definitely_not_a_column"})
        self.assertEqual(resp.status, 400)
        data = json.loads(resp.read())
        self.assertIn("error", data)

    def test_mine_center_rejects_unknown_target(self):
        resp = self._post(
            "/api/mine_center",
            {"target": "definitely_not_a_column", "center_coords": {}, "i_z_x_f": 1.0},
        )
        self.assertEqual(resp.status, 400)

    def test_graph_inference_and_mine_links_use_server_default_and_reject_unknown(self):
        resp = self._post("/api/graph_inference", {"inputs": {"color": "red"}, "nmi_threshold": 0.01})
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read())
        self.assertIn("nodes", data)

        resp = self._post("/api/graph_inference", {"inputs": {}, "target": "nope", "nmi_threshold": 0.01})
        self.assertEqual(resp.status, 400)

        resp = self._post("/api/mine_graph_links", {"min_nmi": 0.01})
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read())
        self.assertIn("target", data)
        self.assertEqual(data["target"], "color")

        resp = self._post("/api/mine_graph_links", {"target": "nope", "min_nmi": 0.01})
        self.assertEqual(resp.status, 400)


# ---------------------------------------------------------------------------
# Packaged static asset serving via importlib.resources
# ---------------------------------------------------------------------------

class TestPackagedStaticAssets(_LiveServerTestBase):
    def test_index_and_root_serve_the_same_packaged_html(self):
        with self._get("/") as r:
            root_body = r.read()
            self.assertEqual(r.headers.get("Content-Type"), "text/html; charset=utf-8")
        with self._get("/index.html") as r:
            index_body = r.read()
        self.assertEqual(root_body, index_body)
        self.assertIn(b"<html", root_body.lower())

    def test_graph_html_served(self):
        with self._get("/graph.html") as r:
            self.assertEqual(r.status, 200)
            body = r.read()
        self.assertIn(b"<html", body.lower())

    def test_css_and_js_assets_served_with_correct_content_type(self):
        with self._get("/static/css/styles.css") as r:
            self.assertEqual(r.status, 200)
            self.assertIn("text/css", r.headers.get("Content-Type", ""))
        with self._get("/static/css/graph_reasoning.css") as r:
            self.assertEqual(r.status, 200)
        with self._get("/static/js/app.js") as r:
            self.assertEqual(r.status, 200)
            self.assertIn("javascript", r.headers.get("Content-Type", ""))
            body = r.read()
        self.assertIn(b"function init", body)
        with self._get("/static/js/graph_reasoning.js") as r:
            self.assertEqual(r.status, 200)

    def test_unknown_path_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/static/js/does_not_exist.js")
        self.assertEqual(ctx.exception.code, 404)


# ---------------------------------------------------------------------------
# serve() itself: validation + open_browser behavior
# ---------------------------------------------------------------------------

class TestServeFunction(unittest.TestCase):
    def test_rejects_non_dataframe(self):
        with self.assertRaises(TypeError):
            vsf.serve([1, 2, 3], port=_free_port(), open_browser=False)

    def test_rejects_empty_dataframe(self):
        with self.assertRaises(ValueError):
            vsf.serve(pd.DataFrame(), port=_free_port(), open_browser=False)

    def test_open_browser_true_opens_exactly_once_with_correct_url(self):
        opened = []

        class _StopServe(Exception):
            pass

        def fake_open(url):
            opened.append(url)
            raise _StopServe()

        port = _free_port()
        orig_open = vserve.webbrowser.open
        vserve.webbrowser.open = fake_open
        try:
            df = _synthetic_df()
            with self.assertRaises(_StopServe):
                vsf.serve(df, host="127.0.0.1", port=port, open_browser=True)
        finally:
            vserve.webbrowser.open = orig_open

        self.assertEqual(opened, [f"http://127.0.0.1:{port}/"])

        # The `finally: httpd.server_close()` inside serve() must have run
        # even though webbrowser.open raised before serve_forever() — proven
        # by immediately being able to bind the same port again.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
        finally:
            probe.close()

    def test_open_browser_false_never_calls_webbrowser_open(self):
        calls = []
        orig_open = vserve.webbrowser.open
        vserve.webbrowser.open = lambda url: calls.append(url)

        created = {}
        orig_build = vserve._build_server

        def spy_build(*args, **kwargs):
            s = orig_build(*args, **kwargs)
            created["srv"] = s
            return s

        vserve._build_server = spy_build
        try:
            port = _free_port()
            df = _synthetic_df()
            t = threading.Thread(
                target=vsf.serve,
                kwargs=dict(df=df, host="127.0.0.1", port=port, open_browser=False),
                daemon=True,
            )
            t.start()
            for _ in range(100):
                if "srv" in created:
                    break
                time.sleep(0.05)
            self.assertIn("srv", created)
            time.sleep(0.1)
            created["srv"].shutdown()
            t.join(timeout=5)
        finally:
            vserve.webbrowser.open = orig_open
            vserve._build_server = orig_build

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
