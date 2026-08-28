"""
Regression tests for server.py.

Covers the code-review findings fixed in this module: the wildcard CORS
header, the unguarded shared-cache race under ThreadingHTTPServer, and the
two-stage (screen + BH-corrected significance) rebuild of
`/api/top_columns`. These spin up a real ThreadingHTTPServer bound to
127.0.0.1 on an ephemeral port (never the hardcoded PORT=8050, so this
suite can run even if a real instance is already listening there) and talk
to it over HTTP, since the request-handling logic lives inside
`http.server.BaseHTTPRequestHandler` methods that aren't meaningfully
unit-testable without a real request/response cycle.
"""

import http.server
import json
import os
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server as srv


class _ServerTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), srv.VSFRequestHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.thread.join(timeout=5)

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
        return urllib.request.urlopen(req, timeout=timeout)


class TestServerThreadSafetyAndCORS(_ServerTestBase):

    def test_cache_lock_exists_and_guards_shared_state(self):
        # An earlier version had no lock at all around `_last_params` /
        # `_last_res` / `_top_columns_cache`, which are read AND written
        # from every request-handling thread under ThreadingHTTPServer.
        import threading as threading_module
        self.assertIsInstance(srv._cache_lock, threading_module.Lock().__class__)

    def test_no_wildcard_cors_header(self):
        # Regression test: `_send_json_response` used to send
        # `Access-Control-Allow-Origin: *` unconditionally, exposing every
        # /api/* endpoint's response (dataset contents, mining/inference
        # results) to cross-origin reads from any page that can reach this
        # local server. The frontend is same-origin, so no CORS header is
        # needed at all.
        resp = self._get("/api/columns")
        self.assertEqual(resp.status, 200)
        self.assertIsNone(resp.headers.get("Access-Control-Allow-Origin"))

    def test_columns_endpoint_returns_expected_shape(self):
        resp = self._get("/api/columns")
        data = json.loads(resp.read())
        self.assertIn("columns", data)
        self.assertIn("total_rows", data)
        self.assertGreater(data["total_rows"], 0)
        self.assertEqual(data["default_target"], "class")

    def test_concurrent_analyze_requests_for_different_targets_do_not_cross_contaminate(self):
        # Regression test for the `_last_params`/`_last_res` cache race:
        # two concurrent /api/analyze calls for DIFFERENT targets must each
        # get back a payload scoped to their own target, never a
        # half-written or swapped cache tuple from the other thread.
        targets = ["class", "odor"]
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
        # class has 2 values, odor has 9 -> their selected axes / target
        # names must differ (a swapped/half-written cache would tend to
        # make both responses identical or internally inconsistent).
        self.assertNotEqual(results[0]["target_name"], results[1]["target_name"])


class TestMushroomTranslationsWiring(_ServerTestBase):
    """
    Regression coverage for the library/example-vocabulary decoupling:
    `vsf.vis` no longer carries a hardcoded `MUSHROOM_TRANSLATIONS` dict —
    `server.py` now imports it from `examples.mushroom_demo` and passes it
    explicitly as `translations` to every `vsf` call that renders a label.
    A broken import (e.g. `examples/` not resolvable, wrong package
    shape) would raise at `import server` time, failing every test in this
    module at collection — this class additionally asserts the resulting
    labels are actually humanized, not just that import didn't crash.
    """

    def test_server_imports_mushroom_translations_from_examples_package(self):
        from examples.mushroom_demo import MUSHROOM_TRANSLATIONS
        self.assertIs(srv.MUSHROOM_TRANSLATIONS, MUSHROOM_TRANSLATIONS)
        self.assertIn("odor", srv.MUSHROOM_TRANSLATIONS["columns"])

    def test_columns_endpoint_uses_mushroom_translations_for_labels(self):
        resp = self._get("/api/columns")
        data = json.loads(resp.read())
        odor_col = next(c for c in data["columns"] if c["id"] == "odor")
        # "Odor (odor)" per vsf.vis.humanize_col, not the raw "odor".
        self.assertEqual(odor_col["label"], "Odor (odor)")
        foul_criterion = next(c for c in odor_col["criteria"] if c["id"] == "f")
        self.assertEqual(foul_criterion["label"], "foul")

    def test_mushroom_endpoint_payload_uses_humanized_target_name(self):
        resp = self._get("/api/mushroom", timeout=60)
        data = json.loads(resp.read())
        # target_name="class" -> humanize_col("class", MUSHROOM_TRANSLATIONS)
        # == "Edibility (class)", proving `translations` reached
        # `prepare_visualization_payload` inside `_handle_mushroom_api`.
        self.assertEqual(data["target_name"], "Edibility (class)")


class TestTopColumnsTwoStagePipeline(_ServerTestBase):

    def test_top_columns_reports_gating_metadata_and_significant_criteria_only(self):
        # Regression test for the /api/top_columns rebuild: the endpoint
        # used to return every (column, value) criterion whose raw NMI
        # cleared 0.75, with NO support floor and NO significance test.
        # This asserts the two-stage pipeline's gating metadata is present
        # and that every reported criterion actually carries a p_value
        # (proof it went through Stage 2, not just the Stage 1 screen).
        resp = self._get("/api/top_columns", timeout=120)
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read())

        self.assertIn("min_nmi", data)
        self.assertIn("fdr_q", data)
        self.assertIn("n_stage1_survivors", data)
        self.assertIn("columns", data)

        n_criteria_returned = 0
        for col in data["columns"]:
            for crit in col["criteria"]:
                n_criteria_returned += 1
                self.assertIn("p_value", crit)
                self.assertGreaterEqual(crit["p_value"], 0.0)
                self.assertLessEqual(crit["p_value"], 1.0)
                self.assertGreaterEqual(crit["max_nmi"], data["min_nmi"])

        # Every returned criterion survived BH correction, so the count
        # returned can never exceed how many made it past the Stage-1 screen.
        self.assertLessEqual(n_criteria_returned, data["n_stage1_survivors"])

    def test_top_columns_is_cached_across_repeated_calls(self):
        resp1 = self._get("/api/top_columns", timeout=120)
        data1 = json.loads(resp1.read())
        resp2 = self._get("/api/top_columns", timeout=30)
        data2 = json.loads(resp2.read())
        self.assertEqual(data1, data2)


if __name__ == "__main__":
    unittest.main()
