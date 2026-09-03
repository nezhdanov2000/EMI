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
  5. `/api/analyze` defaults to the SERVER'S resolved `default_target` when
     no `target` is given in the request, never a hardcoded "class" — and
     rejects an unknown target with 400 instead of silently falling back.
  6. Packaged static assets (`vsf/webapp/index.html`, `static/css/*`,
     `static/js/*`) are served correctly via `importlib.resources`, proving
     the packaging works independent of process cwd (constructed with cwd
     deliberately left unchanged from the test runner's).
  7. Dataset-agnosticism smoke test: analysis succeeds end-to-end against a
     synthetic dataset whose target column is NOT named "class" and whose
     values are not mushroom vocabulary.
  8. `serve()`'s input validation (non-DataFrame / empty DataFrame) and its
     `open_browser` behavior (opens exactly once, with the right URL, only
     when requested) — exercised without blocking the test on
     `serve_forever()` forever, by making `webbrowser.open` raise a
     sentinel exception that unwinds through `serve()`'s try/finally
     (proving `httpd.server_close()` still runs) for the "opens" case, and
     by shutting the constructed server down from another thread for the
     "does not open" case.
  9. Global Pattern Scan (`/api/scan/start`/`/api/scan/status`/
     `/api/scan/cancel`, `TestGlobalPatternScan`): idle status before any
     scan; threshold/FDR validation; explicit rejection of the removed
     `nmi_threshold` parameter; a planted XOR ground truth (`target = a
     XOR b`, `c`/`d` pure noise) actually gets found and correctly
     filtered/sorted by `max_u_adj`; an all-noise dataset completes with an
     empty result set rather than erroring, which under v2.1 is enforced by
     Benjamini-Hochberg FDR control across the swept family and not merely by
     a high effect-size threshold; 409 on a concurrent start; cancel
     genuinely stops a scan before every pair is scanned (verified via a
     slowed-down `discover_branches`, not a size/speed-dependent race); a
     cancel with nothing running is a harmless no-op; scan state is
     per-`_VSFServer`-instance, not shared.

v2.0 "Clean Core" note (Project_Master_Document.md Section 0): `/api/analyze`
now runs Independent Branch Discovery and returns UP TO `MAX_BRANCH_D`
branches in one response (`{target, criterion, branches, branch_dims,
default_branch}`) instead of a single AVR-fit payload with a `target_name`
field. `/api/mine_center`, `/api/top_columns`, `/api/graph_inference`, and
`/api/mine_graph_links` — along with the static routes `/graph.html`,
`/static/css/graph_reasoning.css`, and `/static/js/graph_reasoning.js` — are
REMOVED (404) entirely, not merely deprecated; `_VSFServer.last_res` was
renamed to `.last_branches_payload`.
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
from unittest import mock

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
    # Make `outcome` deterministically dependent on `color` so the
    # information-theoretic code paths have real signal to find rather than
    # pure noise.
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
            self.assertIsNotNone(srv_a.last_branches_payload)
            self.assertIsNone(srv_b.last_branches_payload)
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
        self.assertNotEqual(results[0]["target"], results[1]["target"])
        self.assertEqual(results[0]["target"], "color")
        self.assertEqual(results[1]["target"], "outcome")
        # Each response's branches must actually be independent fits for
        # its own target, not a leaked/aliased copy of the other request's
        # in-flight cache entry.
        self.assertIn("branches", results[0])
        self.assertIn("branches", results[1])
        self.assertNotEqual(
            json.dumps(results[0]["branches"], sort_keys=True),
            json.dumps(results[1]["branches"], sort_keys=True),
        )


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
        # `target` falls back to the server's resolved default column, and
        # the v2.0 response shape carries up to MAX_BRANCH_D branches keyed
        # by dimensionality rather than a single AVR-fit payload.
        self.assertEqual(data["target"], "color")
        self.assertIsNone(data["criterion"])
        self.assertIn("branches", data)
        self.assertIn("branch_dims", data)
        self.assertIn("default_branch", data)
        self.assertTrue(set(data["branches"].keys()).issubset({"1", "2", "3"}))
        self.assertEqual(data["default_branch"], str(max(data["branch_dims"])))

    def test_analyze_rejects_unknown_target_instead_of_silently_falling_back(self):
        # server.py silently rewrites an unknown target to "class" (which
        # may not even exist in an arbitrary df); this implementation must
        # reject it instead of analyzing the wrong column.
        resp = self._post("/api/analyze", {"target": "definitely_not_a_column"})
        self.assertEqual(resp.status, 400)
        data = json.loads(resp.read())
        self.assertIn("error", data)


# ---------------------------------------------------------------------------
# Global Pattern Scan (/api/scan/start, /api/scan/status, /api/scan/cancel)
# — added after v2.0. Every test here builds its OWN `_VSFServer` (rather
# than sharing `_LiveServerTestBase`'s class-scoped instance) because scan
# state (`scan_job`) is mutable, cross-request server state, and several
# tests here depend on catching it mid-run — sharing a server across test
# methods would make those tests order-dependent. This mirrors
# `TestPerInstanceIsolation`'s own pattern above.
# ---------------------------------------------------------------------------

class TestGlobalPatternScan(unittest.TestCase):
    @staticmethod
    def _xor_df(seed: int = 0, n: int = 300) -> pd.DataFrame:
        """
        `target = a XOR b` (both binary), `c`/`d` pure noise — so a scan has
        an unambiguous ground truth: every (a, value)/(b, value)/(target,
        value) pair should score coverage == 1.0 (found via the branch that
        pairs the other two of {a, b, target}: two cells, each 100% pure and
        large enough to certify), while every (c, *)/(d, *) pair should score
        far below any reasonable threshold AND fail FDR control.
        """
        rng = np.random.RandomState(seed)
        a = rng.randint(0, 2, n)
        b = rng.randint(0, 2, n)
        return pd.DataFrame({
            "a": a,
            "b": b,
            "c": rng.randint(0, 3, n),
            "d": rng.randint(0, 2, n),
            "target": a ^ b,
        })

    @staticmethod
    def _start_server(df):
        httpd = _build_server(df, host="127.0.0.1", port=0, translations=None)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd, thread

    @staticmethod
    def _stop_server(httpd, thread):
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    @staticmethod
    def _get(port, path, timeout=30):
        return urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout)

    @staticmethod
    def _post(port, path, payload, timeout=30):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            return e

    @classmethod
    def _poll_until_terminal(cls, port, timeout=30):
        deadline = time.time() + timeout
        status = None
        while time.time() < deadline:
            with cls._get(port, "/api/scan/status") as r:
                status = json.loads(r.read())
            if status["status"] in ("done", "cancelled", "error"):
                return status
            time.sleep(0.05)
        raise AssertionError(f"scan did not reach a terminal state within {timeout}s (last: {status})")

    def test_idle_before_any_scan_ever_ran(self):
        httpd, thread = self._start_server(self._xor_df())
        try:
            port = httpd.server_address[1]
            with self._get(port, "/api/scan/status") as r:
                data = json.loads(r.read())
            self.assertEqual(data, {"status": "idle"})
        finally:
            self._stop_server(httpd, thread)

    def test_start_rejects_out_of_range_or_non_numeric_threshold(self):
        httpd, thread = self._start_server(self._xor_df())
        try:
            port = httpd.server_address[1]
            # 100 is deliberately NOT in this list: the comparison against
            # coverage_threshold is >= (see
            # test_scan_coverage_threshold_is_inclusive_at_100), so a pair
            # certifying exactly 100% coverage must be reachable by typing
            # 100 into the threshold field, not silently unrepresentable.
            for bad in (150, -1, "not-a-number"):
                resp = self._post(port, "/api/scan/start", {"coverage_threshold": bad})
                self.assertEqual(resp.code, 400, msg=f"threshold={bad!r} should be rejected")
                data = json.loads(resp.read())
                self.assertIn("error", data)
            for bad_q in (0, -0.1, 1.5, "nope"):
                resp = self._post(port, "/api/scan/start",
                                  {"coverage_threshold": 50, "fdr_q": bad_q})
                self.assertEqual(resp.code, 400, msg=f"fdr_q={bad_q!r} should be rejected")
            # A scan with no permutation null at all has nothing for BH to
            # control across, so it must be refused rather than silently
            # degrading to an uncorrected effect-size filter.
            resp = self._post(port, "/api/scan/start", {
                "coverage_threshold": 50, "n_permutations": 0,
                "n_permutations_familywise": 0,
            })
            self.assertEqual(resp.code, 400)
            # A fresh, valid start must still work afterward — the rejected
            # attempts above must not have left scan_job in a bad state.
            resp = self._post(port, "/api/scan/start", {"coverage_threshold": 50})
            self.assertEqual(resp.status, 200)
            self._poll_until_terminal(port, timeout=60)
        finally:
            self._stop_server(httpd, thread)

    def test_scan_coverage_threshold_is_inclusive_at_100(self):
        # _xor_df's docstring guarantees every (a, value)/(b, value)/
        # (target, value) pair certifies coverage == 1.0 exactly (two
        # cells, each 100% pure, large enough to certify). A threshold of
        # 100 must still keep these — coverage >= threshold_pct / 100, not
        # coverage > threshold_pct / 100 — or "100%" would be an
        # unsatisfiable filter despite being a legal input value.
        httpd, thread = self._start_server(self._xor_df())
        try:
            port = httpd.server_address[1]
            resp = self._post(port, "/api/scan/start",
                              {"coverage_threshold": 100, "n_permutations": 99})
            self.assertEqual(resp.status, 200)

            status = self._poll_until_terminal(port, timeout=60)
            self.assertEqual(status["status"], "done")

            results = status["results"]
            self.assertGreater(len(results), 0)
            found_columns = {r["column"] for r in results}
            self.assertEqual(found_columns, {"a", "b", "target"})
            for r in results:
                self.assertEqual(r["coverage"], 1.0)
        finally:
            self._stop_server(httpd, thread)

    def test_start_rejects_the_removed_nmi_threshold_parameter(self):
        # `nmi_threshold` is not merely renamed: NMI_min saturates toward
        # 100% on rare One-vs-Rest criteria with no signal, which is exactly
        # what a dataset-wide scan manufactures in bulk. Silently accepting
        # the old parameter would let a caller keep filtering on the artifact.
        httpd, thread = self._start_server(self._xor_df())
        try:
            port = httpd.server_address[1]
            resp = self._post(port, "/api/scan/start", {"nmi_threshold": 80})
            self.assertEqual(resp.code, 400)
            data = json.loads(resp.read())
            self.assertIn("coverage_threshold", data["error"])
        finally:
            self._stop_server(httpd, thread)

    def test_start_rejects_the_removed_u_adj_threshold_parameter(self):
        # v2.2: u_adj answers "does an association exist", the scan answers
        # "does a certifiable centre exist", and the two select different
        # pairs. Accepting the old parameter would silently keep filtering on
        # the wrong quantity.
        httpd, thread = self._start_server(self._xor_df())
        try:
            port = httpd.server_address[1]
            resp = self._post(port, "/api/scan/start", {"u_adj_threshold": 50})
            self.assertEqual(resp.code, 400)
            data = json.loads(resp.read())
            self.assertIn("coverage_threshold", data["error"])
        finally:
            self._stop_server(httpd, thread)

    def test_scan_finds_the_planted_relationship_and_filters_by_threshold(self):
        httpd, thread = self._start_server(self._xor_df())
        try:
            port = httpd.server_address[1]
            resp = self._post(port, "/api/scan/start",
                              {"coverage_threshold": 80, "n_permutations": 99})
            self.assertEqual(resp.status, 200)

            status = self._poll_until_terminal(port, timeout=60)
            self.assertEqual(status["status"], "done")
            self.assertEqual(status["progress"]["current"], status["progress"]["total"])

            results = status["results"]
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)

            # Every surviving pair must genuinely exceed the threshold —
            # the endpoint's own filtering, not just this test's assertion.
            for r in results:
                self.assertGreater(r["coverage"], 0.80)
                self.assertGreater(r["n_centers"], 0)
                self.assertTrue(r["fdr_significant"])
                self.assertLessEqual(r["p_value"], 0.05)
                # The raw MI must clear its own noise floor, not merely be
                # large: `mi > mi_null` is the property v2.0 could not state.
                self.assertGreater(r["best_mi"], r["best_mi_null"])
                self.assertIn(r["column"], {"a", "b", "target"})  # c/d are pure noise, must not survive

            # Results are sorted by coverage descending.
            scores = [r["coverage"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

            # a=0, a=1, b=0, b=1, target=0, target=1 should all be found
            # (the XOR relationship is symmetric in all three columns).
            found_columns = {r["column"] for r in results}
            self.assertEqual(found_columns, {"a", "b", "target"})
        finally:
            self._stop_server(httpd, thread)

    def test_all_noise_dataset_yields_no_results_even_at_a_permissive_threshold(self):
        # An all-noise dataset must survive NOTHING, and — the point of the
        # v2.1 change — it must survive nothing even when the effect-size
        # threshold is set low enough that v2.0's NMI_min filter would have
        # passed rare One-vs-Rest criteria on this exact data. The FDR gate
        # is what enforces that, so this asserts on `n_fdr_significant`
        # directly rather than only on the filtered output.
        rng = np.random.RandomState(1)
        n = 150
        noise_df = pd.DataFrame({f"col{i}": rng.randint(0, 3, n) for i in range(5)})
        httpd, thread = self._start_server(noise_df)
        try:
            port = httpd.server_address[1]
            resp = self._post(port, "/api/scan/start",
                              {"coverage_threshold": 5, "n_permutations": 99})
            self.assertEqual(resp.status, 200)
            status = self._poll_until_terminal(port, timeout=60)
            self.assertEqual(status["status"], "done")
            self.assertEqual(status["results"], [])
            self.assertGreater(status["n_tested"], 0)
            self.assertEqual(status["n_fdr_significant"], 0)
        finally:
            self._stop_server(httpd, thread)

    def test_start_while_running_is_rejected_with_409(self):
        httpd, thread = self._start_server(self._xor_df())
        try:
            port = httpd.server_address[1]
            # Slow discover_branches down so the first scan is still
            # "running" when the second /api/scan/start arrives — avoids a
            # dataset-size/CPU-speed-dependent race.
            real_discover_branches = vserve.discover_branches

            def _slow_discover_branches(*args, **kwargs):
                time.sleep(0.5)
                return real_discover_branches(*args, **kwargs)

            with mock.patch.object(vserve, "discover_branches", _slow_discover_branches):
                resp1 = self._post(port, "/api/scan/start", {"coverage_threshold": 50})
                self.assertEqual(resp1.status, 200)
                time.sleep(0.05)  # let the background thread actually start

                resp2 = self._post(port, "/api/scan/start", {"coverage_threshold": 50})
                self.assertEqual(resp2.code, 409)
                data = json.loads(resp2.read())
                self.assertIn("error", data)

                self._poll_until_terminal(port, timeout=30)
        finally:
            self._stop_server(httpd, thread)

    def test_cancel_stops_a_running_scan_before_it_finishes_all_pairs(self):
        httpd, thread = self._start_server(self._xor_df())
        try:
            port = httpd.server_address[1]
            real_discover_branches = vserve.discover_branches

            def _slow_discover_branches(*args, **kwargs):
                time.sleep(0.3)
                return real_discover_branches(*args, **kwargs)

            with mock.patch.object(vserve, "discover_branches", _slow_discover_branches):
                resp = self._post(port, "/api/scan/start", {"coverage_threshold": 50})
                self.assertEqual(resp.status, 200)
                time.sleep(0.15)  # ensure it's mid-flight, not finished or unstarted

                cancel_resp = self._post(port, "/api/scan/cancel", {})
                self.assertEqual(cancel_resp.status, 200)
                cancel_data = json.loads(cancel_resp.read())
                self.assertEqual(cancel_data["status"], "cancelling")

                status = self._poll_until_terminal(port, timeout=30)
                self.assertEqual(status["status"], "cancelled")
                self.assertLess(status["progress"]["current"], status["progress"]["total"])
        finally:
            self._stop_server(httpd, thread)

    def test_cancel_when_nothing_is_running_is_a_harmless_no_op(self):
        httpd, thread = self._start_server(self._xor_df())
        try:
            port = httpd.server_address[1]
            resp = self._post(port, "/api/scan/cancel", {})
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read())
            self.assertEqual(data["status"], "not_running")
        finally:
            self._stop_server(httpd, thread)

    def test_scan_state_is_per_instance_not_shared(self):
        httpd_a, thread_a = self._start_server(self._xor_df(seed=0))
        httpd_b, thread_b = self._start_server(self._xor_df(seed=1))
        try:
            port_a = httpd_a.server_address[1]
            port_b = httpd_b.server_address[1]

            resp = self._post(port_a, "/api/scan/start", {"coverage_threshold": 50})
            self.assertEqual(resp.status, 200)
            self._poll_until_terminal(port_a, timeout=30)

            with self._get(port_a, "/api/scan/status") as r:
                status_a = json.loads(r.read())
            with self._get(port_b, "/api/scan/status") as r:
                status_b = json.loads(r.read())

            self.assertIn(status_a["status"], ("done", "cancelled"))
            self.assertEqual(status_b, {"status": "idle"})
            self.assertIsNot(httpd_a.scan_lock, httpd_b.scan_lock)
        finally:
            self._stop_server(httpd_a, thread_a)
            self._stop_server(httpd_b, thread_b)


# ---------------------------------------------------------------------------
# Removed endpoints (v1.0 Auto-Discovery / dirty-center mining / Graph
# Inference — Project_Master_Document.md Section 0) must now 404, not just
# behave differently.
# ---------------------------------------------------------------------------

class TestRemovedEndpoints(_LiveServerTestBase):
    def test_mine_center_is_404(self):
        resp = self._post(
            "/api/mine_center",
            {"target": "color", "center_coords": {}, "i_z_x_f": 1.0},
        )
        self.assertEqual(resp.code, 404)

    def test_top_columns_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/api/top_columns")
        self.assertEqual(ctx.exception.code, 404)

    def test_graph_inference_is_404(self):
        resp = self._post("/api/graph_inference", {"inputs": {"color": "red"}, "nmi_threshold": 0.01})
        self.assertEqual(resp.code, 404)

    def test_mine_graph_links_is_404(self):
        resp = self._post("/api/mine_graph_links", {"min_nmi": 0.01})
        self.assertEqual(resp.code, 404)


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

    def test_graph_html_route_is_404(self):
        # graph.html and its dedicated static assets are gone entirely
        # (Project_Master_Document.md Section 0), not just unlinked.
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/graph.html")
        self.assertEqual(ctx.exception.code, 404)

    def test_css_and_js_assets_served_with_correct_content_type(self):
        with self._get("/static/css/styles.css") as r:
            self.assertEqual(r.status, 200)
            self.assertIn("text/css", r.headers.get("Content-Type", ""))
        with self._get("/static/js/app.js") as r:
            self.assertEqual(r.status, 200)
            self.assertIn("javascript", r.headers.get("Content-Type", ""))
            body = r.read()
        self.assertIn(b"function init", body)

    def test_removed_graph_static_assets_are_404(self):
        for path in ("/static/css/graph_reasoning.css", "/static/js/graph_reasoning.js"):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self._get(path)
            self.assertEqual(ctx.exception.code, 404, msg=path)

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
