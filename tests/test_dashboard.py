"""
Regression tests for vsf.dashboard.export_full_dashboard, v2.0 "Clean Core".

Covers:
  1. End-to-end smoke: the export runs, returns a well-formed self-contained
     HTML document, and embeds exactly the two v2.0 data blocks
     (DASHBOARD_DATA, BRANCHES_DATA) -- never the removed v1.0 blocks
     (SCENARIOS_DATA, MINING_DATA, TOP_INSIGHTS_DATA, GRAPH_DATA).
  2. `_safe_json`'s "</script" escaping -- a string value containing a
     literal "</script>" must not be able to prematurely close the
     surrounding <script> tag.
  3. The `max_d` dimensionality cap: `export_full_dashboard` must reject
     `max_d` outside `[1, 4]` and accept every value inside it.
  4. `BRANCHES_DATA` is keyed by dimensionality (as a string), matches
     `branch_dims`/`default_branch` in `DASHBOARD_DATA`, and branches are
     independent (not required to be nested) -- exercised end-to-end
     through the real export.
  5. Removed v1.0 parameters (`targets`, `alpha`, `n_permutations`,
     `mine_center_*`, `top_insights_*`, `vir_threshold`, `random_state` as
     a targets-sweep concept) are genuinely gone from the signature, not
     just ignored.
  6. Dataset-agnosticism: the embedded DASHBOARD_DATA block carries no
     hardcoded UCI Mushroom vocabulary for an unrelated dataset.
  7. `dashboard.js` is syntactically valid (via `node --check`, skipped if
     node isn't on PATH) and defines the v2.0 branch-selector / 4D
     frame-controller functions (`selectBranch`, `stepSlice`,
     `toggleSliceAutoplay`, `setSliceAutoplaySpeed`) with a functional
     smoke test of their guard-clause/step/wrap/autoplay logic -- and does
     NOT define the removed v1.0 `selectScenario`.
  8. Packaging: templates are readable via `importlib.resources` the same
     way `vsf/dashboard.py`'s `_read_template` reads them at call time.
"""

import inspect
import json
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vsf
from vsf.avr import MAX_BRANCH_D
from vsf.dashboard import _safe_json

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "vsf" / "templates"


def _small_dataset(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.choice(["x", "y", "z"], size=n),
        "b": rng.choice(["lo", "hi"], size=n),
        "class": rng.choice(["p", "e"], size=n),
    })


def _extract_json_const(html, name, next_name):
    """
    Pulls one `const NAME = {...};` block out of the exported page's single
    combined <script> data block, given the literal name of the NEXT const
    that immediately follows it (see `_render_html`'s fixed const order:
    DASHBOARD_DATA, BRANCHES_DATA).
    """
    pattern = rf"const {re.escape(name)} = (\{{.*?\}});\nconst {re.escape(next_name)}"
    match = re.search(pattern, html, re.DOTALL)
    assert match is not None, f"could not find const {name} block before const {next_name}"
    return json.loads(match.group(1))


def _extract_last_json_const(html, name):
    """Like `_extract_json_const`, for the LAST const in the data block (no
    known following const name to anchor on -- anchor on `;\n</script>` instead)."""
    pattern = rf"const {re.escape(name)} = (\{{.*?\}});\n</script>"
    match = re.search(pattern, html, re.DOTALL)
    assert match is not None, f"could not find const {name} block before </script>"
    return json.loads(match.group(1))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_export_full_dashboard_smoke():
    df = _small_dataset()
    html = vsf.export_full_dashboard(df, target="class", criterion="p")

    assert isinstance(html, str)
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")

    assert "const DASHBOARD_DATA" in html
    assert "const BRANCHES_DATA" in html
    # v1.0 data blocks must be genuinely gone, not just unused.
    for banned in ("const SCENARIOS_DATA", "const MINING_DATA", "const TOP_INSIGHTS_DATA", "const GRAPH_DATA"):
        assert banned not in html

    # The three real <script> tags (Plotly CDN, embedded data, dashboard.js)
    # -- asserted so a future template edit that silently drops one is
    # caught here rather than only at runtime in a browser.
    assert html.count("<script") == 3


def test_export_full_dashboard_is_dataset_agnostic_by_default():
    # No `translations` passed -> every label must fall back to the raw
    # column/value strings actually present in `df`, never UCI Mushroom
    # vocabulary that isn't even in this dataframe.
    df = _small_dataset()
    html = vsf.export_full_dashboard(df, target="class", criterion="p")
    match = re.search(r"<script>\nconst DASHBOARD_DATA.*?const BRANCHES_DATA.*?</script>", html, re.DOTALL)
    assert match is not None
    data_block = match.group(0)
    for banned in ("odor", "cap-color", "gill-size", "spore-print", "mushroom", "poisonous", "edible"):
        assert banned not in data_block.lower()


def test_export_full_dashboard_rejects_non_dataframe():
    with pytest.raises(TypeError):
        vsf.export_full_dashboard([1, 2, 3], target="class")


def test_export_full_dashboard_rejects_unknown_target():
    df = _small_dataset()
    with pytest.raises(ValueError, match="not found in the dataframe"):
        vsf.export_full_dashboard(df, target="nonexistent_col")


# ---------------------------------------------------------------------------
# `_safe_json` "</script" escaping, exercised both directly and end-to-end
# ---------------------------------------------------------------------------

def test_safe_json_escapes_script_close_tag():
    assert _safe_json({"x": "</script><script>alert(1)</script>"}) == (
        '{"x": "<\\/script><script>alert(1)<\\/script>"}'
    )


def test_safe_json_allows_nan_as_bare_js_literal():
    # Deliberate: this output is inlined as a JS literal, never JSON.parse'd
    # -- allow_nan=True emitting bare NaN is correct here, not a bug.
    assert _safe_json({"x": float("nan")}) == '{"x": NaN}'


def test_export_escapes_malicious_value_in_translations():
    df = _small_dataset()
    payload = "</script><script>alert(1)</script>"
    translations = {
        "columns": {},
        "values": {"a": {"x": payload}},
    }
    html = vsf.export_full_dashboard(df, target="class", criterion="p", translations=translations)
    # The literal, unescaped "</script>" must appear exactly the 3 real
    # closing script tags' worth of times -- never as part of embedded data.
    assert html.count("</script>") == 3
    # The escaped form of the payload IS present (proves it made it into
    # the page at all, just safely).
    assert "<\\/script><script>alert(1)<\\/script>" in html


# ---------------------------------------------------------------------------
# max_d / dimensionality cap
# ---------------------------------------------------------------------------

def test_max_d_default_is_max_branch_d():
    sig = inspect.signature(vsf.export_full_dashboard)
    assert sig.parameters["max_d"].default == MAX_BRANCH_D
    assert MAX_BRANCH_D == 4


@pytest.mark.parametrize("bad_max_d", [0, -1, 5, 6, 7])
def test_max_d_out_of_range_raises(bad_max_d):
    df = _small_dataset()
    with pytest.raises(ValueError, match=r"max_d must be in \[1, 4\]"):
        vsf.export_full_dashboard(df, target="class", criterion="p", max_d=bad_max_d)


@pytest.mark.parametrize("ok_max_d", [1, 2, 3, 4])
def test_max_d_in_range_is_accepted(ok_max_d):
    df = _small_dataset()
    html = vsf.export_full_dashboard(df, target="class", criterion="p", max_d=ok_max_d)
    assert isinstance(html, str) and len(html) > 0
    branches_data = _extract_last_json_const(html, "BRANCHES_DATA")
    assert max(int(k) for k in branches_data.keys()) <= ok_max_d


def test_branches_data_never_exceeds_4d():
    df = _small_dataset()
    html = vsf.export_full_dashboard(df, target="class", criterion="p")
    branches_data = _extract_last_json_const(html, "BRANCHES_DATA")
    dims = [int(k) for k in branches_data.keys()]
    assert dims, "expected at least one branch for this test fixture"
    assert max(dims) <= 4, f"BRANCHES_DATA contains a dimensionality beyond the 4D cap: {dims}"


# ---------------------------------------------------------------------------
# BRANCHES_DATA shape: keyed by dimensionality, matches branch_dims/default_branch
# ---------------------------------------------------------------------------

def test_branches_data_keyed_by_dimensionality_and_matches_dashboard_data():
    df = _small_dataset()
    html = vsf.export_full_dashboard(df, target="class", criterion="p")
    dashboard_data = _extract_json_const(html, "DASHBOARD_DATA", "BRANCHES_DATA")
    branches_data = _extract_last_json_const(html, "BRANCHES_DATA")

    assert set(branches_data.keys()) == {str(d) for d in dashboard_data["branch_dims"]}
    assert dashboard_data["default_branch"] == str(max(dashboard_data["branch_dims"]))

    for d_str, payload in branches_data.items():
        assert payload["metrics"]["d"] == int(d_str)
        # v2.3 "Coverage Only" metrics shape -- no scenario/vir/history, no
        # MI/U_adj/p_value (see vsf.avr's module docstring). Coverage-search
        # statistics live in `search_centers`/`centers`, not here.
        assert set(payload["metrics"].keys()) == {"d"}


def test_branches_are_independent_not_required_to_be_nested():
    # Real end-to-end demonstration of Section 4.4: construct a dataset with
    # an XOR-synergistic pair that no single feature reveals, plus a
    # separately-informative single feature, and confirm the exported
    # dashboard's d=1 branch does not share features with its d=2 branch.
    rng = np.random.default_rng(123)
    n = 3000
    a_bit = rng.integers(0, 2, size=n)
    b_bit = rng.integers(0, 2, size=n)
    z_bit = np.bitwise_xor(a_bit, b_bit)
    flip = rng.random(n) < 0.3
    c_bit = np.where(flip, 1 - z_bit, z_bit)

    df = pd.DataFrame({
        "a": a_bit.astype(float) + rng.normal(0, 0.15, size=n),
        "b": b_bit.astype(float) + rng.normal(0, 0.15, size=n),
        "c": c_bit.astype(float) + rng.normal(0, 0.15, size=n),
        "class": z_bit,
    })

    html = vsf.export_full_dashboard(df, target="class", max_d=2)
    branches_data = _extract_last_json_const(html, "BRANCHES_DATA")

    # `selected_features` in the payload is already the human-readable list
    # (see vsf.vis.prepare_visualization_payload's return shape).
    d1_selected = set(branches_data["1"]["selected_features"])
    d2_selected = set(branches_data["2"]["selected_features"])
    assert d1_selected == {"c"}
    assert d2_selected == {"a", "b"}
    assert d1_selected.isdisjoint(d2_selected)


# ---------------------------------------------------------------------------
# Removed v1.0 parameters are genuinely gone from the signature
# ---------------------------------------------------------------------------

def test_removed_v1_parameters_are_rejected():
    df = _small_dataset()
    removed_kwargs = [
        {"targets": []},
        {"alpha": 0.01},
        {"n_permutations": 20},
        {"mine_center_n_permutations": 20},
        {"mine_center_purity_range": (0.0, 1.0)},
        {"max_dirty_cells": 1},
        {"top_insights_n_permutations": 20},
        {"vir_threshold": 0.5},
        {"random_state": 42},
    ]
    for kwargs in removed_kwargs:
        with pytest.raises(TypeError):
            vsf.export_full_dashboard(df, target="class", criterion="p", **kwargs)


def test_export_full_dashboard_signature_has_no_v1_parameters():
    sig = inspect.signature(vsf.export_full_dashboard)
    params = set(sig.parameters.keys())
    # v2.2 adds `center_spec`: a static export cannot be re-certified after
    # the fact, so the certificate it was built under is an explicit
    # parameter and is stated in the exported page's own legend.
    # 2026-09 adds `direction` (presence / absence search, `vsf.avr.Direction`).
    assert params == {
        "df", "target", "criterion", "translations", "title", "max_d",
        "center_spec", "direction",
    }


# ---------------------------------------------------------------------------
# dashboard.js: syntax + the v2.0 branch-selector / 4D frame-controller API
# ---------------------------------------------------------------------------

def test_dashboard_js_has_no_stale_v1_references():
    js_source = (TEMPLATES_DIR / "dashboard.js").read_text(encoding="utf-8")
    for banned in (
        "function selectScenario", "SCENARIOS_DATA[", "MINING_DATA[",
        "TOP_INSIGHTS_DATA.columns", "graphView", "graph_client",
    ):
        assert banned not in js_source, f"stale v1.0 reference left in dashboard.js: {banned!r}"
    assert "function selectBranch" in js_source
    assert "BRANCHES_DATA" in js_source


@pytest.mark.skipif(shutil.which("node") is None, reason="node.js not available")
def test_dashboard_js_is_syntactically_valid():
    proc = subprocess.run(
        ["node", "--check", str(TEMPLATES_DIR / "dashboard.js")], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"dashboard.js failed node --check: {proc.stderr}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node.js not available")
def test_select_branch_js_logic():
    # Functional smoke test of dashboard.js's selectBranch() in isolation:
    # stub the DOM and override updateDashboard() (the full Plotly render
    # pipeline is out of scope here); verify selectBranch's OWN guard
    # clauses and state reset/mutation.
    js_source = (TEMPLATES_DIR / "dashboard.js").read_text(encoding="utf-8")
    script = f"""
      global.document = {{
        getElementById: () => null,
        querySelectorAll: () => [],
      }};
      global.window = global;
      global.addEventListener = () => {{}};
      {js_source}

      BRANCHES_DATA = {{
        '1': {{ metrics: {{ d: 1, mi: 0.1, mi_null: 0.01, mi_adj: 0.09, u_adj: 0.1, p_value: 0.5, p_value_familywise: null }}, slice_axis: null, target_name: 'A' }},
        '2': {{ metrics: {{ d: 2, mi: 0.9, mi_null: 0.02, mi_adj: 0.88, u_adj: 0.9, p_value: 0.001, p_value_familywise: null }}, slice_axis: null, target_name: 'B' }},
      }};
      let updateDashboardCallCount = 0;
      updateDashboard = function(payload) {{ updateDashboardCallCount++; }};

      activeBranchD = '1';
      currentPayload = BRANCHES_DATA['1'];
      activeDimensionality = 1;
      activeSliceIndex = null;

      const observed = {{}};

      // Re-selecting the already-active branch is a no-op (no re-render).
      selectBranch(1);
      observed.noop_same_id_calls = updateDashboardCallCount;

      // Selecting a dimensionality with no discovered branch is a no-op too.
      selectBranch(99);
      observed.noop_missing_id_calls = updateDashboardCallCount;
      observed.unchanged_branch_id = activeBranchD;

      // Switching to a real, different branch: swaps currentPayload,
      // resets dimensionality to the NEW branch's own d (never null --
      // unlike v1.0 scenario switches, a branch has no XAI history to
      // re-derive from), and triggers exactly one re-render.
      selectBranch(2);
      observed.switched_branch_id = activeBranchD;
      observed.switched_payload_target = currentPayload.target_name;
      observed.reset_dimensionality = activeDimensionality;
      observed.render_calls_after_switch = updateDashboardCallCount;

      process.stdout.write(JSON.stringify(observed));
    """
    proc = subprocess.run(["node"], input=script, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    observed = json.loads(proc.stdout)
    assert observed == {
        "noop_same_id_calls": 0,
        "noop_missing_id_calls": 0,
        "unchanged_branch_id": "1",
        "switched_branch_id": "2",
        "switched_payload_target": "B",
        "reset_dimensionality": 2,
        "render_calls_after_switch": 1,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node.js not available")
def test_slice_frame_controller_step_and_autoplay_logic():
    # Functional smoke test of stepSlice()/toggleSliceAutoplay()/
    # setSliceAutoplaySpeed() without a full DOM: stub the handful of
    # document.* calls dashboard.js's frame controller touches, load the
    # real source, and drive the frame index purely through
    # currentPayload/activeSliceIndex/selectSlice's side effects.
    js_source = (TEMPLATES_DIR / "dashboard.js").read_text(encoding="utf-8")
    script = f"""
      global.document = {{
        getElementById: () => null,
        querySelectorAll: () => [],
      }};
      global.window = global;
      global.addEventListener = () => {{}};
      {js_source}

      // Isolate stepSlice()'s wrap-around math from the rest of the app
      // (renderPlot/transitionDimensionality/Plotly are not stubbed here).
      currentPayload = {{ slice_axis: {{ ticks: ['almond', 'anise', 'none'] }} }};
      selectSlice = function(idx) {{ activeSliceIndex = idx; }};

      const observed = [];
      activeSliceIndex = null;
      stepSlice(1); observed.push(activeSliceIndex);   // null -> 0
      stepSlice(1); observed.push(activeSliceIndex);   // 0 -> 1
      stepSlice(1); observed.push(activeSliceIndex);   // 1 -> 2
      stepSlice(1); observed.push(activeSliceIndex);   // 2 -> wraps to 0
      stepSlice(-1); observed.push(activeSliceIndex);  // 0 -> wraps to 2

      isAnimating = true;
      const beforeGuard = activeSliceIndex;
      stepSlice(1);
      observed.push(activeSliceIndex === beforeGuard); // no-op while animating
      isAnimating = false;

      setSliceAutoplaySpeed(50);
      observed.push(sliceAutoplaySpeedMs);

      process.stdout.write(JSON.stringify(observed));
    """
    proc = subprocess.run(["node"], input=script, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    observed = json.loads(proc.stdout)
    assert observed == [0, 1, 2, 0, 2, True, 50]


# ---------------------------------------------------------------------------
# Packaging: templates must be readable via importlib.resources the same
# way vsf/dashboard.py's `_read_template` reads them at call time.
# ---------------------------------------------------------------------------

def test_templates_readable_via_importlib_resources():
    for name in ("dashboard.css", "dashboard_shell.html", "dashboard.js"):
        text = resources.files("vsf.templates").joinpath(name).read_text(encoding="utf-8")
        assert len(text) > 0


def test_graph_client_js_no_longer_shipped():
    # Graph Inference / Knowledge-Base chain mining belongs to a separate
    # project and was never ported to this export (see vsf/dashboard.py's
    # module docstring) -- the template asset itself must be gone, not
    # merely unreferenced.
    assert not (TEMPLATES_DIR / "graph_client.js").exists()
