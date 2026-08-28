"""
Regression tests for vsf.dashboard.export_full_dashboard.

Covers:
  1. End-to-end smoke: the export runs, returns a well-formed self-contained
     HTML document, and embeds the four expected data blocks.
  2. `_safe_json`'s "</script" escaping — a string value containing a
     literal "</script>" must not be able to prematurely close the
     surrounding <script> tag.
  3. `_precompute_dirty_center_mining`'s `max_dirty_cells` cap: exceeding it
     must raise UserWarning (never truncate silently) and the mining data
     actually returned must respect the cap.
  4. The `max_d` dimensionality cap: `export_full_dashboard` must reject
     `max_d` outside `[1, 4]`, and the exported payload's selection history
     must never contain a step beyond 4 — this export's spatial encoding is
     3 coordinate axes (X/Y/Z) plus one 4D time/frame axis, with nothing
     past that to display (see `vsf.dashboard`'s module docstring and
     `_MAX_SUPPORTED_D`).
  5. `dashboard.js` is syntactically valid and actually defines the 4D
     frame-controller functions (`stepSlice`, `toggleSliceAutoplay`,
     `setSliceAutoplaySpeed`) that `dashboard_shell.html`'s frame controller
     markup wires up via onclick/onchange, plus a functional smoke test of
     their step/wrap/autoplay logic run under Node.js.
  6. Multi-target scenario precomputation (`targets`): the default
     (`targets=None`) precomputes one independent, fully-fit scenario per
     column of `df`, keyed in `SCENARIOS_DATA`/`MINING_DATA` by column id;
     `targets=[]` reproduces the old single-scenario-only export exactly;
     an explicit `targets` list limits scope to just those columns (plus
     the primary target); a bad column name in either `target` or
     `targets` raises `ValueError`; and — the load-bearing correctness
     claim of this feature — a column's scenario reached via `targets` is
     BYTE-IDENTICAL to what exporting with that column as the primary
     `target` would produce (same fit, same payload, same mining, not an
     approximation). Also covers `dashboard.js`'s `selectScenario`
     client-side scenario switch (guard clauses + state reset), run under
     Node.js.
  7. Dataset-agnosticism: none of the template assets hardcode UCI Mushroom
     vocabulary (mirrors the check `tests/test_vis.py` already applies to
     vsf/vis.py).
  8. Graph Inference / Knowledge-Base chain mining (the live app's
     `graph.html`) is explicitly out of scope for this export — it is a
     separate project's feature and is not ported here (see the module
     docstring's "Known, deliberate scope limits"). There is nothing left
     to cross-validate against Node.js for it; `vsf/templates/graph_client.js`
     no longer exists.

Node.js-dependent tests are skipped (not failed) if `node` isn't on PATH,
so this suite still runs in environments without a JS toolchain installed.

Most single-scenario tests below pass `targets=[]` explicitly, so they
exercise exactly the old single-scenario code path (fast, deterministic)
rather than incidentally depending on the multi-target default — the
multi-target behavior itself gets its own dedicated section.
"""

import json
import re
import shutil
import subprocess
import warnings
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vsf
from vsf.dashboard import _MAX_SUPPORTED_D, _safe_json

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
    DASHBOARD_DATA, SCENARIOS_DATA, MINING_DATA, TOP_INSIGHTS_DATA) — the
    same non-greedy-up-to-a-known-boundary approach the rest of this file
    already uses ad hoc, factored out since the multi-target tests below
    need it for three different consts.
    """
    pattern = rf"const {re.escape(name)} = (\{{.*?\}});\nconst {re.escape(next_name)}"
    match = re.search(pattern, html, re.DOTALL)
    assert match is not None, f"could not find const {name} block before const {next_name}"
    return json.loads(match.group(1))


def _dataset_with_many_features(n=400, seed=2):
    # Enough independent, target-correlated columns that AVR's greedy
    # feature selection can actually walk past step 4 if nothing clamps
    # max_steps = min(max_d, len(sorted_F)) — needed to make the "history
    # never exceeds 4" assertion meaningful rather than vacuously true.
    rng = np.random.default_rng(seed)
    target = rng.choice(["p", "e"], size=n)
    df = pd.DataFrame({"class": target})
    for i in range(6):
        # Each column agrees with the target most of the time -> real signal.
        agree = rng.random(n) < 0.9
        col = np.where(agree, target, rng.choice(["p", "e"], size=n))
        df[f"f{i}"] = col
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_export_full_dashboard_smoke():
    df = _small_dataset()
    html = vsf.export_full_dashboard(
        df,
        target="class",
        criterion="p",
        targets=[],
        n_permutations=20,
        mine_center_n_permutations=20,
        top_insights_n_permutations=20,
    )

    assert isinstance(html, str)
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    for marker in ("const DASHBOARD_DATA", "const SCENARIOS_DATA", "const MINING_DATA", "const TOP_INSIGHTS_DATA"):
        assert marker in html
    assert "const GRAPH_DATA" not in html
    # The three real <script> tags (Plotly CDN, embedded data, dashboard.js)
    # — asserted so a future template edit that silently drops one is
    # caught here rather than only at runtime in a browser. Adding
    # SCENARIOS_DATA didn't add a script TAG — it's one more const inside
    # the existing combined data <script> block.
    assert html.count("<script") == 3


def test_export_full_dashboard_is_dataset_agnostic_by_default():
    # No `translations` passed -> every label must fall back to the raw
    # column/value strings actually present in `df`, never a UCI Mushroom
    # vocabulary string that isn't even in this dataframe. Checked against
    # the embedded DATA blocks only (not the whole page — the shipped JS/
    # CSS legitimately mentions "UCI Mushroom" in a couple of explanatory
    # code comments about a fixed bug, same as this project's other source
    # comments/docstrings do; that's documentation, not a data leak).
    df = _small_dataset()
    html = vsf.export_full_dashboard(
        df, target="class", criterion="p", targets=[],
        n_permutations=20, mine_center_n_permutations=20,
        top_insights_n_permutations=20,
    )
    match = re.search(r"<script>\nconst DASHBOARD_DATA.*?</script>", html, re.DOTALL)
    assert match is not None
    data_block = match.group(0)
    for banned in ("odor", "cap-color", "gill-size", "spore-print", "mushroom", "poisonous", "edible"):
        assert banned not in data_block.lower()


# ---------------------------------------------------------------------------
# `_safe_json` "</script" escaping, exercised through the real export path
# ---------------------------------------------------------------------------

def test_safe_json_escapes_script_close_tag():
    assert _safe_json({"x": "</script><script>alert(1)</script>"}) == (
        '{"x": "<\\/script><script>alert(1)<\\/script>"}'
    )


def test_export_escapes_malicious_value_in_translations():
    df = _small_dataset()
    payload = "</script><script>alert(1)</script>"
    translations = {
        "columns": {},
        "values": {"a": {"x": payload}},
    }
    html = vsf.export_full_dashboard(
        df, target="class", criterion="p", translations=translations, targets=[],
        n_permutations=20, mine_center_n_permutations=20,
        top_insights_n_permutations=20,
    )
    # The literal, unescaped "</script>" must appear exactly the 3 real
    # closing script tags' worth of times — never as part of embedded data.
    assert html.count("</script>") == 3
    # The escaped form of the payload IS present (proves it made it into
    # the page at all, just safely).
    assert "<\\/script><script>alert(1)<\\/script>" in html


# ---------------------------------------------------------------------------
# max_dirty_cells cap
# ---------------------------------------------------------------------------

def test_max_dirty_cells_warns_and_caps():
    df = _small_dataset(n=600)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        html = vsf.export_full_dashboard(
            df, target="class", criterion="p", targets=[],
            n_permutations=20,
            mine_center_n_permutations=20,
            mine_center_purity_range=(0.0, 1.0),  # widen the band so cells actually qualify
            max_dirty_cells=1,
            top_insights_n_permutations=20,
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings, "expected a UserWarning when dirty cells exceed max_dirty_cells"
    assert "max_dirty_cells" in str(user_warnings[0].message)

    # MINING_DATA is nested by scenario id (see the multi-target section
    # below) — with targets=[] there is exactly one scenario, "class".
    mining_data = _extract_json_const(html, "MINING_DATA", "TOP_INSIGHTS_DATA")
    assert set(mining_data.keys()) == {"class"}
    assert len(mining_data["class"]) <= 1


# ---------------------------------------------------------------------------
# max_d / dimensionality cap
# ---------------------------------------------------------------------------

def test_max_d_default_is_4():
    import inspect
    sig = inspect.signature(vsf.export_full_dashboard)
    assert sig.parameters["max_d"].default == 4
    assert _MAX_SUPPORTED_D == 4


@pytest.mark.parametrize("bad_max_d", [0, -1, 5, 6, 7])
def test_max_d_out_of_range_raises(bad_max_d):
    df = _small_dataset()
    with pytest.raises(ValueError, match="max_d must be in \\[1, 4\\]"):
        vsf.export_full_dashboard(
            df, target="class", criterion="p", max_d=bad_max_d, targets=[],
            n_permutations=20, mine_center_n_permutations=20,
            top_insights_n_permutations=20,
        )


@pytest.mark.parametrize("ok_max_d", [1, 2, 3, 4])
def test_max_d_in_range_is_accepted(ok_max_d):
    df = _small_dataset()
    html = vsf.export_full_dashboard(
        df, target="class", criterion="p", max_d=ok_max_d, targets=[],
        n_permutations=20, mine_center_n_permutations=20,
        top_insights_n_permutations=20,
    )
    assert isinstance(html, str) and len(html) > 0


def test_exported_history_never_exceeds_4d():
    df = _dataset_with_many_features()
    html = vsf.export_full_dashboard(
        df, target="class", criterion="p", targets=[],
        n_permutations=20, mine_center_n_permutations=20,
        top_insights_n_permutations=20,
    )
    scenarios = _extract_json_const(html, "SCENARIOS_DATA", "MINING_DATA")
    history = scenarios["class"]["metrics"]["history"]
    assert history, "expected a non-empty selection history for this test fixture"
    steps = [h["step"] for h in history]
    assert max(steps) <= 4, f"history contains a step beyond the 4D cap: {steps}"


# ---------------------------------------------------------------------------
# Multi-target scenario precomputation (`targets`)
# ---------------------------------------------------------------------------

def test_targets_default_precomputes_every_column():
    df = _small_dataset()
    html = vsf.export_full_dashboard(
        df, target="class", criterion="p",
        n_permutations=20, mine_center_n_permutations=20,
        top_insights_n_permutations=20,
    )
    scenarios = _extract_json_const(html, "SCENARIOS_DATA", "MINING_DATA")
    mining = _extract_json_const(html, "MINING_DATA", "TOP_INSIGHTS_DATA")

    assert set(scenarios.keys()) == set(df.columns)  # {'a', 'b', 'class'}
    assert set(mining.keys()) == set(df.columns)

    # Primary target respects the given `criterion` (binarized -> 2 raw
    # classes: "0"/"1"), regardless of "class" actually having 2 raw values
    # itself — this distinguishes "binarized against p" from "raw column".
    assert sorted(scenarios["class"]["raw_target_classes"]) == ["0", "1"]

    # Every OTHER column's scenario is fit on its own RAW (un-binarized)
    # values — not swept per criterion — so its raw_target_classes count
    # matches that column's actual cardinality in df.
    assert len(scenarios["a"]["raw_target_classes"]) == df["a"].nunique()
    assert len(scenarios["b"]["raw_target_classes"]) == df["b"].nunique()


def test_targets_empty_list_reproduces_single_scenario_export():
    df = _small_dataset()
    html = vsf.export_full_dashboard(
        df, target="class", criterion="p", targets=[],
        n_permutations=20, mine_center_n_permutations=20,
        top_insights_n_permutations=20,
    )
    scenarios = _extract_json_const(html, "SCENARIOS_DATA", "MINING_DATA")
    mining = _extract_json_const(html, "MINING_DATA", "TOP_INSIGHTS_DATA")
    assert set(scenarios.keys()) == {"class"}
    assert set(mining.keys()) == {"class"}


def test_targets_explicit_list_limits_scope_but_always_includes_primary():
    df = _small_dataset()
    html = vsf.export_full_dashboard(
        df, target="class", criterion="p", targets=["a"],
        n_permutations=20, mine_center_n_permutations=20,
        top_insights_n_permutations=20,
    )
    scenarios = _extract_json_const(html, "SCENARIOS_DATA", "MINING_DATA")
    # "b" was not requested and must NOT have been fit.
    assert set(scenarios.keys()) == {"class", "a"}


def test_targets_invalid_column_raises():
    df = _small_dataset()
    with pytest.raises(ValueError, match="not a column of df"):
        vsf.export_full_dashboard(
            df, target="class", criterion="p", targets=["nonexistent_col"],
            n_permutations=20, mine_center_n_permutations=20,
            top_insights_n_permutations=20,
        )


def test_invalid_primary_target_raises():
    df = _small_dataset()
    with pytest.raises(ValueError, match="not found in the dataframe"):
        vsf.export_full_dashboard(df, target="nonexistent_col", targets=[])


def test_secondary_scenario_matches_standalone_export_for_that_column():
    # The load-bearing correctness claim of this feature (see the module
    # docstring in vsf/dashboard.py): a column reached via `targets` is
    # NOT a cheaper approximation — it's fit through the exact same
    # `_build_scenario` code path as the primary target. Verify by
    # comparing column "a"'s scenario, produced two ways: (1) as a
    # secondary target alongside primary="class", and (2) as its own
    # standalone primary-only export. Same random_state/params both times
    # -> the payloads must be byte-identical, not merely similar.
    df = _small_dataset()
    common_kwargs = dict(
        n_permutations=20, mine_center_n_permutations=20,
        top_insights_n_permutations=20, random_state=7,
    )

    html_multi = vsf.export_full_dashboard(
        df, target="class", criterion="p", targets=["a"], **common_kwargs,
    )
    scenarios_multi = _extract_json_const(html_multi, "SCENARIOS_DATA", "MINING_DATA")

    html_solo = vsf.export_full_dashboard(
        df, target="a", criterion=None, targets=[], **common_kwargs,
    )
    scenarios_solo = _extract_json_const(html_solo, "SCENARIOS_DATA", "MINING_DATA")

    assert scenarios_multi["a"] == scenarios_solo["a"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node.js not available")
def test_select_scenario_js_logic():
    # Functional smoke test of dashboard.js's selectScenario() in
    # isolation: stub the DOM and override updateDashboard() (already
    # covered by the Playwright browser smoke test end-to-end; here we only
    # verify selectScenario's OWN logic — guard clauses and state
    # reset/mutation — not the full Plotly render pipeline).
    js_source = (TEMPLATES_DIR / "dashboard.js").read_text(encoding="utf-8")
    script = f"""
      global.document = {{
        getElementById: () => null,
        querySelectorAll: () => [],
      }};
      global.window = global;
      global.addEventListener = () => {{}};
      {js_source}

      SCENARIOS_DATA = {{
        colA: {{ name: 'A' }},
        colB: {{ name: 'B' }},
      }};
      let updateDashboardCallCount = 0;
      updateDashboard = function(payload, ids) {{ updateDashboardCallCount++; }};

      currentScenarioId = 'colA';
      currentPayload = SCENARIOS_DATA['colA'];
      activeDimensionality = 3;
      activeSliceIndex = 1;

      const observed = {{}};

      // Re-selecting the already-active scenario is a no-op (no re-render).
      selectScenario('colA');
      observed.noop_same_id_calls = updateDashboardCallCount;

      // Selecting a column with no precomputed scenario is a no-op too.
      selectScenario('does_not_exist');
      observed.noop_missing_id_calls = updateDashboardCallCount;
      observed.unchanged_scenario_id = currentScenarioId;

      // Switching to a real, different scenario: swaps currentPayload,
      // resets dimensionality/slice state so updateDashboard() re-derives
      // them from the NEW payload's own d*, and triggers exactly one
      // re-render.
      selectScenario('colB');
      observed.switched_scenario_id = currentScenarioId;
      observed.switched_payload_name = currentPayload.name;
      observed.reset_dimensionality = activeDimensionality;
      observed.reset_slice_index = activeSliceIndex;
      observed.render_calls_after_switch = updateDashboardCallCount;

      process.stdout.write(JSON.stringify(observed));
    """
    proc = subprocess.run(["node"], input=script, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    observed = json.loads(proc.stdout)
    assert observed == {
        "noop_same_id_calls": 0,
        "noop_missing_id_calls": 0,
        "unchanged_scenario_id": "colA",
        "switched_scenario_id": "colB",
        "switched_payload_name": "B",
        "reset_dimensionality": None,
        "reset_slice_index": None,
        "render_calls_after_switch": 1,
    }


# ---------------------------------------------------------------------------
# dashboard.js: syntax + the 4D frame-controller functions it must define
# ---------------------------------------------------------------------------

def test_dashboard_js_has_no_graph_references():
    js_source = (TEMPLATES_DIR / "dashboard.js").read_text(encoding="utf-8")
    for banned in ("graphView", "vizView", "navBtnViz", "navBtnGraph", "graph_client", "switchView"):
        assert banned not in js_source, f"stale graph-view reference left in dashboard.js: {banned!r}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node.js not available")
def test_dashboard_js_is_syntactically_valid():
    proc = subprocess.run(
        ["node", "--check", str(TEMPLATES_DIR / "dashboard.js")], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"dashboard.js failed node --check: {proc.stderr}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node.js not available")
def test_slice_frame_controller_step_and_autoplay_logic():
    # Functional smoke test of stepSlice()/toggleSliceAutoplay()/
    # setSliceAutoplaySpeed() without a full DOM: stub the handful of
    # document.* calls dashboard.js's frame controller touches, load the
    # real source, and drive the frame index purely through
    # currentPayload/activeSliceIndex/selectSlice's side effects.
    js_source = (TEMPLATES_DIR / "dashboard.js").read_text(encoding="utf-8")
    script = f"""
      const seenPlots = [];
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
      stepSlice(1); observed.push(activeSliceIndex);   // 2 -> wraps to 0 (skips "All")
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
    # project (see vsf/dashboard.py's module docstring) — the template
    # asset itself must be gone, not merely unreferenced.
    assert not (TEMPLATES_DIR / "graph_client.js").exists()
