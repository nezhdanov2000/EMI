"""
vsf.dashboard: standalone, self-contained HTML dashboard export.

`export_full_dashboard()` bakes a *primary* fitted AVR scenario (a chosen
target/criterion) plus, by default, one additional raw-target AVR scenario
per remaining column in `df` — each with its own 1D-4D visualization
payload and dirty-center rule-mining results — into a single HTML file
that works via `file://` with no running server. The catalog tree's column
items are live: clicking one swaps the 3D visualizer to that column's
precomputed scenario client-side (see `dashboard.js`'s `selectScenario`),
no refit, no network request.

Architecture note (read this before touching anything below): every
statistic in the exported page — feature selection, purity, Miller-Madow-
corrected local NMI, Benjamini-Hochberg-corrected significance — is
computed HERE, in Python, by calling the same `vsf` functions the live
`server.py` calls. The client-side JavaScript this module embeds
(`vsf/templates/dashboard.js`) never recomputes a p-value or refits
anything; it only re-slices, masks, counts, and renders data that was
already validated in Python. That split is deliberate: reimplementing
hypothesis testing in JavaScript with no independent test coverage is
exactly the kind of duplicated, unverified logic this project's code-
review pass spent a great deal of effort removing from the Python side.
Keep new interactivity on the "cheap lookup/count" side of that line, or
compute it here instead.

Dimensionality is capped at 4D — see `_MAX_SUPPORTED_D` below — because 4D
is where this export's spatial encoding actually stops: `vsf.vis.
prepare_visualization_payload` only ever positions samples on 3 spatial
axes (X/Y/Z) plus, at d* >= 4, a 4th feature used as a discrete TIME/FRAME
axis (`payload["slice_axis"]`, rendered as the frame controller in
`dashboard.js`) — it has no 5th/6th/7th spatial or color/shape encoding.
`vsf.AVREngine` itself supports selecting up to `max_d` (7 by default)
features for the live app's HUD-only "Scenario C" warning math; this
export clamps `max_d` to the display's actual ceiling instead of letting
the UI grow "5D"/"6D"/"7D" dimensionality buttons that select a feature
subset with no distinct visual representation (`buildPlotData` would
silently reuse the 3D/4D grid for any of them, mislabeled).

Multi-scenario precomputation (see `targets` on `export_full_dashboard`):
  - The PRIMARY scenario (`target`/`criterion`) is fit exactly as before —
    `criterion` binarizes against a specific value, or `None` uses the raw
    (possibly multiclass) column.
  - Every OTHER column named in `targets` (default: every column in `df`)
    gets its own scenario using that column as a RAW multiclass target
    (`criterion` is not swept — that would be a combinatorial explosion of
    column x criterion-value scenarios, not the "one scenario per column"
    this feature asks for). Each such scenario is a fully independent
    `AVREngine.fit` + `prepare_visualization_payload` +
    dirty-center-mining run — no shortcuts, no data shared across
    scenarios except the dataframe itself — so a column reached by
    clicking it in the catalog is identical to what re-exporting with that
    column as the primary `target` would have produced.
  - Cost is genuinely per-scenario: fitting N targets is ~N times the
    single-scenario runtime and ~N times the single-scenario payload size
    (dirty-center mining time varies most, since it scales with how many
    grid cells actually land in the ambiguous purity band for that
    target — a cleanly separable target can be ~1s, a noisier one tens of
    seconds). Pass an explicit `targets` list (or `targets=[]` for
    single-scenario-only, the old default) to bound this.
  - The "Filter Search" composite AND-condition builder from the live app
    is NOT ported. It reruns the full permutation-tested AVR feature
    selection on an arbitrary user-built condition, which cannot be
    exhaustively precomputed and is not safe to reimplement client-side.
    Use the live `server.py` app for that feature.
  - Graph Inference / Knowledge-Base chain mining (the live app's
    `graph.html`) is NOT ported — it belongs to a separate reasoning-graph
    project and does not share this export's "one 3D/4D scatter scenario"
    scope. Use the live `server.py` app for that feature.
"""

from __future__ import annotations

import json
import warnings
from importlib import resources
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .avr import AVREngine
from .mining import compute_top_insights, mine_dirty_center
from .vis import (
    Translations,
    catalog_from_dataframe,
    humanize_col,
    humanize_val,
    prepare_visualization_payload,
)

__all__ = ["export_full_dashboard"]

_GRID_DIMS_FOR_MINING = ("1", "2", "3")

# Hard ceiling on `max_d`: `vsf.vis.prepare_visualization_payload` only
# builds a 3D spatial grid (X/Y/Z) plus, at d* >= 4, one additional feature
# used as a discrete time/frame axis — there is no encoding beyond that to
# cap a higher `max_d` against. See the module docstring above.
_MAX_SUPPORTED_D = 4


def _safe_json(obj: Any) -> str:
    """
    Serializes `obj` for direct inlining into a `<script>const X = ...;`
    block (NOT a `JSON.parse`'d string — see the `<script>` embedding in
    `_render_html`).

    Two things this guards against that a plain `json.dumps` call would
    not:
      1. A literal "</script" appearing inside embedded string data (e.g.
         a hover-text value) would prematurely close the surrounding
         `<script>` tag in the browser's HTML parser — this is a real
         (if usually latent) bug class for any "embed JSON in a script
         tag" pattern, `vsf.vis.generate_interactive_html`'s existing
         payload embedding included. Escaping "</" as "<\\/" — a no-op for
         JSON semantics, since "\\/" and "/" decode identically — defuses
         it without touching the data.
      2. `json.dumps`'s default `allow_nan=True` emits bare `NaN`/
         `Infinity`/`-Infinity` tokens, which are invalid JSON but ARE
         valid JavaScript identifiers (global values) — since this output
         is inlined as a JS literal rather than passed to `JSON.parse`,
         that default is exactly right here and is kept deliberately
         (raising on a legitimate NaN metric would just crash the export).
    """
    return json.dumps(obj, allow_nan=True).replace("</", "<\\/")


def _read_template(name: str) -> str:
    return resources.files("vsf.templates").joinpath(name).read_text(encoding="utf-8")


def _prepare_scenario(
    df: pd.DataFrame, target: str, criterion: Optional[str], translations: Optional[Translations]
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, str]:
    """
    Mirrors `server.py`'s `_handle_analyze_api` target-preparation logic
    for a plain (non-composite) target, so the baked-in scenario is
    identical to what clicking this same target/criterion in the live app
    would produce. `sort_Z` is always the raw column values (used for
    target-conditioned axis-category ordering), independent of whether a
    binary `criterion` was requested.
    """
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found in the dataframe")
    sort_Z = df[target].values
    if criterion is not None:
        Z = (df[target].astype(str) == str(criterion)).astype(int).values
        display_target_name = (
            f"{humanize_col(target, translations)} = "
            f"{humanize_val(target, str(criterion), translations)}"
        )
    else:
        Z = df[target].values
        display_target_name = target
    X_df = df.drop(columns=[target])
    return Z, sort_Z, X_df, display_target_name


def _prepare_mine_center_target(
    df: pd.DataFrame, target: str, criterion: Optional[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Mirrors `server.py`'s `_handle_mine_center_api` target-preparation
    branch (non-composite): the plain-multiclass case integer-codes a
    categorical `Z` via `np.unique(..., return_inverse=True)` before
    handing it to `mine_dirty_center`, which the AVR-fit path does not do.
    Kept as a distinct helper (rather than reusing `_prepare_scenario`'s Z)
    to preserve that exact, previously-reviewed behavior.
    """
    drop_cols = [target]
    if criterion is not None:
        Z = (df[target].astype(str) == str(criterion)).astype(int).values
    else:
        Z = df[target].values
        if df[target].dtype.kind in ("U", "S", "O", "b"):
            _, Z = np.unique(Z, return_inverse=True)
    return Z, drop_cols


def _iter_grid_cells(payload: Dict[str, Any], dims=_GRID_DIMS_FOR_MINING):
    """Yields (dim, cell_index, cdata) for every cell in the requested grids."""
    grids = payload.get("grids", {})
    for dim in dims:
        grid = grids.get(dim)
        if not grid:
            continue
        for idx, cdata in enumerate(grid.get("customdata", [])):
            if cdata is not None:
                yield dim, idx, cdata


def _precompute_dirty_center_mining(
    df: pd.DataFrame,
    target: str,
    criterion: Optional[str],
    payload: Dict[str, Any],
    translations: Optional[Translations],
    purity_range: Tuple[float, float],
    max_cells: int,
    n_permutations: int,
    fdr_q: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Runs `mine_dirty_center` once for every occupied grid cell (across the
    1D/2D/3D views) whose purity falls in `purity_range` — the same
    "dirty center" gate the live app's `plotly_click` handler already
    applies before firing `/api/mine_center` — and writes a `mining_key`
    onto each such cell's `customdata` entry (mutating `payload` in
    place) so the client never has to reconstruct a lookup key from raw
    coordinate values. Returns `{mining_key: mine_dirty_center(...) results}`.

    Bounded by construction (grid cells are finitely many, and only those
    in the dirty band are mined), but a denser dataset than the one this
    was validated against could still produce more occupied cells than is
    reasonable to mine in one export call — `max_cells` caps that, and a
    `UserWarning` is raised (never silently) if the cap is hit, naming how
    many cells were skipped.
    """
    lo, hi = purity_range
    i_z_x_f = float(payload.get("metrics", {}).get("nmi") or 1.0)

    candidates: List[Tuple[str, int, Dict[str, Any]]] = []
    for dim, idx, cdata in _iter_grid_cells(payload):
        pur = cdata.get("pur")
        if pur is None or not (lo <= pur <= hi):
            continue
        candidates.append((dim, idx, cdata))

    candidates.sort(key=lambda c: c[2].get("N", 0), reverse=True)
    if len(candidates) > max_cells:
        warnings.warn(
            f"vsf.dashboard: {len(candidates)} dirty cells (purity in "
            f"[{lo}, {hi}]) found for target={target!r} criterion={criterion!r}, "
            f"exceeding max_dirty_cells={max_cells}. Mining the {max_cells} "
            f"largest by sample count; the remaining {len(candidates) - max_cells} "
            "cells will be clickable in the exported dashboard but show no "
            "precomputed rules. Raise max_dirty_cells to cover them.",
            UserWarning,
            stacklevel=2,
        )
        candidates = candidates[:max_cells]

    mining_data: Dict[str, List[Dict[str, Any]]] = {}
    for dim, idx, cdata in candidates:
        coords: Dict[str, Any] = cdata.get("coords", {})
        Z, drop_cols = _prepare_mine_center_target(df, target, criterion)
        mask = np.ones(len(df), dtype=bool)
        for col, val in coords.items():
            if col in df.columns:
                mask &= df[col].astype(str) == str(val)
                if col not in drop_cols:
                    drop_cols.append(col)
        X_df = df.drop(columns=drop_cols)

        results = mine_dirty_center(
            X_df,
            Z,
            mask,
            i_z_x_f,
            n_permutations=n_permutations,
            fdr_q=fdr_q,
            translations=translations,
        )
        mining_key = f"{dim}:{idx}"
        cdata["mining_key"] = mining_key
        mining_data[mining_key] = results

    return mining_data


def _build_scenario(
    df: pd.DataFrame,
    target: str,
    criterion: Optional[str],
    translations: Optional[Translations],
    alpha: float,
    vir_threshold: float,
    max_d: int,
    n_permutations: int,
    random_state: Optional[int],
    mine_center_purity_range: Tuple[float, float],
    max_dirty_cells: int,
    mine_center_n_permutations: int,
    mine_center_fdr_q: float,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Fits one complete AVR scenario end-to-end: feature selection ->
    1D-4D visualization payload -> dirty-center rule mining. This is the
    ONE code path both `export_full_dashboard`'s primary (target,
    criterion) scenario and every additional per-column scenario in
    `targets` run through — so a column reached by clicking it in the
    exported catalog is byte-for-byte identical to what re-exporting with
    that column passed as the primary `target` would have produced, never
    a cheaper/approximate variant.
    """
    Z, sort_Z, X_df, display_target_name = _prepare_scenario(df, target, criterion, translations)
    feature_names = list(X_df.columns)
    X = X_df.values

    engine = AVREngine(
        alpha=alpha,
        vir_threshold=vir_threshold,
        max_d=max_d,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    res = engine.fit(X, Z, feature_names=feature_names)
    payload = prepare_visualization_payload(
        res,
        X,
        Z,
        feature_names=feature_names,
        target_name=display_target_name,
        sort_Z=sort_Z,
        translations=translations,
    )
    mining_data = _precompute_dirty_center_mining(
        df,
        target,
        criterion,
        payload,
        translations,
        mine_center_purity_range,
        max_dirty_cells,
        mine_center_n_permutations,
        mine_center_fdr_q,
    )
    return payload, mining_data


def _render_html(
    title: str,
    dashboard_data: Dict[str, Any],
    scenarios_data: Dict[str, Any],
    mining_data: Dict[str, Any],
    top_insights: Dict[str, Any],
) -> str:
    css = _read_template("dashboard.css")
    body = _read_template("dashboard_shell.html")
    dashboard_js = _read_template("dashboard.js")

    data_block = (
        "<script>\n"
        f"const DASHBOARD_DATA = {_safe_json(dashboard_data)};\n"
        f"const SCENARIOS_DATA = {_safe_json(scenarios_data)};\n"
        f"const MINING_DATA = {_safe_json(mining_data)};\n"
        f"const TOP_INSIGHTS_DATA = {_safe_json(top_insights)};\n"
        "</script>\n"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
{css}
</style>
</head>
<body>
{body}
{data_block}
<script>
{dashboard_js}
</script>
</body>
</html>
"""


def export_full_dashboard(
    df: pd.DataFrame,
    target: str = "class",
    criterion: Optional[str] = None,
    targets: Optional[List[str]] = None,
    translations: Optional[Translations] = None,
    title: str = "VSF Interactive Dashboard",
    alpha: float = 0.01,
    vir_threshold: float = 0.85,
    max_d: int = _MAX_SUPPORTED_D,
    n_permutations: int = 100,
    random_state: Optional[int] = 42,
    mine_center_purity_range: Tuple[float, float] = (0.3, 0.7),
    mine_center_n_permutations: int = 200,
    mine_center_fdr_q: float = 0.05,
    max_dirty_cells: int = 300,
    top_insights_min_nmi: float = 0.75,
    top_insights_n_permutations: int = 200,
    top_insights_fdr_q: float = 0.05,
) -> str:
    """
    Builds one self-contained HTML dashboard: a fitted AVR 3D+time visual
    scenario per target column (with 1D/2D/3D coordinate switching, a 4D
    time/frame controller with play/pause/speed animation, purity-contour
    highlighting, and precomputed dirty-center rule mining), switchable
    client-side by clicking any column in the Insights Catalog — a single
    `file://`-openable page with no running server.

    Parameters mirror the live server's defaults so the primary scenario
    matches what clicking the same target/criterion in `server.py` would
    show. See the module docstring for the features deliberately NOT
    ported (the composite AND-filter builder, and Graph Inference /
    Knowledge-Base chain mining) and for the multi-scenario cost tradeoff.

    Args:
        df: the full dataset (e.g. `pandas.read_csv(...)`), target column
            included.
        target: PRIMARY column to analyze — the scenario shown when the
            page first loads. Default "class" matches the demo app.
        criterion: a specific value of `target` to binarize against (e.g.
            "p"), or `None` for the raw (possibly multiclass) column —
            matches the live app's initial-load view when `None`. Only
            applies to the primary `target`; every other scenario in
            `targets` always uses its raw column (see the module
            docstring for why criterion-sweeping isn't done there too).
        targets: additional columns to precompute their own switchable
            scenario for, each fit as `AVREngine.fit(X, Z=<that column>,
            ...)` (raw, criterion=None). `None` (the default) means every
            column in `df` — `target` is always included regardless of
            whether it's in this list. Pass `[]` to export only the
            primary scenario (the pre-multi-scenario behavior). Every name
            must be an actual column of `df`.
        translations: optional dataset-specific display table (see
            `vsf.vis.Translations`); falls back to raw column/value
            strings with none, same as every other `vsf` function.
        title: HTML `<title>`.
        alpha, vir_threshold, n_permutations, random_state: passed
            through to `vsf.AVREngine` for EVERY scenario's fit — defaults
            match `server.py`.
        max_d: upper bound on how many axes AVR may select — passed
            through to `vsf.AVREngine` for every scenario, and also gates
            the exported UI's dimensionality buttons/frame controller (see
            the module docstring). Must be in `[1, 4]`; this export's
            spatial encoding stops at 3D + one time/frame axis, so a
            higher value would only grow feature selection with no
            corresponding visual representation.
        mine_center_purity_range: purity band (inclusive) a grid cell must
            fall in to be mined for dirty-center rules — matches the live
            app's client-side gate before firing `/api/mine_center`.
            Applied independently within every scenario.
        mine_center_n_permutations, mine_center_fdr_q: passed to
            `mine_dirty_center` per dirty cell, in every scenario.
        max_dirty_cells: safety cap on how many dirty cells get mined per
            SCENARIO (see `_precompute_dirty_center_mining`); raises
            `UserWarning`, never silently truncates, if exceeded — once
            per scenario that exceeds it.
        top_insights_min_nmi, top_insights_n_permutations,
            top_insights_fdr_q: passed to `compute_top_insights` for the
            "Top Insights" catalog panel — computed once for the whole
            dataset, independent of `target`/`targets`.

    Returns:
        The complete HTML document as a string. Write it to a file, e.g.
        `Path("dashboard.html").write_text(vsf.export_full_dashboard(df))`.

    Cost: with the `targets=None` default, this fits AVR (and mines dirty
    centers) once per column of `df` — for the 23-column UCI Mushroom demo
    dataset (8124 rows) this measured ~1-2s per scenario for the AVR fit +
    payload, plus 0-35s per scenario for dirty-center mining depending on
    how many grid cells land in the ambiguous purity band for that
    column's split (a cleanly-separable target mines almost nothing; a
    noisy one mines dozens of cells) — several minutes end-to-end, and an
    output file tens of MB (each scenario embeds its own full per-sample
    1D-4D grid data, so file size is ~N_scenarios x single-scenario size).
    Pass a shorter `targets` list to bound both.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df).__name__}")
    if not (1 <= max_d <= _MAX_SUPPORTED_D):
        raise ValueError(
            f"max_d must be in [1, {_MAX_SUPPORTED_D}] — this export's spatial encoding is "
            f"3 coordinate axes plus one 4D time/frame axis (see vsf.dashboard's module "
            f"docstring); got max_d={max_d!r}."
        )
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found in the dataframe")

    # Resolve the full scenario column list up front (target always first,
    # duplicates dropped, order otherwise preserved) so a bad name in
    # `targets` fails fast — before spending any time on the primary fit.
    other_targets = list(df.columns) if targets is None else list(targets)
    resolved_targets: List[str] = []
    _seen = set()
    for t in [target] + other_targets:
        if t not in _seen:
            _seen.add(t)
            resolved_targets.append(t)
    for t in resolved_targets:
        if t not in df.columns:
            raise ValueError(
                f"targets contains {t!r}, which is not a column of df. "
                f"df.columns = {list(df.columns)!r}"
            )

    scenario_kwargs = dict(
        alpha=alpha,
        vir_threshold=vir_threshold,
        max_d=max_d,
        n_permutations=n_permutations,
        random_state=random_state,
        mine_center_purity_range=mine_center_purity_range,
        max_dirty_cells=max_dirty_cells,
        mine_center_n_permutations=mine_center_n_permutations,
        mine_center_fdr_q=mine_center_fdr_q,
    )

    payload, mining_data = _build_scenario(df, target, criterion, translations, **scenario_kwargs)

    scenarios: Dict[str, Any] = {target: payload}
    scenario_mining: Dict[str, Any] = {target: mining_data}
    for t in resolved_targets:
        if t == target:
            continue
        t_payload, t_mining = _build_scenario(df, t, None, translations, **scenario_kwargs)
        scenarios[t] = t_payload
        scenario_mining[t] = t_mining

    catalog = catalog_from_dataframe(df, translations)
    for col in catalog:
        for crit in col["criteria"]:
            crit["active"] = (col["id"] == target and criterion is not None and crit["id"] == str(criterion))

    top_insights = compute_top_insights(
        df,
        min_nmi=top_insights_min_nmi,
        n_permutations=top_insights_n_permutations,
        fdr_q=top_insights_fdr_q,
        random_state=random_state,
        translations=translations,
    )
    for col in top_insights["columns"]:
        for crit in col["criteria"]:
            crit["active"] = (col["id"] == target and criterion is not None and crit["id"] == str(criterion))

    dashboard_data = {
        "target": target,
        "criterion": criterion,
        "catalog": catalog,
        "total_rows": len(df),
    }

    return _render_html(title, dashboard_data, scenarios, scenario_mining, top_insights)
