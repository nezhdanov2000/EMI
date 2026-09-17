"""
vsf.dashboard: standalone, self-contained HTML dashboard export.

`export_full_dashboard()`
bakes up to 4 independently-discovered branches (Project_Master_Document.md
Section 4, `vsf.avr.discover_branches`) — one per dimensionality 1D-4D — for
a SINGLE chosen target/criterion into one self-contained HTML file that
works via `file://` with no running server. The branch selector in the
exported page is live: clicking a dimensionality swaps the 3D visualizer to
that branch's precomputed payload client-side (see `dashboard.js`'s
`selectBranch`), no refit, no network request.

Removed relative to v1.0 (see the master doc's revision table for the full
list and reasoning — all four removals are direct product decisions, not
oversights):
  - The `targets=` multi-scenario sweep (one scenario per COLUMN of `df`).
    This module now always fits exactly one target/criterion, expressed as
    up to 4 branches (dimensionalities), not N independent per-column
    scenarios. Re-running `export_full_dashboard` with a different `target`
    is how you look at a different column now, same as the live app.
  - Dirty-center conjunctive-filter mining (`vsf.mining.mine_dirty_center`)
    and its `mine_center_*`/`max_dirty_cells` parameters — the module they
    lived in, `vsf.mining`, no longer exists.
  - The "Top Insights" auto-discovery catalog (`vsf.mining.compute_top_insights`)
    and its `top_insights_*` parameters — same module removal.
  - Graph Inference / Knowledge-Base chain mining was never ported to this
    export (unchanged from v1.0) — it belonged to a separate reasoning-graph
    project, now deleted from the live app too.

Architecture note (unchanged from v1.0, still the governing principle):
every statistic in the exported page — feature selection, purity — is
computed HERE, in Python, by calling the same `vsf` functions the live
`vsf.serve()` app calls. The client-side JavaScript this module embeds
(`vsf/templates/dashboard.js`) never recomputes a statistic or refits
anything; it only re-slices, masks, counts, and renders data already
computed in Python.

Dimensionality is capped at 4 — see `vsf.avr.MAX_BRANCH_D` — because that is
where this export's spatial encoding actually stops: `vsf.vis.
prepare_visualization_payload` only ever positions samples on 3 spatial axes
(X/Y/Z) plus, for a 4D branch, a 4th feature used as a discrete TIME/FRAME
axis (`payload["slice_axis"]`, rendered as the frame controller in
`dashboard.js`) — there is no 5th/6th/7th spatial or color/shape encoding to
select for.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, Dict, Optional

import pandas as pd

from .avr import DEFAULT_N_PERMUTATIONS, MAX_BRANCH_D, Direction, discover_branches
from .centers import CenterSpec
from .vis import (
    Translations,
    catalog_from_dataframe,
    humanize_col,
    humanize_val,
    prepare_visualization_payload,
)

__all__ = ["export_full_dashboard"]


def _safe_json(obj: Any) -> str:
    """
    Serializes `obj` for direct inlining into a `<script>const X = ...;`
    block (NOT a `JSON.parse`'d string — see the `<script>` embedding in
    `_render_html`).

    Two things this guards against that a plain `json.dumps` call would
    not:
      1. A literal "</script" appearing inside embedded string data (e.g.
         a hover-text value) would prematurely close the surrounding
         `<script>` tag in the browser's HTML parser. Escaping "</" as
         "<\\/" — a no-op for JSON semantics, since "\\/" and "/" decode
         identically — defuses it without touching the data.
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


def _prepare_target(
    df: pd.DataFrame, target: str, criterion: Optional[str], translations: Optional[Translations],
    direction: Direction = "presence",
):
    """
    Mirrors `vsf.server`'s `_build_analyze_response` target-preparation logic,
    so the baked-in branches are identical to what selecting this same
    target/criterion in the live app would produce. `sort_Z` is always the
    raw column values (used for target-conditioned axis-category ordering),
    independent of whether a binary `criterion` was requested.
    """
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found in the dataframe")
    sort_Z = df[target].values
    indicator_labels = None
    if criterion is not None:
        Z = (df[target].astype(str) == str(criterion)).astype(int).values
        display_target_name = (
            f"{humanize_col(target, translations)} = "
            f"{humanize_val(target, str(criterion), translations)}"
        )
        if direction == "absence":
            # The renderer's positive indicator is the complement (see
            # `vsf.avr.Direction`); the class labels keep naming the value.
            Z = 1 - Z
            indicator_labels = (display_target_name, f"not {display_target_name}")
            display_target_name = (
                f"{humanize_col(target, translations)} \u2260 "
                f"{humanize_val(target, str(criterion), translations)}"
            )
    elif direction == "absence":
        raise ValueError("an absence export needs an explicit criterion whose absence to certify")
    else:
        Z = df[target].values
        display_target_name = target
    X_df = df.drop(columns=[target])
    return Z, sort_Z, X_df, display_target_name, indicator_labels


def _render_html(
    title: str,
    dashboard_data: Dict[str, Any],
    branches_data: Dict[str, Any],
) -> str:
    css = _read_template("dashboard.css")
    body = _read_template("dashboard_shell.html")
    dashboard_js = _read_template("dashboard.js")

    data_block = (
        "<script>\n"
        f"const DASHBOARD_DATA = {_safe_json(dashboard_data)};\n"
        f"const BRANCHES_DATA = {_safe_json(branches_data)};\n"
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


#: The certificate an export is built under when the caller names none: the
#: same one the live application colours by by default.
DEFAULT_EXPORT_SPEC = CenterSpec()


def export_full_dashboard(
    df: pd.DataFrame,
    target: str = "class",
    criterion: Optional[str] = None,
    translations: Optional[Translations] = None,
    title: str = "VSF Interactive Dashboard",
    max_d: int = MAX_BRANCH_D,
    center_spec: Optional[CenterSpec] = None,
    direction: Direction = "presence",
) -> str:
    """
    Builds one self-contained HTML dashboard: up to 4 independently
    discovered branches (Project_Master_Document.md Section 4) for a single
    target/criterion, each with its own 1D-3D coordinate grid (or, for the
    4D branch, a 4D time/frame controller with play/pause/speed animation)
    and purity-based coloring — switchable client-side by clicking a
    dimensionality in the branch selector, a single `file://`-openable page
    with no running server.

    Args:
        df: the full dataset (e.g. `pandas.read_csv(...)`), target column
            included.
        target: the column to analyze. Default "class" matches the demo
            app; pass the actual target column name for any other dataset.
        criterion: a specific value of `target` to binarize against (e.g.
            "p"), or `None` to use the raw column directly -- valid only
            when `target` itself has exactly two distinct values (the higher
            one is then the positive class, matching the live app). A
            multiclass `target` with `criterion=None` raises `ValueError`
            from `discover_branches`: v2.3 requires a resolvable positive
            class always (see `vsf.avr`'s module docstring) -- there is no
            longer an association-based ranking to fall back to.
        translations: optional dataset-specific display table (see
            `vsf.vis.Translations`); falls back to raw column/value strings
            with none, same as every other `vsf` function.
        title: HTML `<title>`.
        center_spec: the discrete-centre certificate (`vsf.centers.CenterSpec`)
            baked into this export: the purity floor `tau`, the simultaneous
            error rate `alpha`, and the multiplicity policy. Defaults to
            `DEFAULT_EXPORT_SPEC` = `CenterSpec()`: a cell is green when
            the share of the value among its rows reaches tau. A static page
            cannot be re-certified after the fact, so this value is final
            for the exported document and is stated in its legend.
        direction: `"presence"` (default) or `"absence"` — see
            `vsf.avr.Direction`. Under `"absence"` the exported branches
            certify cells where `criterion` is almost missing; they are
            drawn red, every coverage figure refers to rows without the
            value, and `criterion` is required.
        max_d: upper bound on branch dimensionality — passed through to
            `vsf.avr.discover_branches`. Must be in `[1, 4]`; this export's
            spatial encoding stops at 3D + one time/frame axis (see the
            module docstring), so a higher value has no corresponding
            visual representation.

    Returns:
        The complete HTML document as a string. Write it to a file, e.g.
        `Path("dashboard.html").write_text(vsf.export_full_dashboard(df))`.

    Cost: `discover_branches` performs an honest, unbounded full
    enumeration of every feature combination for every dimensionality up to
    `max_d` (Project_Master_Document.md Section 4.6) — for the 22-feature
    UCI Mushroom demo dataset this is ~9,100 combinations total and
    finishes in low single-digit seconds; a much wider dataset (M gtrsim 50)
    should be expected to take minutes. This is the documented, accepted
    cost of never approximating the search, not a defect.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df).__name__}")
    if not (1 <= max_d <= MAX_BRANCH_D):
        raise ValueError(
            f"max_d must be in [1, {MAX_BRANCH_D}] — this export's spatial encoding is "
            f"3 coordinate axes plus one 4D time/frame axis (see vsf.dashboard's module "
            f"docstring); got max_d={max_d!r}."
        )
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found in the dataframe")

    Z, sort_Z, X_df, display_target_name, indicator_labels = _prepare_target(
        df, target, criterion, translations, direction
    )
    feature_names = list(X_df.columns)
    X = X_df.values

    # v2.2: the export carries the certified-centre layer, which is what
    # v2.3 also ranks branches by. A static page cannot re-run anything, so
    # it must ship the certificate it was built under; `spec` is baked into
    # every payload and into the legend. `discover_branches` raises
    # ValueError when `criterion=None` and `target` is not itself
    # two-valued -- see this function's docstring.
    spec = center_spec if center_spec is not None else DEFAULT_EXPORT_SPEC
    positive_class = 1 if criterion is not None else None
    branches = discover_branches(
        X, Z, feature_names=feature_names, max_d=max_d, random_state=0,
        positive_class=positive_class, center_spec=spec,
        n_permutations_centers=DEFAULT_N_PERMUTATIONS,
        direction=direction,
    )

    branches_data: Dict[str, Any] = {}
    for d, branch in branches.items():
        # The spec the search resolved (a family certificate carries its
        # family size) - the display must certify at the same level.
        spec = branch.centers.spec
        payload = prepare_visualization_payload(
            branch,
            X,
            Z,
            feature_names=feature_names,
            target_name=display_target_name,
            # `_prepare_target` returns a One-vs-Rest 0/1 vector whenever a
            # criterion was given, so its class labels must read as the
            # criterion and its negation rather than as "0" and "1".
            target_is_indicator=criterion is not None,
            sort_Z=sort_Z,
            translations=translations,
            positive_value=(1 if criterion is not None else None),
            center_spec=spec,
            indicator_labels=indicator_labels,
        )
        branches_data[str(d)] = payload

    catalog = catalog_from_dataframe(df, translations)
    for col in catalog:
        for crit in col["criteria"]:
            crit["active"] = (col["id"] == target and criterion is not None and crit["id"] == str(criterion))

    dashboard_data = {
        "target": target,
        "criterion": criterion,
        "direction": direction,
        "catalog": catalog,
        "total_rows": len(df),
        "branch_dims": sorted(branches.keys()),
        "default_branch": str(max(branches.keys())) if branches else None,
    }

    return _render_html(title, dashboard_data, branches_data)
