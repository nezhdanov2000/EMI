"""
VSF Visualization Module: Payload Generator & Standalone HTML Exporter
Prepares 3D visual coordinates, HUD metrics, and generates interactive WebGL scatter plots.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Union
from .avr import AVRResult


# Human-readable English translations for UCI Mushroom Dataset
MUSHROOM_TRANSLATIONS = {
    "columns": {
        "class": "Edibility",
        "cap-shape": "Cap Shape",
        "cap-surface": "Cap Surface",
        "cap-color": "Cap Color",
        "bruises": "Bruises",
        "odor": "Odor",
        "gill-attachment": "Gill Attachment",
        "gill-spacing": "Gill Spacing",
        "gill-size": "Gill Size",
        "gill-color": "Gill Color",
        "stalk-shape": "Stalk Shape",
        "stalk-root": "Stalk Root",
        "stalk-surface-above-ring": "Stalk Surface Above Ring",
        "stalk-surface-below-ring": "Stalk Surface Below Ring",
        "stalk-color-above-ring": "Stalk Color Above Ring",
        "stalk-color-below-ring": "Stalk Color Below Ring",
        "veil-type": "Veil Type",
        "veil-color": "Veil Color",
        "ring-number": "Ring Number",
        "ring-type": "Ring Type",
        "spore-print-color": "Spore Print Color",
        "population": "Population",
        "habitat": "Habitat",
    },
    "values": {
        "class": {"e": "Edible (e)", "p": "Poisonous (p)"},
        "cap-shape": {
            "b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"
        },
        "cap-surface": {
            "f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"
        },
        "cap-color": {
            "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green",
            "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"
        },
        "bruises": {"t": "bruises", "f": "no bruises"},
        "odor": {
            "a": "almond", "l": "anise", "c": "creosote", "y": "fishy", 
            "f": "foul", "m": "musty", "n": "none", "p": "pungent", "s": "spicy"
        },
        "gill-attachment": {
            "a": "attached", "d": "descending", "f": "free", "n": "notched"
        },
        "gill-spacing": {
            "c": "close", "w": "crowded", "d": "distant"
        },
        "gill-size": {
            "b": "broad", "n": "narrow"
        },
        "gill-color": {
            "k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray",
            "r": "green", "o": "orange", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"
        },
        "stalk-shape": {
            "e": "enlarging", "t": "tapering"
        },
        "stalk-root": {
            "b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs", "r": "rooted", "?": "missing"
        },
        "stalk-surface-above-ring": {
            "f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"
        },
        "stalk-surface-below-ring": {
            "f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"
        },
        "stalk-color-above-ring": {
            "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
            "p": "pink", "e": "red", "w": "white", "y": "yellow"
        },
        "stalk-color-below-ring": {
            "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
            "p": "pink", "e": "red", "w": "white", "y": "yellow"
        },
        "veil-type": {
            "p": "partial", "u": "universal"
        },
        "veil-color": {
            "n": "brown", "o": "orange", "w": "white", "y": "yellow"
        },
        "ring-number": {
            "n": "none", "o": "one", "t": "two"
        },
        "ring-type": {
            "c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large",
            "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"
        },
        "spore-print-color": {
            "k": "black", "n": "brown", "b": "buff", "h": "chocolate", 
            "r": "green", "o": "orange", "u": "purple", "w": "white", "y": "yellow"
        },
        "population": {
            "a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered", "v": "several", "y": "solitary"
        },
        "habitat": {
            "g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", 
            "u": "urban", "w": "waste", "d": "woods"
        },
    }
}


def humanize_val(col_name: str, val: str) -> str:
    """Translates raw category code to human readable name if available."""
    val_str = str(val)
    if col_name in MUSHROOM_TRANSLATIONS["values"]:
        return MUSHROOM_TRANSLATIONS["values"][col_name].get(val_str, val_str)
    if "class" in col_name and val_str in MUSHROOM_TRANSLATIONS["values"]["class"]:
        return MUSHROOM_TRANSLATIONS["values"]["class"][val_str]
    return val_str


def humanize_col(col_name: str) -> str:
    """Translates column name to human readable name."""
    ru_name = MUSHROOM_TRANSLATIONS["columns"].get(col_name, col_name)
    return f"{ru_name} ({col_name})" if ru_name != col_name else col_name


def target_conditioned_sort(x_vals: np.ndarray, z_vals: np.ndarray, col_name: str = ""):
    """
    Sorts categories of x_vals based on their association with the target z_vals.
    Uses deterministic tie-breaking (by frequency and lexical order) to prevent spatial warping.
    Returns:
        x_num: Integer coordinates for x_vals
        x_sorted_labels: List of human-readable category string labels in sorted order (for axis ticks)
    """
    _, z_idx = np.unique(z_vals, return_inverse=True)
    x_unique, x_counts = np.unique(x_vals, return_counts=True)
    
    cat_stats = []
    for c, cnt in zip(x_unique, x_counts):
        mask = (x_vals == c)
        score = float(np.mean(z_idx[mask])) if np.any(mask) else 0.0
        cat_stats.append((score, -int(cnt), str(c), c))
        
    cat_stats.sort(key=lambda item: (item[0], item[1], item[2]))
    x_sorted = [item[3] for item in cat_stats]
    
    x_to_num = {val: i for i, val in enumerate(x_sorted)}
    x_num = np.array([x_to_num[val] for val in x_vals])
    
    # Translate tick labels into human readable Russian words
    x_labels = [humanize_val(col_name, v) for v in x_sorted]
    return x_num, x_labels


def prepare_visualization_payload(
    result: AVRResult,
    X_matrix: np.ndarray,
    Z_target: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_display_samples: int = 10000,
    target_name: str = "class",
    sort_Z: Optional[np.ndarray] = None,
) -> Dict:
    """
    Prepares a structured visualization payload with human readable Russian axis titles,
    category labels, color mappings, and cluster occupancy density counts.
    """
    X_arr = np.asarray(X_matrix)
    Z_arr = np.asarray(Z_target).ravel()
    n_samples, n_features = X_arr.shape
    
    if feature_names is None:
        feature_names = [f"Feature_{j+1}" for j in range(n_features)]

    if n_samples > max_display_samples:
        indices = np.random.default_rng(42).choice(n_samples, size=max_display_samples, replace=False)
        X_sub = X_arr[indices]
        Z_sub = Z_arr[indices]
    else:
        indices = np.arange(n_samples)
        X_sub = X_arr
        Z_sub = Z_arr

    if sort_Z is not None:
        sort_Z_arr = np.asarray(sort_Z).ravel()
        sort_Z_sub = sort_Z_arr[indices] if len(sort_Z_arr) == n_samples else sort_Z_arr
    else:
        sort_Z_sub = Z_sub

    selected_idx = result.selected_features
    d_star = result.d_star

    x_col_idx = selected_idx[0] if len(selected_idx) > 0 else 0
    y_col_idx = selected_idx[1] if len(selected_idx) > 1 else (1 if n_features > 1 else 0)
    z_col_idx = selected_idx[2] if len(selected_idx) > 2 else (2 if n_features > 2 else 0)

    # 4th dimension (slice axis) — extracted when d* >= 4
    has_4d = d_star >= 4 and len(selected_idx) >= 4
    w_col_idx = selected_idx[3] if has_4d else None

    x_name = feature_names[x_col_idx]
    y_name = feature_names[y_col_idx]
    z_name = feature_names[z_col_idx]
    w_name = feature_names[w_col_idx] if has_4d else None

    x_vals = X_sub[:, x_col_idx]
    y_vals = X_sub[:, y_col_idx]
    z_vals = X_sub[:, z_col_idx]
    w_vals = X_sub[:, w_col_idx] if has_4d else None

    # Human-readable value strings
    x_human = [humanize_val(x_name, v) for v in x_vals]
    y_human = [humanize_val(y_name, v) for v in y_vals]
    z_human = [humanize_val(z_name, v) for v in z_vals]
    w_human = [humanize_val(w_name, v) for v in w_vals] if has_4d else None
    z_target_human = [humanize_val(target_name, v) for v in Z_sub]

    # Target-Conditioned Categorical Ordering using canonical sort_Z_sub
    x_num, x_ticks = target_conditioned_sort(x_vals, sort_Z_sub, col_name=x_name)
    y_num, y_ticks = target_conditioned_sort(y_vals, sort_Z_sub, col_name=y_name)
    z_num, z_ticks = target_conditioned_sort(z_vals, sort_Z_sub, col_name=z_name)
    if has_4d:
        w_num, w_ticks = target_conditioned_sort(w_vals, sort_Z_sub, col_name=w_name)
    else:
        w_num, w_ticks = None, None
    
    unique_targets, color_num = np.unique(Z_sub, return_inverse=True)
    unique_target_labels = [humanize_val(target_name, t) for t in unique_targets]

    # Group sample indices by cell for 3D Voxel Crystal Lattice Packing
    from collections import defaultdict, Counter
    cell_groups = defaultdict(list)
    for idx_in_sub, (cx, cy, cz) in enumerate(zip(x_num, y_num, z_num)):
        cell_groups[(cx, cy, cz)].append(idx_in_sub)

    x_cube = np.zeros(len(x_num), dtype=float)
    y_cube = np.zeros(len(y_num), dtype=float)
    z_cube = np.zeros(len(z_num), dtype=float)

    for (cx, cy, cz), cell_indices in cell_groups.items():
        # Sort samples within cell by target class Z so colors form clean stratified layers in the cube
        cell_indices.sort(key=lambda idx: color_num[idx])
        N_cell = len(cell_indices)
        
        # Grid edge dimension S (cube root)
        S = int(np.ceil(N_cell ** (1.0 / 3.0)))
        if S <= 1:
            step = 0.0
        else:
            # Clean spacing so individual spheres are clearly visible with gaps
            step = min(0.052, 0.78 / max(S - 1, 1))

        for rank, idx_in_sub in enumerate(cell_indices):
            # Compute 3D lattice indices (i, j, k)
            i = rank % S
            j = (rank // S) % S
            k = rank // (S * S)

            off_x = (i - (S - 1) / 2.0) * step
            off_y = (j - (S - 1) / 2.0) * step
            off_z = (k - (S - 1) / 2.0) * step

            x_cube[idx_in_sub] = np.round(cx + off_x, 4)
            y_cube[idx_in_sub] = np.round(cy + off_y, 4)
            z_cube[idx_in_sub] = np.round(cz + off_z, 4)

    coords = list(zip(x_num, y_num, z_num))
    cell_counts = Counter(coords)

    target_display_name = humanize_col(target_name)

    hover_texts = [
        f"<b>🍄 Sample #{idx+1}</b><br>"
        f"🎯 <b>{target_display_name}:</b> {z_target_human[i]}<br>"
        f"📍 <b>{humanize_col(x_name)}:</b> {x_human[i]}<br>"
        f"📍 <b>{humanize_col(y_name)}:</b> {y_human[i]}<br>"
        f"📍 <b>{humanize_col(z_name)}:</b> {z_human[i]}<br>"
        f"📦 <b>Cube Volume (Density):</b> {cell_counts[(x_num[i], y_num[i], z_num[i])]} samples in cell"
        for i, idx in enumerate(indices)
    ]

    def build_grid(dim, slice_mask=None):
        """Build discrete center grid. If slice_mask is provided, only include samples where slice_mask[i] is True."""
        g_groups = defaultdict(list)
        for idx_in_sub, (cx, cy, cz) in enumerate(zip(x_num, y_num, z_num)):
            if slice_mask is not None and not slice_mask[idx_in_sub]:
                continue
            _cy = cy if dim >= 2 else -0.5
            _cz = cz if dim >= 3 else -0.5
            g_groups[(cx, _cy, _cz)].append(idx_in_sub)
            
        gx, gy, gz, gop, gpur, gsz, ghov, gdata = [], [], [], [], [], [], [], []
        m_N = max([len(lst) for lst in g_groups.values()]) if g_groups else 1
        
        for (cx, cy, cz), c_idx in g_groups.items():
            N_c = len(c_idx)
            c_cols = [color_num[i] for i in c_idx]
            pur = float(np.mean(c_cols)) / max(len(unique_targets) - 1, 1)
            norm_d = 0.2 + 0.8 * (np.sqrt(N_c) / np.sqrt(m_N))
            
            gx.append(float(cx))
            gy.append(float(cy))
            gz.append(float(cz))
            gop.append(float(norm_d))
            gpur.append(float(pur))
            gsz.append(N_c)

            
            hx = x_human[c_idx[0]]
            hy = y_human[c_idx[0]] if dim >= 2 else "Collapsed"
            hz = z_human[c_idx[0]] if dim >= 3 else "Collapsed"
            
            # Include 4D slice label in hover if applicable
            hw = w_human[c_idx[0]] if (has_4d and w_human and slice_mask is not None) else None
            
            target_pos_label = unique_target_labels[-1] if unique_target_labels else "Positive Class"
            
            dim_label = "4D Slice" if (slice_mask is not None and has_4d) else f"{dim}D"
            hov = (
                f"<b>📍 Discrete Center ({dim_label})</b><br>"
                f"🎯 <b>Share of {target_pos_label}:</b> {pur*100:.1f}%<br>"
                f"📦 <b>Objects:</b> {N_c} pcs.<br>"
                f"💠 <b>X:</b> {hx}<br>"
                f"💠 <b>Y:</b> {hy}<br>"
                f"💠 <b>Z:</b> {hz}"
            )
            if hw is not None:
                hov += f"<br>🎞️ <b>{humanize_col(w_name)}:</b> {hw}"
            ghov.append(hov)
            
            # Use raw values for API queries so filtering works, cast to standard types for JSON
            def _cast(val):
                if hasattr(val, 'item'): return val.item()
                return val
            
            rx = _cast(x_vals[c_idx[0]])
            ry = _cast(y_vals[c_idx[0]]) if dim >= 2 else None
            rz = _cast(z_vals[c_idx[0]]) if dim >= 3 else None
            rw = _cast(w_vals[c_idx[0]]) if (has_4d and w_vals is not None and slice_mask is not None) else None
            
            cdata = {"N": N_c, "pur": float(pur), "coords": {x_name: rx}}
            if dim >= 2: cdata["coords"][y_name] = ry
            if dim >= 3: cdata["coords"][z_name] = rz
            if rw is not None: cdata["coords"][w_name] = rw
            gdata.append(cdata)
            
        return {"x": gx, "y": gy, "z": gz, "opacity": gop, "purity": gpur, "sizes": gsz, "hover_text": ghov, "customdata": gdata}

    # Calculate global max points per cell based on 1D view for consistent scaling
    g_groups_1d = {}
    for cx in x_num:
        g_groups_1d[cx] = g_groups_1d.get(cx, 0) + 1
    global_max_n = max(g_groups_1d.values()) if g_groups_1d else 1

    grids = {
        "1": build_grid(1),
        "2": build_grid(2),
        "3": build_grid(3)
    }

    # Generate per-slice grids for 4D visualization
    slice_axis_info = None
    if has_4d and w_num is not None and w_ticks is not None:
        slice_counts = []
        for sv_idx in range(len(w_ticks)):
            mask = (w_num == sv_idx)
            grids[f"4_{sv_idx}"] = build_grid(3, slice_mask=mask)
            slice_counts.append(int(np.sum(mask)))
        grids["4_all"] = build_grid(3)
        slice_axis_info = {
            "name": humanize_col(w_name),
            "ticks": w_ticks,
            "counts": slice_counts
        }

    return {
        "x": x_num.tolist(),
        "y": y_num.tolist(),
        "z": z_num.tolist(),
        "x_jitter": x_cube.tolist(),
        "y_jitter": y_cube.tolist(),
        "z_jitter": z_cube.tolist(),
        "grids": grids,
        "global_max_n": global_max_n,
        "grid_sizes": grids["3"]["sizes"],
        "grid_x": grids["3"]["x"],
        "grid_y": grids["3"]["y"],
        "grid_z": grids["3"]["z"],
        "grid_opacity": grids["3"]["opacity"],
        "grid_purity": grids["3"]["purity"],
        "grid_hover_text": grids["3"]["hover_text"],
        "grid_customdata": grids["3"].get("customdata", []),
        "color": color_num.tolist(),
        "target_labels": z_target_human,
        "unique_target_classes": unique_target_labels,
        "raw_target_classes": [str(t) for t in unique_targets],
        "target_name": target_display_name,
        "hover_text": hover_texts,
        "axis_names": {
            "x": humanize_col(x_name),
            "y": humanize_col(y_name),
            "z": humanize_col(z_name),
        },
        "axis_ticks": {
            "x": {"vals": list(range(len(x_ticks))), "text": x_ticks},
            "y": {"vals": list(range(len(y_ticks))), "text": y_ticks},
            "z": {"vals": list(range(len(z_ticks))), "text": z_ticks},
        },
        "slice_axis": slice_axis_info,
        "total_samples": len(indices),
        "all_feature_names": [humanize_col(fn) for fn in feature_names],
        "raw_feature_names": feature_names,
        "selected_features": [humanize_col(sfn) for sfn in result.selected_feature_names],
        "metrics": {
            "d_star": result.d_star,
            "scenario": result.scenario.value,
            "vir": float(result.vir),
            "nmi": float(result.nmi_full if result.d_star == 0 else (1.0 - result.l_target)),
            "l_target": float(result.l_target),
            "l_feat": float(result.l_feat),
            "nmi_full": float(result.nmi_full),
            "xai_message": result.xai_message,
            "history": [
                {
                    **h,
                    "feature": humanize_col(h["feature"]),
                    "alternatives": [
                        {
                            **alt,
                            "feature": humanize_col(alt["feature"])
                        } for alt in h.get("alternatives", [])
                    ]
                } for h in result.selection_history
            ] if hasattr(result, 'selection_history') and result.selection_history else [],
        },
    }


def generate_interactive_html(payload: Dict, title: str = "VSF 3D Visualizer") -> str:
    """
    Generates a standalone, beautiful glassmorphism dark HTML document with Plotly 3D scatter plot.
    """
    payload_json = json.dumps(payload)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@500;700&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: #070a12;
            color: #f3f4f6;
            overflow: hidden;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            padding: 16px 24px;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }}
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(135deg, #a855f7, #6366f1, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .badge {{
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .SCENARIO_A {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }}
        .SCENARIO_B {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }}
        .SCENARIO_C {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .SCENARIO_D {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        
        main {{
            flex: 1;
            position: relative;
            display: flex;
        }}
        #plot {{
            width: 100%;
            height: 100%;
        }}
        .hud-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            width: 320px;
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            z-index: 5;
        }}
        .metric-card {{
            margin-bottom: 14px;
        }}
        .metric-title {{
            font-size: 12px;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 4px;
        }}
        .metric-value {{
            font-size: 22px;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            color: #ffffff;
        }}
        .xai-box {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 12px;
            font-size: 13px;
            line-height: 1.5;
            color: #d1d5db;
            border-left: 3px solid #8b5cf6;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Visual Sufficiency Framework (VSF) — 3D Visualizer</h1>
        <span id="scenarioBadge" class="badge">Loading...</span>
    </header>
    <main>
        <div id="plot"></div>
        <div class="hud-panel">
            <div class="metric-card">
                <div class="metric-title">Optimal Dimensionality (d*)</div>
                <div id="dStarVal" class="metric-value">-</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Visual Information Ratio (VIR)</div>
                <div id="virVal" class="metric-value">-</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Target Projection Loss (L_target)</div>
                <div id="lTargetVal" class="metric-value">-</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Selected Visual Axes</div>
                <div id="axesVal" style="font-size: 14px; color: #c084fc; font-weight: 600; margin-top: 4px;">-</div>
            </div>
            <div class="xai-box" id="xaiMsg">-</div>
        </div>
    </main>

    <script>
        const payload = {payload_json};
        
        // Populate HUD
        document.getElementById('dStarVal').innerText = payload.metrics.d_star + " Visual Axes";
        document.getElementById('virVal').innerText = (payload.metrics.vir * 100).toFixed(1) + "%";
        document.getElementById('lTargetVal').innerText = (payload.metrics.l_target * 100).toFixed(1) + "%";
        document.getElementById('axesVal').innerText = payload.selected_features.join(', ');
        document.getElementById('xaiMsg').innerText = payload.metrics.xai_message;
        
        const badge = document.getElementById('scenarioBadge');
        badge.innerText = payload.metrics.scenario;
        badge.className = 'badge ' + payload.metrics.scenario;
        
        // Plot 3D Scatter
        const trace = {{
            x: payload.x,
            y: payload.y,
            z: payload.z,
            mode: 'markers',
            marker: {{
                size: 6,
                color: payload.color,
                colorscale: 'Viridis',
                opacity: 0.85,
                line: {{ color: '#ffffff', width: 0.5 }}
            }},
            text: payload.hover_text,
            hoverinfo: 'text',
            customdata: payload.customdata,
            type: 'scatter3d'
        }};
        
        const layout = {{
            paper_bgcolor: '#070a12',
            plot_bgcolor: '#070a12',
            scene: {{
                xaxis: {{ title: payload.axis_names.x, backgroundcolor: '#0f172a', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
                yaxis: {{ title: payload.axis_names.y, backgroundcolor: '#0f172a', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
                zaxis: {{ title: payload.axis_names.z, backgroundcolor: '#0f172a', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
                camera: {{ eye: {{ x: 1.5, y: 1.5, z: 1.2 }} }}
            }},
            margin: {{ l: 0, r: 0, b: 0, t: 0 }},
            font: {{ family: 'Inter', color: '#94a3b8' }}
        }};
        
        Plotly.newPlot('plot', [trace], layout, {{ responsive: true, displayModeBar: false }});
    </script>
</body>
</html>
"""
    return html_content
