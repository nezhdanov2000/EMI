"""
VSF Visualization Module: Payload Generator & Standalone HTML Exporter
Prepares 3D visual coordinates, HUD metrics, and generates interactive WebGL scatter plots.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Union
from .avr import AVRResult


def prepare_visualization_payload(
    result: AVRResult,
    X_matrix: np.ndarray,
    Z_target: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_display_samples: int = 2000,
) -> Dict:
    """
    Prepares a structured visualization payload from AVRResult, X matrix, and Z target.
    
    Returns:
        Dict containing 3D coordinates, color encoding, hover metadata, and VSF HUD metrics.
    """
    X_arr = np.asarray(X_matrix)
    Z_arr = np.asarray(Z_target).ravel()
    n_samples, n_features = X_arr.shape
    
    if feature_names is None:
        feature_names = [f"Feature_{j+1}" for j in range(n_features)]

    # Subsample if dataset > max_display_samples for smooth WebGL rendering
    if n_samples > max_display_samples:
        indices = np.random.default_rng(42).choice(n_samples, size=max_display_samples, replace=False)
        X_sub = X_arr[indices]
        Z_sub = Z_arr[indices]
    else:
        indices = np.arange(n_samples)
        X_sub = X_arr
        Z_sub = Z_arr

    selected_idx = result.selected_features
    d_star = result.d_star

    # Extract 3D Axes (X, Y, Z coordinates)
    x_col_idx = selected_idx[0] if len(selected_idx) > 0 else 0
    y_col_idx = selected_idx[1] if len(selected_idx) > 1 else (1 if n_features > 1 else 0)
    z_col_idx = selected_idx[2] if len(selected_idx) > 2 else (2 if n_features > 2 else 0)

    x_vals = X_sub[:, x_col_idx]
    y_vals = X_sub[:, y_col_idx]
    z_vals = X_sub[:, z_col_idx]

    # Convert non-numeric / string arrays to string representations for hover
    x_str = [str(val) for val in x_vals]
    y_str = [str(val) for val in y_vals]
    z_str = [str(val) for val in z_vals]
    z_target_str = [str(val) for val in Z_sub]

    # Discrete numeric mapping for 3D plot positioning
    _, x_num = np.unique(x_vals, return_inverse=True)
    _, y_num = np.unique(y_vals, return_inverse=True)
    _, z_num = np.unique(z_vals, return_inverse=True)
    _, color_num = np.unique(Z_sub, return_inverse=True)

    hover_texts = [
        f"<b>Sample #{idx}</b><br>"
        f"Target ({'Class'}): {z_target_str[i]}<br>"
        f"{feature_names[x_col_idx]}: {x_str[i]}<br>"
        f"{feature_names[y_col_idx]}: {y_str[i]}<br>"
        f"{feature_names[z_col_idx]}: {z_str[i]}"
        for i, idx in enumerate(indices)
    ]

    return {
        "x": x_num.tolist(),
        "y": y_num.tolist(),
        "z": z_num.tolist(),
        "color": color_num.tolist(),
        "target_labels": z_target_str,
        "hover_text": hover_texts,
        "axis_names": {
            "x": feature_names[x_col_idx],
            "y": feature_names[y_col_idx],
            "z": feature_names[z_col_idx],
        },
        "all_feature_names": feature_names,
        "selected_features": result.selected_feature_names,
        "metrics": {
            "d_star": result.d_star,
            "scenario": result.scenario.value,
            "vir": float(result.vir),
            "l_target": float(result.l_target),
            "l_feat": float(result.l_feat),
            "nmi_full": float(result.nmi_full),
            "xai_message": result.xai_message,
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
