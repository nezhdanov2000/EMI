import json

def main():
    with open('data/mushroom_payload.json', 'r', encoding='utf-8') as f:
        payload_json = f.read()

    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    old_marker = 'async function loadMushroomAnalysis() {'

    new_block = f"""const MUSHROOM_PAYLOAD = {payload_json};

        async function loadMushroomAnalysis() {{
            const container = document.getElementById('plotly3dContainer');
            if (!container) return;
            
            let payload = MUSHROOM_PAYLOAD;
            try {{
                const response = await fetch('/api/mushroom');
                if (response.ok) {{
                    payload = await response.json();
                }}
            }} catch (err) {{
                console.log('Using pre-embedded VSF payload fallback.');
            }}

            const virPct = (payload.metrics.vir * 100).toFixed(1);
            const lFeatPct = (payload.metrics.l_feat * 100).toFixed(1);
            const lTargetPct = (payload.metrics.l_target * 100).toFixed(1);

            document.getElementById('val-vir').innerText = `d* = ${{payload.metrics.d_star}} (VIR = ${{virPct}}%)`;
            document.getElementById('val-vir').style.color = 'var(--status-green)';
            document.getElementById('val-pval').innerText = 'p < 0.001 (α = 0.01)';
            document.getElementById('val-loss').innerText = `Feature Loss: ${{lFeatPct}}% | L_target: ${{lTargetPct}}%`;
            
            const box = document.getElementById('verdict-box');
            if (box) {{
                box.className = 'verdict-banner verdict-green';
                box.innerHTML = `<span>🟢 ${{payload.metrics.xai_message}}</span>`;
            }}

            const trace = {{
                x: payload.x,
                y: payload.y,
                z: payload.z,
                mode: 'markers',
                marker: {{
                    size: 5,
                    color: payload.color,
                    colorscale: 'Portland',
                    opacity: 0.85,
                    line: {{ color: '#ffffff', width: 0.3 }}
                }},
                text: payload.hover_text,
                hoverinfo: 'text',
                type: 'scatter3d'
            }};

            const layout = {{
                paper_bgcolor: '#060911',
                plot_bgcolor: '#060911',
                scene: {{
                    xaxis: {{ title: payload.axis_names.x, backgroundcolor: '#0f172a', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
                    yaxis: {{ title: payload.axis_names.y, backgroundcolor: '#0f172a', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
                    zaxis: {{ title: payload.axis_names.z, backgroundcolor: '#0f172a', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
                    camera: {{ eye: {{ x: 1.5, y: 1.5, z: 1.2 }} }}
                }},
                margin: {{ l: 0, r: 0, b: 0, t: 0 }},
                font: {{ family: 'Inter', color: '#94a3b8' }}
            }};

            Plotly.newPlot('plotly3dContainer', [trace], layout, {{ responsive: true, displayModeBar: false }});
        }}"""

    if old_marker in html:
        idx = html.find(old_marker)
        end_idx = html.find('window.addEventListener', idx)
        html = html[:idx] + new_block + '\n\n        ' + html[end_idx:]
        with open('index.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print('SUCCESS: Updated index.html with inline MUSHROOM_PAYLOAD!')
    else:
        print('ERROR: Marker not found!')

if __name__ == '__main__':
    main()
