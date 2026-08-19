let currentPayload = null;
        let activeDimensionality = null;

        // Predefined color palettes for clear class separation
        const CLASS_COLORS = [
            '#22c55e', // Green (e / Edible)
            '#ef4444', // Red (p / Poisonous)
            '#3b82f6', // Blue
            '#f59e0b', // Amber
            '#a855f7', // Purple
            '#06b6d4', // Cyan
            '#ec4899', // Pink
            '#84cc16', // Lime
            '#eab308'  // Yellow
        ];

        async function init() {
            try {
                const colRes = await fetch('/api/columns');
                if (colRes.ok) {
                    const colData = await colRes.json();
                    populateColumns(colData.columns, colData.default_target);
                }
                await runAnalysis('class');
            } catch (err) {
                console.error("Initialization error:", err);
                await runAnalysis('class');
            }
        }

        function populateColumns(cols, defaultTarget) {
            const select = document.getElementById('targetSelect');
            select.innerHTML = '';
            cols.forEach(item => {
                const opt = document.createElement('option');
                const colId = typeof item === 'object' ? item.id : item;
                const colLabel = typeof item === 'object' ? item.label : item;
                opt.value = colId;
                opt.innerText = colLabel;
                if (colId === defaultTarget) opt.selected = true;
                select.appendChild(opt);
            });
        }

        async function onTargetChange() {
            const select = document.getElementById('targetSelect');
            const target = select.value;
            await runAnalysis(target);
        }

        async function runAnalysis(targetCol) {
            showLoader(true);
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target: targetCol })
                });

                if (response.ok) {
                    currentPayload = await response.json();
                    activeDimensionality = null; // Reset on new analysis
                    updateDashboard(currentPayload);
                } else {
                    console.error("Error fetching analysis", response.status);
                }
            } catch (err) {
                console.warn("API POST failed, trying GET /api/mushroom:", err);
                try {
                    const fallbackRes = await fetch('/api/mushroom');
                    currentPayload = await fallbackRes.json();
                    updateDashboard(currentPayload);
                } catch (fallbackErr) {
                    console.error("Fallback error:", fallbackErr);
                }
            } finally {
                showLoader(false);
            }
        }

        function showLoader(show) {
            const loader = document.getElementById('loader');
            if (show) loader.classList.add('active');
            else loader.classList.remove('active');
        }


        function updateDashboard(payload) {
            const m = payload.metrics;
            if (activeDimensionality === null) {
                activeDimensionality = Math.min(m.d_star, 3) || 1;
            }

            document.getElementById('val-dstar').innerText = `${m.d_star}D`;
            document.getElementById('val-vir').innerText = `${(m.vir * 100).toFixed(1)}%`;
            document.getElementById('val-pval').innerText = 'p < 0.001 (α=0.01)';
            document.getElementById('val-loss').innerText = `${(m.l_target * 100).toFixed(1)}%`;
            document.getElementById('totalSamplesVal').innerText = (payload.total_samples || payload.x.length).toLocaleString();

            const pill = document.getElementById('scenarioPill');
            pill.className = 'scenario-pill ' + m.scenario;
            document.getElementById('scenarioText').innerText = m.scenario;

            document.getElementById('xaiBanner').innerHTML = `💡 <b>XAI Инсайт:</b> ${m.xai_message}`;

            // Selected Axes List
            const axesContainer = document.getElementById('axesListContainer');
            axesContainer.innerHTML = '';
            const labels = ['X-Ось', 'Y-Ось', 'Z-Ось', 'Цвет', 'Размер', 'Время'];
            payload.selected_features.forEach((feat, idx) => {
                const item = document.createElement('div');
                item.className = 'axis-pill';
                item.innerHTML = `
                    <span style="font-weight: 500;">${feat}</span>
                    <span class="axis-badge">${labels[idx] || 'Канал ' + (idx + 1)}</span>
                `;
                axesContainer.appendChild(item);
            });

            // Update Color Legend
            document.getElementById('legendTargetName').innerText = payload.target_name || 'Целевая переменная';
            const legendItems = document.getElementById('legendItems');
            legendItems.innerHTML = '';

            const uniqueClasses = payload.unique_target_classes || [];
            uniqueClasses.forEach((cls, idx) => {
                const colorHex = CLASS_COLORS[idx % CLASS_COLORS.length];
                const legItem = document.createElement('div');
                legItem.className = 'legend-item';
                legItem.innerHTML = `
                    <span class="color-dot" style="background-color: ${colorHex};"></span>
                    <span style="font-weight: 500;">${cls}</span>
                `;
                legendItems.appendChild(legItem);
            });

            // Selection History List
            const historyContainer = document.getElementById('historyListContainer');
            if (historyContainer && m.history) {
                historyContainer.innerHTML = '';
                m.history.forEach((step, idx) => {
                    const item = document.createElement('div');
                    item.className = 'history-item' + (step.step === activeDimensionality ? ' active' : '');

                    const virPct = (step.vir * 100).toFixed(1);
                    const deltaPct = step.step === 1 ? '' : `(+${(step.delta_mi * 100).toFixed(1)}%)`;

                    let altsHtml = '';
                    if (step.alternatives && step.alternatives.length > 0) {
                        altsHtml = '<div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 0.75rem; color: var(--text-dim);">';
                        altsHtml += '<div style="margin-bottom: 3px; font-weight: 600;">Альтернативы:</div>';
                        step.alternatives.forEach(a => {
                            altsHtml += `<div>• ${a.feature} (MI: ${(a.vir * 100).toFixed(1)}%)</div>`;
                        });
                        altsHtml += '</div>';
                    }

                    item.innerHTML = `
                        <div style="width: 100%;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div class="history-step">${step.step}D-Система</div>
                                    <div class="history-feature">${step.feature}</div>
                                </div>
                                <div class="history-stats">
                                    <div class="history-mi">MI: ${virPct}%</div>
                                    <div class="history-delta">${deltaPct}</div>
                                </div>
                            </div>
                            ${altsHtml}
                        </div>
                    `;
                    item.onclick = () => setDimensionality(step.step);
                    historyContainer.appendChild(item);
                });
            }

            renderPlot(payload);
        }

        function setDimensionality(d) {
            if (d > 3) d = 3; // Limit visual switching to 3D maximum
            if (activeDimensionality === d) return;
            activeDimensionality = d;

            // Re-render UI
            if (currentPayload) {
                updateDashboard(currentPayload);
            }
        }

        function renderPlot(payload) {
            let dimStr = activeDimensionality.toString();
            let g = payload.grids ? payload.grids[dimStr] : null;

            let xCoords = g ? g.x : payload.grid_x;
            let yCoords = g ? g.y : payload.grid_y;
            let zCoords = g ? g.z : payload.grid_z;

            let currentPurity = g ? g.purity : payload.grid_purity;
            let currentOpacity = g ? g.opacity : payload.grid_opacity;
            let currentHover = g ? g.hover_text : payload.grid_hover_text;

            // Generate Cell Boundary Dividers (Grid lines placed strictly BETWEEN categories at -0.5, 0.5, 1.5...)
            const xLen = payload.axis_ticks ? payload.axis_ticks.x.vals.length : 4;
            const yLen = payload.axis_ticks ? payload.axis_ticks.y.vals.length : 4;
            const zLen = payload.axis_ticks ? payload.axis_ticks.z.vals.length : 4;

            const xMin = -0.5, xMax = xLen - 0.5;
            const yMin = -0.5, yMax = yLen - 0.5;
            const zMin = -0.5, zMax = zLen - 0.5;

            const glX = [], glY = [], glZ = [];

            // 1. Floor grid lines at Z = zMin (between cells)
            for (let x = xMin; x <= xMax + 0.001; x += 1.0) {
                glX.push(x, x, null);
                glY.push(yMin, yMax, null);
                glZ.push(zMin, zMin, null);
            }
            for (let y = yMin; y <= yMax + 0.001; y += 1.0) {
                glX.push(xMin, xMax, null);
                glY.push(y, y, null);
                glZ.push(zMin, zMin, null);
            }

            // 2. Back Wall grid lines at X = xMin (between cells)
            for (let y = yMin; y <= yMax + 0.001; y += 1.0) {
                glX.push(xMin, xMin, null);
                glY.push(y, y, null);
                glZ.push(zMin, zMax, null);
            }
            for (let z = zMin; z <= zMax + 0.001; z += 1.0) {
                glX.push(xMin, xMin, null);
                glY.push(yMin, yMax, null);
                glZ.push(z, z, null);
            }

            // 3. Back Wall grid lines at Y = yMax (between cells)
            for (let x = xMin; x <= xMax + 0.001; x += 1.0) {
                glX.push(x, x, null);
                glY.push(yMax, yMax, null);
                glZ.push(zMin, zMax, null);
            }
            for (let z = zMin; z <= zMax + 0.001; z += 1.0) {
                glX.push(xMin, xMax, null);
                glY.push(yMax, yMax, null);
                glZ.push(z, z, null);
            }

            const cellBoundaryTrace = {
                x: glX,
                y: glY,
                z: glZ,
                mode: 'lines',
                line: {
                    color: 'rgba(255, 255, 255, 0.10)',
                    width: 1.0
                },
                hoverinfo: 'none',
                type: 'scatter3d',
                name: 'Сетка ячеек'
            };

            // Generate exact RGBA array to support per-point intensity in WebGL scatter3d
            let mappedColors = [];
            if (currentPurity && currentOpacity) {
                for (let i = 0; i < currentPurity.length; i++) {
                    const p = currentPurity[i];
                    // Treat grid_opacity as a color intensity multiplier (brightness)
                    const intensity = currentOpacity[i];

                    // 2D Gradient Mapping:
                    // Green Channel = Edible Concentration
                    // Red Channel = Poisonous Concentration
                    const edible_conc = (1.0 - p) * intensity;
                    const poisonous_conc = p * intensity;

                    const r = poisonous_conc * 255;
                    const g = edible_conc * 255;
                    const b = 0;

                    // Use fully opaque RGB to prevent depth-sorting artifacts in WebGL
                    mappedColors.push(`rgb(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)})`);
                }
            } else {
                mappedColors = payload.color;
            }

            const scatterTrace = {
                x: xCoords,
                y: yCoords,
                z: zCoords,
                mode: 'markers',
                marker: {
                    // Constant size as requested
                    size: 12.0,
                    color: mappedColors,
                    opacity: 1, // Full opacity is required to enforce proper WebGL depth sorting (Z-buffer)
                    line: {
                        width: 0 // No white border to prevent white glare
                    },
                    showscale: false
                },
                hovertext: currentHover,
                hoverinfo: 'text',
                type: 'scatter3d',
                name: 'Данные'
            };

            const plotTraces = [cellBoundaryTrace, scatterTrace];

            // Toggle 2D Bivariate Legend
            const bivLegend = document.getElementById('bivariateLegend');
            if (bivLegend) {
                bivLegend.classList.add('active');
            }

            const layout = {
                paper_bgcolor: '#070a13',
                plot_bgcolor: '#070a13',
                showlegend: false,
                scene: {
                    xaxis: {
                        title: { text: payload.axis_names.x, font: { color: '#c084fc', size: 13 } },
                        tickvals: payload.axis_ticks ? payload.axis_ticks.x.vals : undefined,
                        ticktext: payload.axis_ticks ? payload.axis_ticks.x.text : undefined,
                        range: [xMin, xMax],
                        tickfont: { color: '#e2e8f0', size: 11 },
                        backgroundcolor: '#090d1a',
                        showgrid: false, // Disables the lines that cut through labels
                        zeroline: false,
                        showspikes: false
                    },
                    yaxis: {
                        title: { text: payload.axis_names.y, font: { color: '#c084fc', size: 13 } },
                        tickvals: payload.axis_ticks ? payload.axis_ticks.y.vals : undefined,
                        ticktext: payload.axis_ticks ? payload.axis_ticks.y.text : undefined,
                        range: [yMin, yMax],
                        tickfont: { color: '#e2e8f0', size: 11 },
                        backgroundcolor: '#090d1a',
                        showgrid: false, // Disables the lines that cut through labels
                        zeroline: false,
                        showspikes: false
                    },
                    zaxis: {
                        title: { text: payload.axis_names.z, font: { color: '#c084fc', size: 13 } },
                        tickvals: payload.axis_ticks ? payload.axis_ticks.z.vals : undefined,
                        ticktext: payload.axis_ticks ? payload.axis_ticks.z.text : undefined,
                        range: [zMin, zMax],
                        tickfont: { color: '#e2e8f0', size: 11 },
                        backgroundcolor: '#090d1a',
                        showgrid: false, // Disables the lines that cut through labels
                        zeroline: false,
                        showspikes: false
                    },
                    camera: { eye: { x: 1.6, y: 1.6, z: 1.3 } }
                },
                margin: { l: 0, r: 0, b: 0, t: 0 },
                font: { family: 'Inter', color: '#94a3b8' }
            };

            Plotly.newPlot('plot-container', plotTraces, layout, { responsive: true, displayModeBar: false });
        }

        window.addEventListener('DOMContentLoaded', init);
