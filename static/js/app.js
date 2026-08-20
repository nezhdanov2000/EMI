let currentPayload = null;
let activeDimensionality = null;
let activeCritItem = null;
let allColumnsData = [];

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
            allColumnsData = colData.columns;
            populateCatalog(colData.columns, colData.default_target);
            // Add default first filter row
            addFilterRow();
        }
        await runAnalysis('class', null);
    } catch (err) {
        console.error("Initialization error:", err);
        await runAnalysis('class', null);
    }
}

function populateCatalog(cols, defaultTarget) {
    const accordion = document.getElementById('catalogAccordion');
    accordion.innerHTML = '';

    cols.forEach(col => {
        const charItem = document.createElement('div');
        charItem.className = 'char-item';

        const charHeader = document.createElement('div');
        charHeader.className = 'char-header';
        charHeader.innerHTML = `
                    <span class="char-title">${col.label}</span>
                    <span class="char-icon">▶</span>
                `;

        const charContent = document.createElement('div');
        charContent.className = 'char-content';

        charHeader.onclick = () => {
            charItem.classList.toggle('open');
        };

        col.criteria.forEach(crit => {
            const critItem = document.createElement('div');
            critItem.className = 'crit-item';

            const critHeader = document.createElement('div');
            critHeader.className = 'crit-header';
            critHeader.innerHTML = `<span>${crit.label}</span>`;

            const critContent = document.createElement('div');
            critContent.className = 'crit-content';
            // Unique ID for the history container
            const historyListId = `history-${col.id}-${crit.id.replace(/[^a-zA-Z0-9]/g, '_')}`;
            critContent.id = historyListId;

            critHeader.onclick = async (e) => {
                e.stopPropagation();
                if (activeCritItem && activeCritItem !== critItem) {
                    activeCritItem.classList.remove('active');
                    activeCritItem.classList.remove('open');
                }

                const isActive = critItem.classList.contains('active');
                if (!isActive) {
                    critItem.classList.add('active');
                    activeCritItem = critItem;
                    critContent.innerHTML = '<div style="color:var(--text-dim);font-size:0.8rem;padding:4px;">Анализ Парето-фронта...</div>';
                    critItem.classList.add('open');
                    await runAnalysis(col.id, crit.id, historyListId);
                } else {
                    critItem.classList.toggle('open');
                }
            };

            critItem.appendChild(critHeader);
            critItem.appendChild(critContent);
            charContent.appendChild(critItem);
        });

        charItem.appendChild(charHeader);
        charItem.appendChild(charContent);
        accordion.appendChild(charItem);

        if (col.id === defaultTarget) {
            charItem.classList.add('open');
        }
    });
}

async function runAnalysis(targetCol, criterion = null, targetHistoryContainerId = null) {
    showLoader(true);
    try {
        const reqBody = { target: targetCol };
        if (criterion !== null) {
            reqBody.criterion = criterion;
        }
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqBody)
        });

        if (response.ok) {
            currentPayload = await response.json();
            activeDimensionality = null; // Reset on new analysis
            updateDashboard(currentPayload, targetHistoryContainerId);
        } else {
            console.error("Error fetching analysis", response.status);
        }
    } catch (err) {
        console.error("API POST failed:", err);
    } finally {
        showLoader(false);
    }
}

function showLoader(show) {
    const loader = document.getElementById('loader');
    if (show) loader.classList.add('active');
    else loader.classList.remove('active');
}


function updateDashboard(payload, targetHistoryContainerId) {
    const m = payload.metrics;
    if (activeDimensionality === null) {
        activeDimensionality = Math.min(m.d_star, 3) || 1;
    }

    document.getElementById('val-dstar').innerText = `${m.d_star}D`;
    document.getElementById('val-vir').innerText = `${(m.vir * 100).toFixed(1)}%`;
    document.getElementById('val-pval').innerText = 'p < 0.001';
    document.getElementById('val-loss').innerText = `${(m.l_target * 100).toFixed(1)}%`;
    document.getElementById('totalSamplesVal').innerText = (payload.total_samples || payload.x.length).toLocaleString();

    const pill = document.getElementById('scenarioPill');
    if (pill) {
        pill.className = 'scenario-pill ' + m.scenario;
        document.getElementById('scenarioText').innerText = m.scenario;
    }

    const xaiBanner = document.getElementById('xaiBanner');
    if (xaiBanner) xaiBanner.innerHTML = `💡 <b>XAI Инсайт:</b> ${m.xai_message}`;

    // Selected Axes List
    const axesContainer = document.getElementById('axesListContainer');
    if (axesContainer) {
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
    }

    // Update Color Legend
    const legendTargetName = document.getElementById('legendTargetName');
    if (legendTargetName) legendTargetName.innerText = payload.target_name || 'Целевая переменная';

    const legendItems = document.getElementById('legendItems');
    const uniqueClasses = payload.unique_target_classes || [];
    if (legendItems) {
        legendItems.innerHTML = '';
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
    }

    // Update Bivariate Map Labels (Concentration Map)
    const bivLabelX = document.getElementById('bivLabelX');
    const bivLabelY = document.getElementById('bivLabelY');
    if (bivLabelX && bivLabelY && uniqueClasses.length > 0) {
        bivLabelX.innerText = '100% ' + uniqueClasses[0];
        bivLabelY.innerText = '100% ' + uniqueClasses[uniqueClasses.length - 1];
    }

    // Populate Exact Values for Stroke settings
    const exactSelect = document.getElementById('strokeExactVal');
    if (exactSelect) {
        exactSelect.innerHTML = '<option value="">-- Выберите --</option>';
        // Extract unique purities from the current dimensionality grid
        let dimStr = activeDimensionality.toString();
        let g = payload.grids ? payload.grids[dimStr] : null;
        let purities = g ? g.purity : payload.grid_purity;
        if (purities) {
            let uniqueP = [...new Set(purities)].sort((a, b) => a - b);
            uniqueP.forEach(p => {
                let pct = (p * 100).toFixed(1);
                exactSelect.innerHTML += `<option value="${p}">${pct}%</option>`;
            });
        }
    }

    // Selection History List (Pareto Systems)
    if (targetHistoryContainerId && m.history) {
        const historyContainer = document.getElementById(targetHistoryContainerId);
        if (historyContainer) {
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
                item.onclick = (e) => {
                    e.stopPropagation();
                    Array.from(historyContainer.children).forEach(c => c.classList.remove('active'));
                    item.classList.add('active');
                    setDimensionality(step.step);
                };
                historyContainer.appendChild(item);
            });
        }
    }

    renderPlot(payload);
}

function setDimensionality(d) {
    if (d > 3) d = 3; // Limit visual switching to 3D maximum
    if (activeDimensionality === d) return;
    activeDimensionality = d;

    if (currentPayload) {
        renderPlot(currentPayload);
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

            // 1D Gradient Mapping (Purity Only):
            // Green = Edible (p=0)
            // Red = Poisonous (p=1)
            const r = p * 255;
            const g = (1.0 - p) * 255;
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

    const plotDiv = document.getElementById('plot-container');
    if (plotDiv.removeAllListeners) {
        plotDiv.removeAllListeners('plotly_relayout');
    }
    plotDiv.on('plotly_relayout', function(eventData) {
        if (eventData && eventData['scene.camera']) {
            let cam = eventData['scene.camera'];
            if (cam.eye) {
                let dist = Math.sqrt(cam.eye.x**2 + cam.eye.y**2 + cam.eye.z**2);
                let baseDist = Math.sqrt(1.6**2 + 1.6**2 + 1.3**2);
                let newScale = baseDist / dist;
                newScale = Math.max(0.1, Math.min(newScale, 15));
                
                window._currentCameraScale = window._currentCameraScale || 1.0;
                if (Math.abs(window._currentCameraScale - newScale) > 0.05) {
                    window._currentCameraScale = newScale;
                    applyStroke(window._isStrokeActive || false);
                }
            }
        }
    });

    // Re-apply stroke if it was active
    applyStroke(window._isStrokeActive || false);
}

let currentStrokeMode = 'range';
window._isStrokeActive = false;
window._currentCameraScale = 1.0;

function setStrokeMode(mode) {
    currentStrokeMode = mode;
    const btnRange = document.getElementById('btnStrokeModeRange');
    const btnExact = document.getElementById('btnStrokeModeExact');
    const divRange = document.getElementById('strokeRangeControls');
    const divExact = document.getElementById('strokeExactControls');

    if (mode === 'range') {
        btnRange.className = 'filter-btn run-btn';
        btnExact.className = 'filter-btn add-btn';
        divRange.style.display = 'flex';
        divExact.style.display = 'none';
    } else {
        btnExact.className = 'filter-btn run-btn';
        btnRange.className = 'filter-btn add-btn';
        divExact.style.display = 'flex';
        divRange.style.display = 'none';
    }
}

function applyStroke(enable) {
    window._isStrokeActive = enable;
    if (!currentPayload) return;

    let dimStr = activeDimensionality.toString();
    let g = currentPayload.grids ? currentPayload.grids[dimStr] : null;
    let currentPurity = g ? g.purity : currentPayload.grid_purity;
    let currentOpacity = g ? g.opacity : currentPayload.grid_opacity;

    if (!currentPurity) return;

    const n = currentPurity.length;
    let scale = window._currentCameraScale || 1.0;
    
    let lineColors = new Array(n).fill('rgb(0,0,0)');
    let lineWidths = new Array(n).fill(0);
    let markerSizes = new Array(n);
    
    for (let i = 0; i < n; i++) {
        let baseSize = currentOpacity ? (6 + currentOpacity[i] * 18) : 12;
        markerSizes[i] = baseSize * scale;
    }

    if (enable) {
        const color = document.getElementById('strokeColor').value || '#ffffff';
        const width = parseFloat(document.getElementById('strokeWidth').value) || 2;

        let checkMatch = (p) => false;

        if (currentStrokeMode === 'range') {
            const minP = parseFloat(document.getElementById('strokeMin').value) / 100.0;
            const maxP = parseFloat(document.getElementById('strokeMax').value) / 100.0;
            checkMatch = (p) => (p >= minP - 0.001 && p <= maxP + 0.001);
        } else {
            const exactVal = document.getElementById('strokeExactVal').value;
            if (exactVal !== "") {
                const targetP = parseFloat(exactVal);
                checkMatch = (p) => Math.abs(p - targetP) < 0.001;
            }
        }

        for (let i = 0; i < n; i++) {
            if (checkMatch(currentPurity[i])) {
                lineColors[i] = color;
                lineWidths[i] = width;
                let baseSize = currentOpacity ? (6 + currentOpacity[i] * 18) : 12;
                markerSizes[i] = (baseSize + 6) * scale; // Slightly larger for highlighted points
            }
        }
    }

    // Trace 0 is cell boundaries, Trace 1 is the scatter points
    Plotly.restyle('plot-container', {
        'marker.line.color': [lineColors],
        'marker.line.width': [lineWidths],
        'marker.size': [markerSizes]
    }, 1);
}

function toggleMainAcc(id) {
    const isCatalog = (id === 'catalog');
    const hSearch = document.getElementById('headerSearch');
    const cSearch = document.getElementById('contentSearch');
    const hCat = document.getElementById('headerCatalog');
    const cCat = document.getElementById('contentCatalog');

    if (isCatalog) {
        hSearch.classList.remove('open');
        cSearch.classList.remove('open');
        hCat.classList.add('open');
        cCat.classList.add('open');
    } else {
        hCat.classList.remove('open');
        cCat.classList.remove('open');
        hSearch.classList.add('open');
        cSearch.classList.add('open');
    }
}

function addFilterRow() {
    const container = document.getElementById('filterRowsContainer');
    const row = document.createElement('div');
    row.className = 'filter-row';

    const colSelect = document.createElement('select');
    colSelect.className = 'filter-select';
    let colOptions = '<option value="">-- Характеристика --</option>';
    allColumnsData.forEach(c => {
        colOptions += `<option value="${c.id}">${c.label}</option>`;
    });
    colSelect.innerHTML = colOptions;

    const valSelect = document.createElement('select');
    valSelect.className = 'filter-select';
    valSelect.innerHTML = '<option value="">-- Значение --</option>';

    colSelect.onchange = () => {
        const colId = colSelect.value;
        const col = allColumnsData.find(c => c.id === colId);
        valSelect.innerHTML = '<option value="">-- Значение --</option>';
        if (col) {
            col.criteria.forEach(crit => {
                valSelect.innerHTML += `<option value="${crit.id}">${crit.label}</option>`;
            });
        }
    };

    const removeBtn = document.createElement('button');
    removeBtn.className = 'filter-remove';
    removeBtn.innerHTML = '×';
    removeBtn.title = 'Удалить';
    removeBtn.onclick = () => row.remove();

    row.appendChild(colSelect);
    row.appendChild(valSelect);
    row.appendChild(removeBtn);
    container.appendChild(row);
}

async function runCompositeAnalysis() {
    const container = document.getElementById('filterRowsContainer');
    const rows = container.querySelectorAll('.filter-row');

    const compositeTarget = [];
    rows.forEach(row => {
        const selects = row.querySelectorAll('select');
        const col = selects[0].value;
        const val = selects[1].value;
        if (col && val) {
            compositeTarget.push({ col: col, val: val });
        }
    });

    if (compositeTarget.length === 0) {
        alert("Пожалуйста, добавьте хотя бы одно полное условие (Характеристика + Значение).");
        return;
    }

    showLoader(true);
    try {
        const reqBody = { composite_target: compositeTarget };
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqBody)
        });

        if (response.ok) {
            currentPayload = await response.json();
            activeDimensionality = null;
            if (activeCritItem) {
                activeCritItem.classList.remove('active');
                activeCritItem.classList.remove('open');
                activeCritItem = null;
            }
            updateDashboard(currentPayload, 'compositeHistoryContainer');
        } else {
            console.error("Error fetching composite analysis", response.status);
        }
    } catch (err) {
        console.error("API POST failed:", err);
    } finally {
        showLoader(false);
    }
}

window.addEventListener('DOMContentLoaded', init);
