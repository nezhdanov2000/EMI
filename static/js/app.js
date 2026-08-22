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
            populateBlueFeatureDropdowns(colData.columns);
            renderDiscreteColorMatrix();
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
        if (currentBlueFeature !== null) {
            reqBody.blue_feature = currentBlueFeature;
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


let currentBlueFeature = null;

function populateBlueFeatureDropdowns(cols) {
    const colSelect = document.getElementById('blueFeatureCol');
    if (!colSelect) return;

    cols.forEach(col => {
        const option = document.createElement('option');
        option.value = col.id;
        option.innerText = col.label;
        colSelect.appendChild(option);
    });
}

function onBlueColChange() {
    const colSelect = document.getElementById('blueFeatureCol');
    const valSelect = document.getElementById('blueFeatureVal');
    valSelect.innerHTML = '<option value="">-- Выберите --</option>';

    if (colSelect.value === '') return;

    const colData = allColumnsData.find(c => c.id === colSelect.value);
    if (colData && colData.criteria) {
        colData.criteria.forEach(crit => {
            const option = document.createElement('option');
            option.value = crit.id;
            option.innerText = crit.label;
            valSelect.appendChild(option);
        });
    }
}

const DEFAULT_2D_PALETTE = [
    // Ряд 0 (Низ: 25% – 50%): [0-25% Съед (Ядовитый / Красный), 25-50% (Оранжевый), 50-75% (Салатовый), 75-100% (Съедобный / Зеленый)]
    ['#E82B10', '#EB751A', '#6BBF26', '#52FF33'],
    // Ряд 1 (Середина: 50% – 75%): [Желтый, Коричневый, Темно-зеленый, Белый]
    ['#EBF033', '#661A00', '#2C5A14', '#FFFFFF'],
    // Ряд 2 (Верх: 75% – 100%): [Розовый, Фиолетовый, Синий, Голубой]
    ['#F024EB', '#7E0CF5', '#0018F5', '#52F5FF']
];

// Flat list of all 12 unique palette colors (row0col0, row0col1, ..., row2col3)
// Index 0..3 = Row 0 (25-50%), Index 4..7 = Row 1 (50-75%), Index 8..11 = Row 2 (>75%)
const PALETTE_FLAT = DEFAULT_2D_PALETTE.flat();
const PALETTE_COUNT = PALETTE_FLAT.length; // 12

// Build a Plotly discrete colorscale: array of [normalizedVal, hexColor]
// Each color occupies a band of width 1/12 in the [0, 1] range
function buildDiscreteColorscale() {
    const scale = [];
    for (let i = 0; i < PALETTE_COUNT; i++) {
        const lo = i / PALETTE_COUNT;
        const hi = (i + 1) / PALETTE_COUNT;
        scale.push([lo, PALETTE_FLAT[i]]);
        scale.push([hi, PALETTE_FLAT[i]]);
    }
    return scale;
}
const DISCRETE_COLORSCALE = buildDiscreteColorscale();

function get2DMatrixColorIndex(purity, blueVal, hasBlueFeature = false) {
    // purity: 0.0 = 100% Edible, 1.0 = 100% Poisonous
    // col 0 = Ядовитый (0-25% Edible), col 3 = Съедобный (75-100% Edible)
    const edible_ratio = Math.max(0, Math.min(1, 1.0 - purity));
    let col = Math.min(3, Math.floor(edible_ratio * 4));

    // If blue feature is NOT applied: 1D mode, use Row 0
    if (!hasBlueFeature || blueVal === undefined || blueVal === null) {
        return { colorIndex: 0 * 4 + col, isNoise: false };
    }

    // Noise filtering: < 25%
    if (blueVal < 0.25) {
        return { colorIndex: -1, isNoise: true };
    }

    let row = 0;
    if (blueVal >= 0.75) {
        row = 2;
    } else if (blueVal >= 0.50) {
        row = 1;
    } else {
        row = 0;
    }

    return { colorIndex: row * 4 + col, isNoise: false };
}

function renderDiscreteColorMatrix() {
    const grid = document.getElementById('discreteMatrixGrid');
    if (!grid) return;

    const numRows = 3;
    const numCols = 4;

    grid.style.gridTemplateColumns = `repeat(${numCols}, 1fr)`;
    grid.style.gridTemplateRows = `repeat(${numRows}, 1fr)`;
    grid.innerHTML = '';

    const tooltip = document.getElementById('matrixTooltip');
    const rowRanges = [
        { label: '25%–50%' },
        { label: '50%–75%' },
        { label: '>75%' }
    ];

    const colRanges = [
        { label: '0%–25% (Ядовитый)' },
        { label: '25%–50%' },
        { label: '50%–75%' },
        { label: '75%–100% (Съедобный)' }
    ];

    // Render from Row 2 (Top: >75%) down to Row 0 (Bottom: 25-50%)
    for (let r = numRows - 1; r >= 0; r--) {
        for (let c = 0; c < numCols; c++) {
            const cellColor = DEFAULT_2D_PALETTE[r][c];
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.style.backgroundColor = cellColor;
            cell.style.border = '1px solid rgba(255, 255, 255, 0.15)';
            cell.style.borderRadius = '3px';
            cell.style.cursor = 'pointer';

            const rowInfo = rowRanges[r];
            const colInfo = colRanges[c];

            cell.addEventListener('mouseenter', () => {
                if (tooltip) {
                    tooltip.style.display = 'block';
                    tooltip.innerHTML = `🍄 Съедобность: <b>${colInfo.label}</b><br>🔷 Признак: <b>${rowInfo.label}</b><br><span style="font-size:0.68rem; color:#94a3b8;">Цвет: ${cellColor}</span>`;
                }
            });
            cell.addEventListener('mouseleave', () => {
                if (tooltip) {
                    tooltip.style.display = 'none';
                }
            });

            grid.appendChild(cell);
        }
    }
}

function applyBlueFeature() {
    const colSelect = document.getElementById('blueFeatureCol');
    const valSelect = document.getElementById('blueFeatureVal');

    if (colSelect.value && valSelect.value) {
        currentBlueFeature = { col: colSelect.value, val: valSelect.value };
        const blueLegendSection = document.getElementById('blueLegendSection');
        const blueLegendName = document.getElementById('blueLegendName');
        if (blueLegendSection) blueLegendSection.style.display = 'block';
        if (blueLegendName) {
            const colLabel = colSelect.options[colSelect.selectedIndex].text.split(' ')[0];
            const valLabel = valSelect.options[valSelect.selectedIndex].text;
            blueLegendName.innerText = `${colLabel} = ${valLabel}`;
        }
        renderDiscreteColorMatrix();
    } else {
        currentBlueFeature = null;
        const blueLegendSection = document.getElementById('blueLegendSection');
        if (blueLegendSection) blueLegendSection.style.display = 'none';
    }

    // Rerun analysis with the new blue feature if we have a current target
    if (currentPayload && currentPayload.target_labels) {
        runAnalysis('class', null);
    }
}

function updateBlueThresholds() {
    renderDiscreteColorMatrix();
    if (currentPayload) {
        renderPlot(currentPayload);
        applyStroke(window._isStrokeActive || false);
    }
}

function updateDashboard(payload, targetHistoryContainerId) {
    const m = payload.metrics;
    if (activeDimensionality === null) {
        activeDimensionality = Math.min(m.d_star, 3) || 1;
    }

    document.getElementById('val-dstar').innerText = `${m.d_star}D`;
    document.getElementById('val-vir').innerText = `${(m.vir * 100).toFixed(1)}%`;
    const nmiVal = (m.nmi !== undefined) ? m.nmi : (1.0 - m.l_target);
    const nmiEl = document.getElementById('val-nmi');
    if (nmiEl) nmiEl.innerText = `${(nmiVal * 100).toFixed(1)}%`;
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
                const nmiStepVal = (step.nmi !== undefined) ? step.nmi : (step.step === m.d_star ? (1.0 - m.l_target) : 0);
                const nmiPct = (nmiStepVal * 100).toFixed(1);
                const deltaPct = step.step === 1 ? '' : `(+${(step.delta_mi * 100).toFixed(1)}%)`;

                let altsHtml = '';
                if (step.alternatives && step.alternatives.length > 0) {
                    altsHtml = '<div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 0.75rem; color: var(--text-dim);">';
                    altsHtml += '<div style="margin-bottom: 3px; font-weight: 600;">Альтернативы:</div>';
                    step.alternatives.forEach(a => {
                        const altVir = (a.vir * 100).toFixed(1);
                        const altNmi = a.nmi !== undefined ? `NMI: ${(a.nmi * 100).toFixed(1)}%` : `VIR: ${altVir}%`;
                        const extraVir = a.nmi !== undefined ? ` (VIR: ${altVir}%)` : '';
                        altsHtml += `<div>• ${a.feature} (${altNmi}${extraVir})</div>`;
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
                                        <div class="history-nmi" title="Normalized Mutual Information (NMI): реальная предсказательная сила центров относительно цели">NMI: ${nmiPct}%</div>
                                        <div class="history-vir" title="Visual Information Ratio (VIR): полнота осей относительно всего датасета">VIR: ${virPct}%</div>
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
    let currentBlue = g ? g.blue : payload.grid_blue_concentration;

    // Parse thresholds
    let blueThresholds = [0, 0.25, 0.5, 0.75, 1.0];
    const thresInput = document.getElementById('blueThresholds');
    if (thresInput) {
        let vals = thresInput.value.split(',').map(v => parseFloat(v.trim()) / 100).filter(v => !isNaN(v));
        if (vals.length > 0) {
            blueThresholds = vals.sort((a, b) => a - b);
        }
    }
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

    const hasBlue = (currentBlueFeature !== null && currentBlueFeature.col);
    
    let fx = [];
    let fy = [];
    let fz = [];
    let fColors = [];
    let fSizes = [];
    let fHover = [];
    let fPurity = [];
    let fOpacity = [];

    const totalPts = xCoords ? xCoords.length : 0;

    for (let i = 0; i < totalPts; i++) {
        const p = (currentPurity && currentPurity[i] !== undefined) ? currentPurity[i] : 0.5;
        const op = (currentOpacity && currentOpacity[i] !== undefined) ? currentOpacity[i] : 0.5;
        const b_val = (hasBlue && currentBlue && currentBlue[i] !== undefined) ? currentBlue[i] : null;

        const res = get2DMatrixColorIndex(p, b_val, hasBlue);

        // Completely EXCLUDE noise points (<25%) from the dataset
        if (res.isNoise) {
            continue;
        }

        fx.push(xCoords[i]);
        fy.push(yCoords[i]);
        fz.push(zCoords[i]);

        // Convert palette hex to rgb() string — NO rgba, NO alpha channel anywhere
        const hex = PALETTE_FLAT[res.colorIndex];
        const rr = parseInt(hex.slice(1, 3), 16);
        const gg = parseInt(hex.slice(3, 5), 16);
        const bb = parseInt(hex.slice(5, 7), 16);
        fColors.push(`rgb(${rr}, ${gg}, ${bb})`);

        fPurity.push(p);
        fOpacity.push(op);

        const baseSize = 6 + op * 18.0;
        fSizes.push(baseSize);

        if (currentHover && currentHover[i]) {
            fHover.push(currentHover[i]);
        }
    }

    // Save filtered dataset for applyStroke
    window._lastFilteredData = {
        purity: fPurity,
        opacity: fOpacity,
        baseSizes: fSizes
    };

    const scatterTrace = {
        x: fx,
        y: fy,
        z: fz,
        mode: 'markers',
        marker: {
            size: fSizes,
            color: fColors,
            opacity: 1,
            line: {
                width: 0
            },
            showscale: false
        },
        hovertext: fHover,
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

    // Default camera distance is ~2.608 (sqrt(1.6^2 + 1.6^2 + 1.3^2))
    window._currentCameraScale = 1.0;
    
    // Re-apply stroke if it was active
    applyStroke(window._isStrokeActive || false);

    const plotDiv = document.getElementById('plot-container');
    plotDiv.on('plotly_relayout', function(eventData) {
        let eye = null;
        if (eventData['scene.camera'] && eventData['scene.camera'].eye) {
            eye = eventData['scene.camera'].eye;
        } else if (eventData['scene.camera.eye']) {
            eye = eventData['scene.camera.eye'];
        }

        if (eye) {
            const distance = Math.sqrt(eye.x * eye.x + eye.y * eye.y + eye.z * eye.z);
            let scale = 2.61 / distance;
            scale = Math.max(0.1, Math.min(scale, 10.0)); // Restrict scaling limits
            
            // Only restyle if the scale changed by at least 2% to avoid lag during drag
            const currentScale = window._currentCameraScale || 1.0;
            if (Math.abs(scale - currentScale) > 0.02) {
                window._currentCameraScale = scale;
                applyStroke(window._isStrokeActive || false);
            }
        }
    });
}

let currentStrokeMode = 'range';
window._isStrokeActive = false;

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
    if (!window._lastFilteredData) return;

    const { purity, opacity, baseSizes } = window._lastFilteredData;
    const n = purity.length;
    if (n === 0) return;

    let lineColors = new Array(n).fill('rgb(0,0,0)');
    let lineWidths = new Array(n).fill(0);
    let markerSizes = new Array(n);
    
    const scale = window._currentCameraScale || 1.0;

    for (let i = 0; i < n; i++) {
        markerSizes[i] = baseSizes[i] * scale;
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
            if (checkMatch(purity[i])) {
                lineColors[i] = color;
                lineWidths[i] = width;
                markerSizes[i] = (baseSizes[i] + 4) * scale; // Highlighted size
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
        if (currentBlueFeature !== null) {
            reqBody.blue_feature = currentBlueFeature;
        }
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
