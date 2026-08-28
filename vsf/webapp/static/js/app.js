let currentPayload = null;
let activeDimensionality = null;
let activeCritItem = null;
let allColumnsData = [];
let activeSliceIndex = null;  // Current 4D slice index (null = show all / no 4D)
let currentRenderedDim = null;
let isAnimating = false;
let currentAnalysisContext = { target: 'class', criterion: null, composite_target: null };
let _currentClickedCenterPt = null;

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
    // Dataset-agnostic default target: resolved from /api/columns' declared
    // default_target (set server-side to the actual configured target column,
    // not necessarily "class"), falling back to the first catalog column, and
    // only falling back to the literal "class" if neither is available (e.g.
    // /api/columns itself failed before we learned anything about this dataset).
    let defaultTarget = 'class';
    try {
        const colRes = await fetch('/api/columns');
        if (colRes.ok) {
            const colData = await colRes.json();
            allColumnsData = colData.columns;
            defaultTarget = colData.default_target || (colData.columns[0] && colData.columns[0].id) || 'class';
            populateCatalog(colData.columns, colData.default_target);
            // Add default first filter row
            addFilterRow();
            populateBlueFeatureDropdowns(colData.columns);
        }

        // Fetch top insights catalog (NMI >= 75%)
        const topRes = await fetch('/api/top_columns');
        if (topRes.ok) {
            const topData = await topRes.json();
            populateTopCatalog(topData.columns);
        }

        await runAnalysis(defaultTarget, null);
    } catch (err) {
        console.error("Initialization error:", err);
        await runAnalysis(defaultTarget, null);
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
                    critContent.innerHTML = '<div style="color:var(--text-dim);font-size:0.8rem;padding:4px;">Analyzing Pareto-front...</div>';
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

function populateTopCatalog(cols) {
    const accordion = document.getElementById('topCatalogAccordion');
    if (!accordion) return;
    accordion.innerHTML = '';

    if (!cols || cols.length === 0) {
        accordion.innerHTML = '<div style="color: var(--text-dim); font-size: 0.85rem; padding: 1rem;">No insights found with NMI ≥ 75%</div>';
        return;
    }

    cols.forEach(col => {
        const charItem = document.createElement('div');
        charItem.className = 'char-item';

        const charHeader = document.createElement('div');
        charHeader.className = 'char-header';
        const colMaxNmiPct = (col.max_nmi * 100).toFixed(1);
        charHeader.innerHTML = `
            <span class="char-title">${col.label}</span>
            <div style="display:flex; align-items:center; gap:8px;">
                <span class="top-nmi-badge">${colMaxNmiPct}%</span>
                <span class="char-icon">▶</span>
            </div>
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
            const critNmiPct = (crit.max_nmi * 100).toFixed(1);
            critHeader.innerHTML = `
                <span>${crit.label}</span>
                <span class="top-crit-nmi">${critNmiPct}%</span>
            `;

            const critContent = document.createElement('div');
            critContent.className = 'crit-content';
            const historyListId = `top-history-${col.id}-${crit.id.replace(/[^a-zA-Z0-9]/g, '_')}`;
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
                    critContent.innerHTML = '<div style="color:var(--text-dim);font-size:0.8rem;padding:4px;">Analyzing Pareto-front...</div>';
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
    });
}

async function runAnalysis(targetCol, criterion = null, targetHistoryContainerId = null) {
    currentAnalysisContext = { target: targetCol, criterion: criterion, composite_target: null };
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

// 5-bin Probability Scale
const PROB_COLORS = [
    '#f44336', // Red (0-15%) - Non-target
    '#ff9800', // Orange (15-30%)
    '#57463a', // Dark Brown (30-70%) - Murky Zone
    '#ffeb3b', // Yellow (70-85%)
    '#53ea4c'  // Green (85-100%) - Target
];
const PALETTE_COUNT = PROB_COLORS.length;

function buildDiscreteColorscale() {
    const scale = [];
    for (let i = 0; i < PALETTE_COUNT; i++) {
        const lo = i / PALETTE_COUNT;
        const hi = (i + 1) / PALETTE_COUNT;
        scale.push([lo, PROB_COLORS[i]]);
        scale.push([hi, PROB_COLORS[i]]);
    }
    return scale;
}
const DISCRETE_COLORSCALE = buildDiscreteColorscale();

function getColorIndexForPurity(purity) {
    if (purity < 0.15) return 0;
    if (purity < 0.30) return 1;
    if (purity < 0.70) return 2;
    if (purity < 0.85) return 3;
    return 4;
}

function updateDashboard(payload, targetHistoryContainerId) {
    currentPayload = payload;
    const m = payload.metrics;

    currentRenderedDim = null; // Force full plot re-render with new axis titles
    if (activeDimensionality === null) {
        activeDimensionality = (m && m.d_star) ? m.d_star : 3;
    }

    // Handle 4D slice controller setup
    if (payload.slice_axis) {
        if (activeSliceIndex === null) {
            activeSliceIndex = 0;  // Default to first slice
        }
        renderSliceTabs(payload);
    } else {
        activeSliceIndex = null;
        const sliceCtrl = document.getElementById('slice-controller');
        if (sliceCtrl) sliceCtrl.style.display = 'none';
    }

    document.getElementById('val-pval').innerText = 'p < 0.001';
    document.getElementById('totalSamplesVal').innerText = (payload.total_samples || (payload.x ? payload.x.length : 0)).toLocaleString();

    const xaiBanner = document.getElementById('xaiBanner');
    if (xaiBanner) xaiBanner.innerHTML = `💡 <b>XAI Insight:</b> ${m.xai_message}`;

    // Render Dimension Switcher toggle buttons
    renderDimensionButtons(payload);

    // Update Color Legend
    const legendTargetName = document.getElementById('legendTargetName');
    if (legendTargetName) legendTargetName.innerText = payload.target_name || 'Target Variable';

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
        exactSelect.innerHTML = '<option value="">-- Select --</option>';
        let dimStr = (activeDimensionality || 3).toString();
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
                item.dataset.step = step.step;

                const virPct = (step.vir * 100).toFixed(1);
                const nmiStepVal = (step.nmi !== undefined) ? step.nmi : (step.step === m.d_star ? (1.0 - m.l_target) : 0);
                const nmiPct = (nmiStepVal * 100).toFixed(1);
                const deltaPct = step.step === 1 ? '' : `(+${(step.delta_mi * 100).toFixed(1)}%)`;

                let altsHtml = '';
                if (step.alternatives && step.alternatives.length > 0) {
                    altsHtml = '<div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 0.75rem; color: var(--text-dim);">';
                    altsHtml += '<div style="margin-bottom: 3px; font-weight: 600;">Alternatives:</div>';
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
                                        <div class="history-step">${step.step}D System</div>
                                        <div class="history-feature">${step.feature}</div>
                                    </div>
                                    <div class="history-stats">
                                        <div class="history-nmi" title="Normalized Mutual Information (NMI): predictive power of centers relative to target">NMI: ${nmiPct}%</div>
                                        <div class="history-vir" title="Visual Information Ratio (VIR): axis coverage relative to dataset">VIR: ${virPct}%</div>
                                        <div class="history-delta">${deltaPct}</div>
                                    </div>
                                </div>
                                ${altsHtml}
                            </div>
                        `;
                item.onclick = (e) => {
                    e.stopPropagation();
                    setDimensionality(step.step);
                };
                historyContainer.appendChild(item);
            });
        }
    }

    setDimensionality(activeDimensionality);
}

function renderDimensionButtons(payload) {
    const container = document.getElementById('dimButtonsGroup');
    if (!container) return;
    container.innerHTML = '';

    const availableDims = [];
    if (payload.metrics && payload.metrics.history && payload.metrics.history.length > 0) {
        payload.metrics.history.forEach(h => {
            if (!availableDims.includes(h.step)) availableDims.push(h.step);
        });
    } else {
        const maxD = (payload.metrics && payload.metrics.d_star) ? payload.metrics.d_star : 3;
        for (let d = 1; d <= Math.max(maxD, 1); d++) availableDims.push(d);
    }

    availableDims.sort((a, b) => a - b).forEach(d => {
        const btn = document.createElement('button');
        btn.className = 'toggle-btn dim-btn' + (d === activeDimensionality ? ' active' : '');
        btn.dataset.dim = d;
        btn.innerText = `${d}D`;
        btn.title = `Switch dimensionality to ${d}D`;
        btn.onclick = () => setDimensionality(d);
        container.appendChild(btn);
    });
}

function updateHUDForDimension(d) {
    if (!currentPayload || !currentPayload.metrics) return;
    const m = currentPayload.metrics;

    let stepData = null;
    if (m.history && m.history.length > 0) {
        stepData = m.history.find(h => h.step === d);
    }

    let nmiVal = stepData && stepData.nmi !== undefined ? stepData.nmi : (d === m.d_star ? (1.0 - m.l_target) : 0.0);
    let virVal = stepData && stepData.vir !== undefined ? stepData.vir : (d === m.d_star ? m.vir : 1.0);
    let lossVal = Math.max(0.0, 1.0 - nmiVal);

    const nmiEl = document.getElementById('val-nmi');
    if (nmiEl) nmiEl.innerText = `${(nmiVal * 100).toFixed(1)}%`;

    const lossEl = document.getElementById('val-loss');
    if (lossEl) lossEl.innerText = `${(lossVal * 100).toFixed(1)}%`;

    const virEl = document.getElementById('val-vir');
    if (virEl) virEl.innerText = `${(virVal * 100).toFixed(1)}%`;

    const dstarEl = document.getElementById('val-dstar');
    if (dstarEl) {
        if (d === m.d_star) {
            dstarEl.innerText = `${d}D`;
        } else {
            dstarEl.innerText = `${d}D (optimal: ${m.d_star}D)`;
        }
    }

    const pill = document.getElementById('scenarioPill');
    const scenarioText = document.getElementById('scenarioText');
    if (pill && scenarioText) {
        let scenarioClass = d <= 3 ? 'SCENARIO_A' : 'SCENARIO_B';
        let scenarioLabel = d <= 3 ? 'Scenario A: Minimalist' : 'Scenario B: Full Load';
        if (d === m.d_star && m.scenario) {
            scenarioClass = m.scenario;
            scenarioLabel = m.scenario;
        }
        pill.className = 'scenario-pill ' + scenarioClass;
        scenarioText.innerText = scenarioLabel;
    }

    updateAxesList(d);
}

function updateAxesList(d) {
    const axesContainer = document.getElementById('axesListContainer');
    if (!axesContainer || !currentPayload || !currentPayload.selected_features) return;
    axesContainer.innerHTML = '';
    const labels = ['X-Axis (1D)', 'Y-Axis (2D)', 'Z-Axis (3D)', '4D Slice (Tabs)', 'Channel 5', 'Channel 6', 'Channel 7'];
    const maxFeatures = Math.min(d, currentPayload.selected_features.length);
    for (let idx = 0; idx < maxFeatures; idx++) {
        const feat = currentPayload.selected_features[idx];
        const item = document.createElement('div');
        item.className = 'axis-pill';
        item.innerHTML = `
            <span style="font-weight: 500;">${feat}</span>
            <span class="axis-badge">${labels[idx] || 'Channel ' + (idx + 1)}</span>
        `;
        axesContainer.appendChild(item);
    }
}

function setDimensionality(d) {
    if (d === currentRenderedDim && d === activeDimensionality) return;
    if (isAnimating) return;

    const fromDim = currentRenderedDim;
    activeDimensionality = d;

    document.querySelectorAll('#dimButtonsGroup .dim-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.dim) === d);
    });

    document.querySelectorAll('.history-item').forEach(item => {
        const stepNum = parseInt(item.dataset.step);
        if (stepNum === d) {
            item.classList.add('active');
        } else {
            const stepEl = item.querySelector('.history-step');
            if (stepEl && stepEl.innerText.startsWith(`${d}D`)) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        }
    });

    updateHUDForDimension(d);

    const sliceCtrl = document.getElementById('slice-controller');
    if (sliceCtrl) {
        if (d >= 4 && currentPayload && currentPayload.slice_axis) {
            sliceCtrl.style.display = 'flex';
        } else {
            sliceCtrl.style.display = 'none';
        }
    }

    if (currentPayload) {
        if (fromDim === null) {
            renderPlot(currentPayload);
        } else {
            transitionDimensionality(fromDim, d);
        }
    }
}

function renderSliceTabs(payload) {
    const sliceCtrl = document.getElementById('slice-controller');
    const sliceName = document.getElementById('slice-axis-name');
    const sliceTabsContainer = document.getElementById('slice-tabs');
    
    if (!sliceCtrl || !payload.slice_axis) return;
    
    sliceName.textContent = payload.slice_axis.name;
    
    let tabsHtml = '';
    // "All" tab
    const allActiveClass = (activeSliceIndex === null) ? ' active' : '';
    tabsHtml += `<button class="slice-tab${allActiveClass}" onclick="selectSlice(null)" data-slice="all">All<span class="slice-count">(${payload.total_samples})</span></button>`;
    
    // Per-category tabs
    payload.slice_axis.ticks.forEach((label, idx) => {
        const count = payload.slice_axis.counts[idx];
        const activeClass = (idx === activeSliceIndex) ? ' active' : '';
        tabsHtml += `<button class="slice-tab${activeClass}" onclick="selectSlice(${idx})" data-slice="${idx}">${label}<span class="slice-count">(n=${count})</span></button>`;
    });
    
    sliceTabsContainer.innerHTML = tabsHtml;
}

function selectSlice(idx) {
    window._lastActiveSliceIndex = activeSliceIndex;
    activeSliceIndex = idx;
    
    const tabs = document.querySelectorAll('#slice-tabs .slice-tab');
    tabs.forEach(tab => {
        const tabSlice = tab.getAttribute('data-slice');
        if (idx === null && tabSlice === 'all') {
            tab.classList.add('active');
        } else if (idx !== null && tabSlice === String(idx)) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    if (activeDimensionality !== 4) {
        activeDimensionality = 4;
        document.querySelectorAll('#dimButtonsGroup .dim-btn').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.dim) === 4);
        });
        updateHUDForDimension(4);
    }
    
    if (currentPayload) {
        if (currentRenderedDim === null) {
            renderPlot(currentPayload);
        } else {
            transitionDimensionality(currentRenderedDim, 4, window._lastActiveSliceIndex);
        }
    }
}
function buildPlotData(payload, dim, sliceIndex) {
    let dimStr = dim.toString();
    
    let gridKey = dimStr;
    if (dim >= 4 && payload.slice_axis) {
        if (sliceIndex !== null) {
            gridKey = `4_${sliceIndex}`;
        } else {
            gridKey = `4_all`;
        }
    }

    let g = null;
    if (payload.grids) {
        g = payload.grids[gridKey] || payload.grids[dimStr] || payload.grids["3"];
    }

    let xCoords = g ? g.x : payload.grid_x;
    let yCoords = g ? g.y : payload.grid_y;
    let zCoords = g ? g.z : payload.grid_z;

    let currentPurity = g ? g.purity : payload.grid_purity;
    let currentSizes = g ? g.sizes : payload.grid_sizes;
    let currentHover = g ? g.hover_text : payload.grid_hover_text;
    let currentCustomdata = g ? g.customdata : payload.grid_customdata;

    const xLen = payload.axis_ticks ? payload.axis_ticks.x.vals.length : 4;
    const yLen = payload.axis_ticks ? payload.axis_ticks.y.vals.length : 4;
    const zLen = payload.axis_ticks ? payload.axis_ticks.z.vals.length : 4;

    const xMin = -0.5, xMax = xLen - 0.5;
    const yMin = -0.5, yMax = yLen - 0.5;
    const zMin = -0.5, zMax = zLen - 0.5;

    const glX = [], glY = [], glZ = [];

    // ALWAYS draw the full 3D room for consistent spatial metaphor
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
        x: glX, y: glY, z: glZ,
        mode: 'lines',
        line: { color: 'rgba(255, 255, 255, 0.12)', width: 1.0 },
        hoverinfo: 'skip', type: 'scatter3d', name: 'Cell Grid'
    };

    let fx = [], fy = [], fz = [], fColors = [], fSizes = [], fHover = [], fPurity = [], fOpacity = [], fCustomdata = [];

    const totalPts = xCoords ? xCoords.length : 0;
    for (let i = 0; i < totalPts; i++) {
        const p = (currentPurity && currentPurity[i] !== undefined) ? currentPurity[i] : 0.5;
        const n_c = (currentSizes && currentSizes[i] !== undefined) ? currentSizes[i] : 1;

        const colorIndex = getColorIndexForPurity(p);

        fx.push(xCoords[i]);
        fy.push(yCoords[i]);
        fz.push(zCoords[i]);

        const hex = PROB_COLORS[colorIndex];
        const rr = parseInt(hex.slice(1, 3), 16);
        const gg = parseInt(hex.slice(3, 5), 16);
        const bb = parseInt(hex.slice(5, 7), 16);
        fColors.push(`rgb(${rr}, ${gg}, ${bb})`);

        fPurity.push(p);

        // Area Scaling (Q1 Standard): Diameter ~ sqrt(N)
        // Set maximum diameter so the 1D cluster perfectly touches cell bounds.
        // We empirically use 55 as the magic constant for Plotly 3D scatter
        const maxN = payload.global_max_n || 1;
        const baseSize = 55 * Math.sqrt(n_c / maxN);
        fSizes.push(baseSize);

        fHover.push((currentHover && currentHover[i]) ? currentHover[i] : '');
        fCustomdata.push((currentCustomdata && currentCustomdata[i]) ? currentCustomdata[i] : null);
    }

    const scatterTrace = {
        x: fx, y: fy, z: fz,
        mode: 'markers',
        marker: { size: fSizes, color: fColors, opacity: 1, line: { width: 0 }, showscale: false },
        hovertext: fHover, hoverinfo: 'text', customdata: fCustomdata, type: 'scatter3d', name: 'Data'
    };

    let cameraConfig = undefined;
    if (currentRenderedDim === null) {
        cameraConfig = { eye: { x: 1.6, y: 1.6, z: 1.3 } };
    }

    const layout = {
        paper_bgcolor: '#070a13', plot_bgcolor: '#070a13', showlegend: false,
        scene: {
            aspectmode: 'cube',
            xaxis: {
                title: { text: payload.axis_names.x, font: { color: '#c084fc', size: 13 } },
                tickvals: payload.axis_ticks ? payload.axis_ticks.x.vals : undefined,
                ticktext: payload.axis_ticks ? payload.axis_ticks.x.text : undefined,
                range: [xMin, xMax], tickfont: { color: '#e2e8f0', size: 11 },
                backgroundcolor: '#090d1a', showgrid: false, zeroline: false, showspikes: false
            },
            yaxis: {
                title: { text: payload.axis_names.y, font: { color: '#c084fc', size: 13 } },
                tickvals: payload.axis_ticks ? payload.axis_ticks.y.vals : undefined,
                ticktext: payload.axis_ticks ? payload.axis_ticks.y.text : undefined,
                range: [yMin, yMax], showticklabels: true,
                tickfont: { color: '#e2e8f0', size: 11 }, backgroundcolor: '#090d1a', showgrid: false, zeroline: false, showspikes: false
            },
            zaxis: {
                title: { text: payload.axis_names.z, font: { color: '#c084fc', size: 13 } },
                tickvals: payload.axis_ticks ? payload.axis_ticks.z.vals : undefined,
                ticktext: payload.axis_ticks ? payload.axis_ticks.z.text : undefined,
                range: [zMin, zMax], showticklabels: true,
                tickfont: { color: '#e2e8f0', size: 11 }, backgroundcolor: '#090d1a', showgrid: false, zeroline: false, showspikes: false
            }
        },
        margin: { l: 0, r: 0, b: 0, t: 0 }, font: { family: 'Inter', color: '#94a3b8' },
        uirevision: 'true'
    };

    if (cameraConfig) {
        layout.scene.camera = cameraConfig;
    }
    
    return { traces: [cellBoundaryTrace, scatterTrace], layout: layout, fPurity, fOpacity, fSizes };
}

function renderPlot(payload) {
    const dim = activeDimensionality || 3;
    const data = buildPlotData(payload, dim, activeSliceIndex);
    
    window._lastFilteredData = {
        purity: data.fPurity,
        opacity: data.fOpacity,
        baseSizes: data.fSizes
    };

    const bivLegend = document.getElementById('bivariateLegend');
    if (bivLegend) bivLegend.classList.add('active');

    Plotly.newPlot('plot-container', data.traces, data.layout, { responsive: true, displayModeBar: false });
    
    currentRenderedDim = dim;

    // Default camera distance is ~2.608 (sqrt(1.6^2 + 1.6^2 + 1.3^2))
    window._currentCameraScale = 1.0;

    // Re-apply stroke if it was active
    applyStroke(window._isStrokeActive || false);

    const plotDiv = document.getElementById('plot-container');
    plotDiv.on('plotly_relayout', function (eventData) {
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

    plotDiv.on('plotly_click', async function (data) {
        try {
            if (!data.points || data.points.length === 0) return;
            
            // In Plotly, the scatter trace is at curveNumber 1 or named 'Data'
            let pt = data.points.find(p => p.curveNumber === 1);
            if (!pt) {
                pt = data.points.find(p => {
                    const trace = plotDiv.data && plotDiv.data[p.curveNumber];
                    return trace && trace.name === 'Data';
                });
            }
            if (!pt) pt = data.points[0];
            
            const trace = (plotDiv.data && plotDiv.data[pt.curveNumber]) ? plotDiv.data[pt.curveNumber] : null;
            const cdata = pt.customdata || (trace && trace.customdata ? trace.customdata[pt.pointNumber] : null);
            
            if (!cdata || !cdata.coords) {
                console.warn("Click on non-data or missing customdata:", pt);
                return;
            }
            
            // Only trigger for dirty centers (e.g. 0.3 <= purity <= 0.7)
            if (cdata.pur < 0.3 || cdata.pur > 0.7) {
                console.log("Center purity is", cdata.pur, "- skipping XAI panel");
                return;
            }

            _currentClickedCenterPt = {
                x: pt.x,
                y: pt.y,
                z: pt.z,
                cdata: cdata
            };

            const i_z_x_f = (currentPayload && currentPayload.metrics) ? (currentPayload.metrics.nmi || 1.0) : 1.0;
            
            showLoader(true);
            try {
                const reqBody = {
                    target: currentAnalysisContext.target,
                    criterion: currentAnalysisContext.criterion,
                    composite_target: currentAnalysisContext.composite_target,
                    center_coords: cdata.coords,
                    i_z_x_f: i_z_x_f
                };

                const res = await fetch('/api/mine_center', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(reqBody)
                });
                if (res.ok) {
                    const results = await res.json();
                    renderXaiPanel(cdata, results.results);
                }
            } catch (e) {
                console.error("Error mining center:", e);
            } finally {
                showLoader(false);
            }
        } catch (eOuter) {
            console.error("Crash in click handler: ", eOuter);
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

function flattenCoordinates(x, y, z, targetDim) {
    let nx = [...x];
    let ny = [...y];
    let nz = [...z];
    if (targetDim <= 2) {
        for (let i = 0; i < nz.length; i++) nz[i] = -0.5;
    }
    if (targetDim <= 1) {
        for (let i = 0; i < ny.length; i++) ny[i] = -0.5;
    }
    return { x: nx, y: ny, z: nz };
}

function parseRGBString(c) {
    const m = c.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if(m) return [parseInt(m[1]), parseInt(m[2]), parseInt(m[3])];
    return [0,0,0];
}

function animateScatter3d(startX, startY, startZ, startSizes, startColors, endX, endY, endZ, endSizes, endColors, duration, onComplete) {
    const startTime = performance.now();
    const plotDiv = document.getElementById('plot-container');
    
    function easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
    
    const startRGB = startColors.map(parseRGBString);
    const endRGB = endColors.map(parseRGBString);
    const len = startX.length;
    
    function update(time) {
        let elapsed = time - startTime;
        let progress = Math.min(elapsed / duration, 1.0);
        let eased = easeInOutCubic(progress);
        
        let curX = [], curY = [], curZ = [], curSizes = [], curColors = [];
        for (let i = 0; i < len; i++) {
            curX.push(startX[i] + (endX[i] - startX[i]) * eased);
            curY.push(startY[i] + (endY[i] - startY[i]) * eased);
            curZ.push(startZ[i] + (endZ[i] - startZ[i]) * eased);
            curSizes.push(startSizes[i] + (endSizes[i] - startSizes[i]) * eased);
            
            const r = Math.round(startRGB[i][0] + (endRGB[i][0] - startRGB[i][0]) * eased);
            const g = Math.round(startRGB[i][1] + (endRGB[i][1] - startRGB[i][1]) * eased);
            const b = Math.round(startRGB[i][2] + (endRGB[i][2] - startRGB[i][2]) * eased);
            curColors.push(`rgb(${r}, ${g}, ${b})`);
        }
        
        Plotly.restyle(plotDiv, {
            'x': [curX], 'y': [curY], 'z': [curZ],
            'marker.size': [curSizes], 'marker.color': [curColors]
        }, 1);
        
        if (progress < 1.0) {
            requestAnimationFrame(update);
        } else {
            if (onComplete) onComplete();
        }
    }
    requestAnimationFrame(update);
}

function getSliceMap(payload) {
    const map = {};
    if (!payload || !payload.grids || !payload.slice_axis || !payload.slice_axis.ticks) return map;
    const count = payload.slice_axis.ticks.length;
    for (let i = 0; i < count; i++) {
        const grid = payload.grids[`4_${i}`];
        if (grid) {
            for (let j = 0; j < grid.x.length; j++) {
                const key = `${grid.x[j]}_${grid.y[j]}_${grid.z[j]}`;
                if (map[key] === undefined) {
                    map[key] = i; // Store first slice where it appears
                }
            }
        }
    }
    return map;
}

function buildAnimationArrays(startData, endData, payload, refSlice) {
    const startTrace = startData.traces[1];
    const endTrace = endData.traces[1];
    const sliceMap = getSliceMap(payload);
    
    const startX = [], startY = [], startZ = [], startSizes = [], startColors = [], startHover = [];
    const endX = [], endY = [], endZ = [], endSizes = [], endColors = [], endHover = [];
    
    const startDict = {};
    for (let i = 0; i < startTrace.x.length; i++) {
        const key = `${startTrace.x[i]}_${startTrace.y[i]}_${startTrace.z[i]}`;
        startDict[key] = i;
    }
    
    const endDict = {};
    for (let i = 0; i < endTrace.x.length; i++) {
        const key = `${endTrace.x[i]}_${endTrace.y[i]}_${endTrace.z[i]}`;
        endDict[key] = i;
    }
    
    const unionKeys = new Set([...Object.keys(startDict), ...Object.keys(endDict)]);
    if (refSlice === undefined || refSlice === null) refSlice = 0;
    
    unionKeys.forEach(key => {
        const sIdx = startDict[key];
        const eIdx = endDict[key];
        let pointSlice = sliceMap[key];
        if (pointSlice === undefined) pointSlice = 0;
        
        let xOffset = 0;
        if (pointSlice < refSlice) xOffset = -25;
        else if (pointSlice > refSlice) xOffset = 25;
        else xOffset = (Math.random() > 0.5 ? 25 : -25);
        
        let cx, cy, cz;
        if (sIdx !== undefined && eIdx !== undefined) {
            cx = startTrace.x[sIdx]; cy = startTrace.y[sIdx]; cz = startTrace.z[sIdx];
            startX.push(cx); startY.push(cy); startZ.push(cz);
            startSizes.push(startTrace.marker.size[sIdx]);
            startColors.push(startTrace.marker.color[sIdx]);
            startHover.push(startTrace.hovertext[sIdx]);
            
            endX.push(endTrace.x[eIdx]); endY.push(endTrace.y[eIdx]); endZ.push(endTrace.z[eIdx]);
            endSizes.push(endTrace.marker.size[eIdx]);
            endColors.push(endTrace.marker.color[eIdx]);
            endHover.push(endTrace.hovertext[eIdx]);
        } else if (sIdx !== undefined) {
            cx = startTrace.x[sIdx]; cy = startTrace.y[sIdx]; cz = startTrace.z[sIdx];
            startX.push(cx); startY.push(cy); startZ.push(cz);
            startSizes.push(startTrace.marker.size[sIdx]);
            startColors.push(startTrace.marker.color[sIdx]);
            startHover.push(startTrace.hovertext[sIdx]);
            
            endX.push(cx + xOffset); endY.push(cy); endZ.push(cz);
            endSizes.push(0.1);
            endColors.push(startTrace.marker.color[sIdx]);
            endHover.push(startTrace.hovertext[sIdx]);
        } else if (eIdx !== undefined) {
            cx = endTrace.x[eIdx]; cy = endTrace.y[eIdx]; cz = endTrace.z[eIdx];
            startX.push(cx + xOffset); startY.push(cy); startZ.push(cz);
            startSizes.push(0.1);
            startColors.push(endTrace.marker.color[eIdx]);
            startHover.push(endTrace.hovertext[eIdx]);
            
            endX.push(cx); endY.push(cy); endZ.push(cz);
            endSizes.push(endTrace.marker.size[eIdx]);
            endColors.push(endTrace.marker.color[eIdx]);
            endHover.push(endTrace.hovertext[eIdx]);
        }
    });
    
    return { startX, startY, startZ, startSizes, startColors, startHover, endX, endY, endZ, endSizes, endColors, endHover };
}

function transitionDimensionality(fromDim, toDim, oldSliceIndex = null) {
    if (fromDim === toDim && toDim !== 4) return;
    if (fromDim === 4 && toDim === 4 && oldSliceIndex === activeSliceIndex) return;
    
    isAnimating = true;
    const wasStrokeActive = window._isStrokeActive;
    if (wasStrokeActive) applyStroke(false);
    
    const duration = 600;
    
    if (fromDim === 4 || toDim === 4) {
        // Film Strip / 4D Transition
        let refSlice = toDim === 4 ? activeSliceIndex : oldSliceIndex;
        if (refSlice === undefined || refSlice === null) refSlice = 0;
        
        let actualFromSlice = fromDim === 4 ? (toDim === 4 ? oldSliceIndex : activeSliceIndex) : null;
        const startData = buildPlotData(currentPayload, fromDim, actualFromSlice);
        const targetData = buildPlotData(currentPayload, toDim, activeSliceIndex);
        
        const anim = buildAnimationArrays(startData, targetData, currentPayload, refSlice);
        
        window._lastFilteredData = {
            purity: targetData.fPurity, opacity: targetData.fOpacity, baseSizes: targetData.fSizes
        };
        
        Plotly.restyle('plot-container', {
            'x': [anim.startX], 'y': [anim.startY], 'z': [anim.startZ],
            'marker.size': [anim.startSizes], 'marker.color': [anim.startColors], 'hovertext': [anim.startHover],
            'customdata': [targetData.traces[1].customdata]
        }, 1).then(() => {
            currentRenderedDim = toDim;
            requestAnimationFrame(() => {
                animateScatter3d(
                    anim.startX, anim.startY, anim.startZ, anim.startSizes, anim.startColors,
                    anim.endX, anim.endY, anim.endZ, anim.endSizes, anim.endColors,
                    duration, () => {
                        isAnimating = false;
                        if (wasStrokeActive) applyStroke(true);
                        Plotly.restyle('plot-container', {
                            'x': [targetData.traces[1].x], 'y': [targetData.traces[1].y], 'z': [targetData.traces[1].z],
                            'marker.size': [targetData.traces[1].marker.size], 'marker.color': [targetData.traces[1].marker.color], 'hovertext': [targetData.traces[1].hovertext],
                            'customdata': [targetData.traces[1].customdata]
                        }, 1);
                    }
                );
            });
        });
    } else if (fromDim < toDim) {
        // SPLIT (1D/2D -> 3D)
        const targetData = buildPlotData(currentPayload, toDim, activeSliceIndex);
        const traceToAnimate = targetData.traces[1];
        const flatStart = flattenCoordinates(traceToAnimate.x, traceToAnimate.y, traceToAnimate.z, fromDim);
        
        const origX = traceToAnimate.x, origY = traceToAnimate.y, origZ = traceToAnimate.z;
        const origSizes = traceToAnimate.marker.size, origColors = traceToAnimate.marker.color, origHover = traceToAnimate.hovertext;
        
        window._lastFilteredData = { purity: targetData.fPurity, opacity: targetData.fOpacity, baseSizes: targetData.fSizes };
        
        Plotly.restyle('plot-container', {
            'x': [flatStart.x], 'y': [flatStart.y], 'z': [flatStart.z],
            'marker.size': [origSizes], 'marker.color': [origColors], 'hovertext': [origHover],
            'customdata': [targetData.traces[1].customdata]
        }, 1).then(() => {
            currentRenderedDim = toDim;
            requestAnimationFrame(() => {
                animateScatter3d(
                    flatStart.x, flatStart.y, flatStart.z, origSizes, origColors,
                    origX, origY, origZ, origSizes, origColors,
                    duration, () => {
                        isAnimating = false;
                        if (wasStrokeActive) applyStroke(true);
                        Plotly.restyle('plot-container', {
                            'customdata': [targetData.traces[1].customdata]
                        }, 1);
                    }
                );
            });
        });
    } else {
        // COLLAPSE (3D -> 1D/2D)
        const startData = buildPlotData(currentPayload, fromDim, activeSliceIndex);
        const traceToAnimate = startData.traces[1];
        const flatEnd = flattenCoordinates(traceToAnimate.x, traceToAnimate.y, traceToAnimate.z, toDim);
        const targetData = buildPlotData(currentPayload, toDim, activeSliceIndex);
        
        const origX = traceToAnimate.x, origY = traceToAnimate.y, origZ = traceToAnimate.z;
        const origSizes = traceToAnimate.marker.size, origColors = traceToAnimate.marker.color;
        
        const targetSizes = targetData.traces[1].marker.size;
        const targetColors = targetData.traces[1].marker.color;
        const targetHover = targetData.traces[1].hovertext;
        
        animateScatter3d(
            origX, origY, origZ, origSizes, origColors,
            flatEnd.x, flatEnd.y, flatEnd.z, origSizes, origColors,
            duration, () => {
                window._lastFilteredData = { purity: targetData.fPurity, opacity: targetData.fOpacity, baseSizes: targetData.fSizes };
                Plotly.restyle('plot-container', {
                    'x': [targetData.traces[1].x], 'y': [targetData.traces[1].y], 'z': [targetData.traces[1].z],
                    'marker.size': [targetSizes], 'marker.color': [targetColors], 'hovertext': [targetHover],
                    'customdata': [targetData.traces[1].customdata]
                }, 1).then(() => {
                    currentRenderedDim = toDim;
                    isAnimating = false;
                    if (wasStrokeActive) applyStroke(true);
                });
            }
        );
    }
}

function toggleMainAcc(id) {
    let headerId = 'headerCatalog';
    let contentId = 'contentCatalog';
    if (id === 'search') {
        headerId = 'headerSearch';
        contentId = 'contentSearch';
    } else if (id === 'top') {
        headerId = 'headerTop';
        contentId = 'contentTop';
    }
    const header = document.getElementById(headerId);
    const content = document.getElementById(contentId);
    if (!header || !content) return;
    
    if (header.classList.contains('open')) {
        header.classList.remove('open');
        content.classList.remove('open');
    } else {
        header.classList.add('open');
        content.classList.add('open');
    }
}

function addFilterRow() {
    const container = document.getElementById('filterRowsContainer');
    const row = document.createElement('div');
    row.className = 'filter-row';

    const colSelect = document.createElement('select');
    colSelect.className = 'filter-select';
    let colOptions = '<option value="">-- Feature --</option>';
    allColumnsData.forEach(c => {
        colOptions += `<option value="${c.id}">${c.label}</option>`;
    });
    colSelect.innerHTML = colOptions;

    const valSelect = document.createElement('select');
    valSelect.className = 'filter-select';
    valSelect.innerHTML = '<option value="">-- Value --</option>';

    colSelect.onchange = () => {
        const colId = colSelect.value;
        const col = allColumnsData.find(c => c.id === colId);
        valSelect.innerHTML = '<option value="">-- Value --</option>';
        if (col) {
            col.criteria.forEach(crit => {
                valSelect.innerHTML += `<option value="${crit.id}">${crit.label}</option>`;
            });
        }
    };

    const removeBtn = document.createElement('button');
    removeBtn.className = 'filter-remove';
    removeBtn.innerHTML = '×';
    removeBtn.title = 'Remove';
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
        alert("Please add at least one complete condition (Feature + Value).");
        return;
    }

    currentAnalysisContext = { target: null, criterion: null, composite_target: compositeTarget };

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

// --- XAI Panel Logic & Mitosis Canvas Engine ---

const MitosisEngine = {
    canvas: null,
    ctx: null,
    animId: null,
    t: 0,
    targetT: 0,
    startT: 0,
    startTime: 0,
    duration: 650,
    centerData: null,
    activeRule: null,

    init(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = (rect.width || 440) * dpr;
        this.canvas.height = 150 * dpr;
        this.ctx.scale(dpr, dpr);
    },

    setCenter(cdata) {
        this.centerData = cdata;
        this.activeRule = null;
        this.t = 0;
        this.targetT = 0;
        this.startT = 0;
        if (this.animId) cancelAnimationFrame(this.animId);
        this.draw(0);
    },

    animateToRule(rule) {
        this.activeRule = rule;
        this.startT = this.t;
        this.targetT = 1.0;
        this.startTime = performance.now();
        if (this.animId) cancelAnimationFrame(this.animId);
        this.loop();
    },

    resetToUnified() {
        this.activeRule = null;
        this.startT = this.t;
        this.targetT = 0.0;
        this.startTime = performance.now();
        if (this.animId) cancelAnimationFrame(this.animId);
        this.loop();
    },

    easeInOutCubic(x) {
        return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
    },

    loop() {
        const now = performance.now();
        const elapsed = now - this.startTime;
        const progress = Math.min(1.0, elapsed / this.duration);
        const ease = this.easeInOutCubic(progress);
        
        this.t = this.startT + (this.targetT - this.startT) * ease;
        this.draw(this.t);

        if (progress < 1.0) {
            this.animId = requestAnimationFrame(() => this.loop());
        }
    },

    draw(t) {
        if (!this.ctx || !this.centerData) return;
        const ctx = this.ctx;
        const dpr = window.devicePixelRatio || 1;
        const w = this.canvas.width / dpr;
        const h = this.canvas.height / dpr;
        
        ctx.clearRect(0, 0, w, h);
        
        const centerX = w / 2;
        const centerY = h / 2 - 8;
        
        // Base sphere parameters
        const baseN = this.centerData.N;
        const basePur = this.centerData.pur;
        const baseColorIndex = getColorIndexForPurity(basePur);
        const baseColor = PROB_COLORS[baseColorIndex];
        const baseRadius = 36;

        if (t <= 0.01 || !this.activeRule) {
            // Single unified dirty sphere
            this.draw3DSphere(ctx, centerX, centerY, baseRadius, baseColor, `${(basePur*100).toFixed(1)}%`, `Initial Cluster (N = ${baseN} pcs.)`);
            return;
        }

        const r = this.activeRule;
        const fracPos = Math.max(0.15, r.n_pos / baseN);
        const fracNeg = Math.max(0.15, r.n_neg / baseN);
        
        // Target radii proportional to sqrt(N)
        const radPos = Math.max(18, Math.min(38, baseRadius * Math.sqrt(fracPos) * 1.3));
        const radNeg = Math.max(18, Math.min(38, baseRadius * Math.sqrt(fracNeg) * 1.3));
        
        const colPos = PROB_COLORS[getColorIndexForPurity(r.purity_pos)];
        const colNeg = PROB_COLORS[getColorIndexForPurity(r.purity_neg)];

        // Separation distance
        const maxOffset = 110;
        const currentOffset = maxOffset * t;
        
        const xPos = centerX - currentOffset;
        const xNeg = centerX + currentOffset;
        
        // Mitosis Bridge (Metaball Waist) during division
        if (t > 0.02 && t < 0.65) {
            const bridgeProgress = t / 0.65;
            const waistWidth = Math.max(0, (1 - bridgeProgress) * baseRadius * 1.2);
            if (waistWidth > 2) {
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(xPos, centerY - radPos * (1 - bridgeProgress * 0.4));
                ctx.quadraticCurveTo(centerX, centerY - waistWidth * 0.4, xNeg, centerY - radNeg * (1 - bridgeProgress * 0.4));
                ctx.lineTo(xNeg, centerY + radNeg * (1 - bridgeProgress * 0.4));
                ctx.quadraticCurveTo(centerX, centerY + waistWidth * 0.4, xPos, centerY + radPos * (1 - bridgeProgress * 0.4));
                ctx.closePath();
                
                const bridgeGrad = ctx.createLinearGradient(xPos, centerY, xNeg, centerY);
                bridgeGrad.addColorStop(0, colPos);
                bridgeGrad.addColorStop(0.5, baseColor);
                bridgeGrad.addColorStop(1, colNeg);
                ctx.fillStyle = bridgeGrad;
                ctx.globalAlpha = 1 - bridgeProgress;
                ctx.fill();
                ctx.restore();
            }
        }
        
        // Child Sphere 1: Positive subgroup
        const curColorPos = this.interpolateColor(baseColor, colPos, t);
        const curRadPos = baseRadius + (radPos - baseRadius) * t;
        const labelPosTop = t > 0.5 ? `${(r.purity_pos * 100).toFixed(1)}%` : '';
        const labelPosSub = t > 0.5 ? `Subgroup (n = ${r.n_pos})` : '';
        this.draw3DSphere(ctx, xPos, centerY, curRadPos, curColorPos, labelPosTop, labelPosSub, t > 0.5 ? '#22c55e' : null);

        // Child Sphere 2: Remainder
        const curColorNeg = this.interpolateColor(baseColor, colNeg, t);
        const curRadNeg = baseRadius + (radNeg - baseRadius) * t;
        const labelNegTop = t > 0.5 ? `${(r.purity_neg * 100).toFixed(1)}%` : '';
        const labelNegSub = t > 0.5 ? `Remainder (n = ${r.n_neg})` : '';
        this.draw3DSphere(ctx, xNeg, centerY, curRadNeg, curColorNeg, labelNegTop, labelNegSub, t > 0.5 ? '#ef4444' : null);
        
        // Center Metrics Badge between separated spheres
        if (t > 0.6) {
            const badgeAlpha = Math.min(1.0, (t - 0.6) / 0.4);
            ctx.save();
            ctx.globalAlpha = badgeAlpha;
            
            ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
            ctx.lineWidth = 1;
            
            const bw = 84, bh = 32, bx = centerX - bw / 2, by = centerY - bh / 2;
            this.roundRect(ctx, bx, by, bw, bh, 6);
            ctx.fill();
            ctx.stroke();
            
            ctx.fillStyle = '#a5b4fc';
            ctx.font = '600 10px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`NMI: ${r.nmi_local.toFixed(2)}`, centerX, by + 13);
            ctx.fillStyle = '#4ade80';
            ctx.fillText(`ΔVIR: +${(r.delta_vir*100).toFixed(1)}%`, centerX, by + 25);
            ctx.restore();
        }
    },

    draw3DSphere(ctx, x, y, radius, hexColor, labelTop, labelBottom, glowColor = null) {
        ctx.save();
        
        // Soft drop shadow
        ctx.beginPath();
        ctx.ellipse(x, y + radius + 6, radius * 0.75, 4, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.filter = 'blur(4px)';
        ctx.fill();
        ctx.filter = 'none';

        // Outer glow
        if (glowColor) {
            ctx.beginPath();
            ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
            ctx.strokeStyle = glowColor;
            ctx.lineWidth = 2;
            ctx.shadowColor = glowColor;
            ctx.shadowBlur = 8;
            ctx.stroke();
            ctx.shadowBlur = 0;
        }

        // Radial gradient for 3D sphere volume
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        
        const grad = ctx.createRadialGradient(
            x - radius * 0.32, y - radius * 0.32, radius * 0.08,
            x, y, radius
        );
        
        const rgb = this.hexToRgb(hexColor);
        grad.addColorStop(0, '#ffffff');
        grad.addColorStop(0.2, `rgba(${Math.min(255, rgb.r + 45)}, ${Math.min(255, rgb.g + 45)}, ${Math.min(255, rgb.b + 45)}, 1)`);
        grad.addColorStop(0.7, hexColor);
        grad.addColorStop(1, `rgba(${Math.max(0, rgb.r - 55)}, ${Math.max(0, rgb.g - 55)}, ${Math.max(0, rgb.b - 55)}, 1)`);
        
        ctx.fillStyle = grad;
        ctx.fill();

        // Labels
        if (labelTop) {
            ctx.font = '700 11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillStyle = '#ffffff';
            ctx.shadowColor = 'rgba(0,0,0,0.85)';
            ctx.shadowBlur = 4;
            ctx.fillText(labelTop, x, y + 4);
            ctx.shadowBlur = 0;
        }
        
        if (labelBottom) {
            ctx.font = '500 10px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillStyle = '#cbd5e1';
            ctx.fillText(labelBottom, x, y + radius + 16);
        }

        ctx.restore();
    },

    hexToRgb(hex) {
        if (!hex || typeof hex !== 'string') return { r: 150, g: 120, b: 80 };
        if (hex.startsWith('rgb')) {
            const m = hex.match(/\d+/g);
            if (m && m.length >= 3) return { r: parseInt(m[0]), g: parseInt(m[1]), b: parseInt(m[2]) };
        }
        const clean = hex.replace('#', '');
        return {
            r: parseInt(clean.substring(0, 2), 16) || 150,
            g: parseInt(clean.substring(2, 4), 16) || 120,
            b: parseInt(clean.substring(4, 6), 16) || 80
        };
    },

    interpolateColor(hex1, hex2, factor) {
        const rgb1 = this.hexToRgb(hex1);
        const rgb2 = this.hexToRgb(hex2);
        const r = Math.round(rgb1.r + (rgb2.r - rgb1.r) * factor);
        const g = Math.round(rgb1.g + (rgb2.g - rgb1.g) * factor);
        const b = Math.round(rgb1.b + (rgb2.b - rgb1.b) * factor);
        return `rgb(${r}, ${g}, ${b})`;
    },

    roundRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    }
};

let _currentXaiFilterIdx = -1;
let _xaiResultsCache = null;

function renderXaiPanel(cdata, results) {
    const panel = document.getElementById('xaiPanel');
    const info = document.getElementById('xaiPanelInfo');
    const list = document.getElementById('xaiFilterList');
    
    _xaiResultsCache = results;
    _currentXaiFilterIdx = -1;
    
    // Format human-friendly coordinates
    let coordsText = Object.entries(cdata.coords).map(([k, v]) => {
        const colObj = allColumnsData ? allColumnsData.find(c => c.id === k) : null;
        const colLabel = colObj ? colObj.label : k;
        let valLabel = v;
        if (colObj && colObj.criteria) {
            const critObj = colObj.criteria.find(cr => cr.id === v);
            if (critObj) valLabel = critObj.label;
        }
        return `<b>${colLabel}:</b> ${valLabel}`;
    }).join(' &nbsp;|&nbsp; ');
    
    info.innerHTML = `
        <div style="font-weight:600; color:#f1f5f9; margin-bottom:4px;">📍 Discrete Center:</div>
        <div style="color:#cbd5e1; margin-bottom:8px; font-size:0.85rem;">${coordsText}</div>
        <div style="display:flex; gap:16px; font-size:0.82rem; color:var(--text-muted); background:rgba(255,255,255,0.04); padding:6px 10px; border-radius:6px;">
            <span>📦 Objects: <b style="color:#f8fafc;">${cdata.N} pcs.</b></span>
            <span>🎯 Initial Purity: <b style="color:#f8fafc;">${(cdata.pur*100).toFixed(1)}%</b></span>
        </div>
    `;
    
    list.innerHTML = '';
    if (!results || results.length === 0) {
        list.innerHTML = '<div style="color:var(--text-dim); padding: 16px; text-align:center; font-size:0.85rem;">No statistically reliable split candidates found for this center.</div>';
    } else {
        results.forEach((r, idx) => {
            const condsText = r.human_text || r.conditions.map(c => `${c.human_col || c.col} = ${c.human_val || c.val}`).join(' ∧ ');
            const nmi = r.nmi_local.toFixed(3);
            const vir = (r.delta_vir * 100).toFixed(1);
            const pPos = (r.purity_pos * 100).toFixed(1);
            const pNeg = (r.purity_neg * 100).toFixed(1);
            
            const html = `
                <div class="xai-filter-item" id="xai-filter-card-${idx}">
                    <div class="xai-filter-header">
                        <div class="xai-filter-conds">${r.reliability} ${condsText}</div>
                    </div>
                    <div class="xai-filter-stats">
                        <span class="xai-stat-badge pos">✨ Subgroup: <b>${pPos}%</b> (n=${r.n_pos})</span>
                        <span class="xai-stat-badge neg">Remainder: <b>${pNeg}%</b> (n=${r.n_neg})</span>
                        <span class="xai-stat-badge metric">NMI: <b>${nmi}</b></span>
                        <span class="xai-stat-badge metric">ΔVIR: <b>+${vir}%</b></span>
                    </div>
                    <div class="xai-filter-actions">
                        <button class="xai-btn" id="btn-highlight-${idx}" onclick="highlightXaiFilter(${idx})">⚡ Split</button>
                    </div>
                </div>
            `;
            list.innerHTML += html;
        });
    }
    
    panel.style.display = 'flex';
    
    // Initialize Mitosis Canvas Engine
    MitosisEngine.init('xaiMitosisCanvas');
    MitosisEngine.setCenter(cdata);
}

function closeXaiPanel() {
    document.getElementById('xaiPanel').style.display = 'none';
    _currentXaiFilterIdx = -1;
    if (MitosisEngine.animId) cancelAnimationFrame(MitosisEngine.animId);
}

function highlightXaiFilter(idx) {
    if (!_xaiResultsCache || !_xaiResultsCache[idx]) return;
    
    const cards = document.querySelectorAll('.xai-filter-item');
    const r = _xaiResultsCache[idx];
    
    if (_currentXaiFilterIdx === idx) {
        // Toggle off - return to unified sphere
        _currentXaiFilterIdx = -1;
        cards.forEach(c => c.classList.remove('active-highlight'));
        const btn = document.getElementById(`btn-highlight-${idx}`);
        if (btn) {
            btn.classList.remove('active');
            btn.innerHTML = '⚡ Split';
        }
        MitosisEngine.resetToUnified();
        return;
    }
    
    _currentXaiFilterIdx = idx;
    cards.forEach((c, i) => {
        c.classList.toggle('active-highlight', i === idx);
        const btn = document.getElementById(`btn-highlight-${i}`);
        if (btn) {
            if (i === idx) {
                btn.classList.add('active');
                btn.innerHTML = '✖ Collapse';
            } else {
                btn.classList.remove('active');
                btn.innerHTML = '⚡ Split';
            }
        }
    });

    // Run Mitosis Animation in the dedicated canvas
    MitosisEngine.animateToRule(r);
}
