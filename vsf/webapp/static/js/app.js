// VSF v2.0 "Clean Core" frontend — Independent Branch Discovery UI
// (see Project_Master_Document.md Section 0 for what changed vs v1.0, and
// UI_Functional_Spec.md for the exact UX this file implements).
//
// State model:
//   currentBranchesResponse — the raw /api/analyze response: { target,
//     criterion, branches: { "1": payload, "2": payload, ... }, branch_dims,
//     default_branch }. One fetch per target/criterion selection.
//   activeBranchDim — which branch (string key into .branches) is currently
//     shown. Switching this is an INSTANT re-render (Project_Master_Document
//     .md Section 5.6, case 1) — never animated, since the feature set can
//     change entirely between branches.
//   currentPayload — currentBranchesResponse.branches[activeBranchDim], the
//     single-branch visualization payload (vsf/vis.py's
//     prepare_visualization_payload shape) the plot is built from.
//   activeDimensionality — the WITHIN-branch view dimensionality (1..branch
//     .metrics.d). Changing this animates a collapse/split of the currently
//     selected branch's own axes (Project_Master_Document.md Section 5.6,
//     case 2) — it never jumps between branches.

let currentBranchesResponse = null;
let activeBranchDim = null;
let currentPayload = null;
let activeDimensionality = null;
let allColumnsData = [];
let activeSliceIndex = null;  // Current 4D slice index (null = "All" slice-marginalized view)
let currentRenderedDim = null;
let isAnimating = false;

// Global Pattern Scan (Display Settings sidebar — see vsf/server.py's
// module docstring for the full algorithm). Scans every (column, value)
// One-vs-Rest criterion in the dataset in a background job on the server;
// `scanResultsByColumn` is null until a scan completes (no filter applied,
// Target Variable shows every column/value as usual), or
// {colId: [{value, max_nmi, best_d, best_mi, best_features}, ...]} once one
// has — the catalog is then rebuilt to show ONLY the columns/values that
// passed. This never touches branch discovery for the currently-analyzed
// target; it only filters which (column, value) pairs are offered as a
// NEW target to pick.
let scanResultsByColumn = null;
let scanPollTimer = null;
let currentDefaultTarget = null;

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
            currentDefaultTarget = defaultTarget;
            // Pick up a scan that was already running/finished before this
            // page load (e.g. a reload mid-scan) rather than losing it.
            await checkExistingScanOnLoad();
            renderCatalog();
        } else {
            showAnalysisError(`Failed to load dataset catalog (HTTP ${colRes.status}).`);
        }

        await runAnalysis(defaultTarget, null);
    } catch (err) {
        console.error("Initialization error:", err);
        showAnalysisError('Initialization failed: ' + err.message);
        await runAnalysis(defaultTarget, null);
    }
}

// Rebuilds allColumnsData into a catalog containing only the columns/
// values a completed scan kept (a no-op copy when no scan filter is
// active). Surviving criteria get their max NMI/dimensionality appended to
// their label so the filtered list still carries that information.
function buildScanFilteredCatalog() {
    if (!scanResultsByColumn) return allColumnsData;
    const filtered = [];
    allColumnsData.forEach(col => {
        const passing = scanResultsByColumn[col.id];
        if (!passing || passing.length === 0) return;
        const passingByValue = {};
        passing.forEach(p => { passingByValue[p.value] = p; });
        const criteria = col.criteria
            .filter(c => Object.prototype.hasOwnProperty.call(passingByValue, c.id))
            .map(c => {
                const info = passingByValue[c.id];
                return {
                    id: c.id,
                    label: `${c.label} · NMI ${(info.max_nmi * 100).toFixed(1)}% (${info.best_d}D)`,
                };
            });
        if (criteria.length > 0) {
            filtered.push({ id: col.id, label: col.label, criteria });
        }
    });
    return filtered;
}

function renderCatalog() {
    populateCatalog(buildScanFilteredCatalog(), currentDefaultTarget);
}

// ---------------------------------------------------------------------
// Global Pattern Scan controls

function setScanControlsRunning(running) {
    const startBtn = document.getElementById('btnStartScan');
    const cancelBtn = document.getElementById('btnCancelScan');
    const thresholdInput = document.getElementById('scanNmiThreshold');
    if (startBtn) startBtn.style.display = running ? 'none' : '';
    if (cancelBtn) cancelBtn.style.display = running ? '' : 'none';
    if (thresholdInput) thresholdInput.disabled = running;
}

function showScanStatusMsg(text, kind) {
    const el = document.getElementById('scanStatusMsg');
    if (!el) return;
    if (!text) {
        el.style.display = 'none';
        el.className = 'scan-status-msg';
        return;
    }
    el.textContent = text;
    el.className = 'scan-status-msg' + (kind ? ' scan-status-' + kind : '');
    el.style.display = 'block';
}

async function startDatasetScan() {
    const thresholdInput = document.getElementById('scanNmiThreshold');
    const threshold = thresholdInput ? Number(thresholdInput.value) : 80;
    showScanStatusMsg(null);

    if (!Number.isFinite(threshold) || threshold < 0 || threshold >= 100) {
        showScanStatusMsg('NMI threshold must be a number in [0, 100).', 'error');
        return;
    }

    try {
        const res = await fetch('/api/scan/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nmi_threshold: threshold }),
        });
        if (!res.ok) {
            let detail = `HTTP ${res.status}`;
            try {
                const body = await res.json();
                if (body && body.error) detail = body.error;
            } catch (_) { /* body wasn't JSON */ }
            showScanStatusMsg('Failed to start scan: ' + detail, 'error');
            return;
        }
        setScanControlsRunning(true);
        const progressWrap = document.getElementById('scanProgressWrap');
        if (progressWrap) progressWrap.style.display = 'flex';
        startScanPolling();
    } catch (err) {
        console.error('Failed to start scan:', err);
        showScanStatusMsg('Failed to start scan: ' + err.message, 'error');
    }
}

async function cancelDatasetScan() {
    try {
        await fetch('/api/scan/cancel', { method: 'POST' });
    } catch (err) {
        console.error('Failed to cancel scan:', err);
    }
}

function clearDatasetScan() {
    scanResultsByColumn = null;
    renderCatalog();
    showScanStatusMsg(null);
    const clearBtn = document.getElementById('btnClearScan');
    if (clearBtn) clearBtn.style.display = 'none';
    const progressWrap = document.getElementById('scanProgressWrap');
    if (progressWrap) progressWrap.style.display = 'none';
}

function startScanPolling() {
    stopScanPolling();
    pollScanStatusOnce();
    scanPollTimer = setInterval(pollScanStatusOnce, 800);
}

function stopScanPolling() {
    if (scanPollTimer !== null) {
        clearInterval(scanPollTimer);
        scanPollTimer = null;
    }
}

async function pollScanStatusOnce() {
    try {
        const res = await fetch('/api/scan/status');
        if (!res.ok) return;
        const status = await res.json();
        handleScanStatus(status);
    } catch (err) {
        console.error('Scan status poll failed:', err);
    }
}

// Applies a completed/cancelled scan's results as the active catalog
// filter and updates the status message — does NOT re-render the catalog
// itself, so callers that already know they'll render afterward (init's
// checkExistingScanOnLoad) don't render twice.
function applyScanResults(results, thresholdPct, wasCancelled) {
    scanResultsByColumn = {};
    results.forEach(r => {
        if (!scanResultsByColumn[r.column]) scanResultsByColumn[r.column] = [];
        scanResultsByColumn[r.column].push(r);
    });

    const clearBtn = document.getElementById('btnClearScan');
    if (clearBtn) clearBtn.style.display = 'block';

    const nCols = Object.keys(scanResultsByColumn).length;
    const nVals = results.length;
    const prefix = wasCancelled ? 'Scan cancelled — ' : '';
    if (nVals === 0) {
        showScanStatusMsg(`${prefix}No column/value exceeded NMI > ${thresholdPct}%.`, 'empty');
    } else {
        showScanStatusMsg(
            `${prefix}${nVals} value(s) across ${nCols} column(s) exceed NMI > ${thresholdPct}% — Target Variable filtered.`,
            'ok'
        );
    }
}

function handleScanStatus(status) {
    if (status.status === 'idle') {
        stopScanPolling();
        setScanControlsRunning(false);
        return;
    }

    if (status.status === 'running') {
        setScanControlsRunning(true);
        const progressWrap = document.getElementById('scanProgressWrap');
        if (progressWrap) progressWrap.style.display = 'flex';
        const p = status.progress || { current: 0, total: 0, label: '' };
        const pct = p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
        const fill = document.getElementById('scanProgressFill');
        if (fill) fill.style.width = pct + '%';
        const label = document.getElementById('scanProgressLabel');
        if (label) label.textContent = `${p.current} / ${p.total}` + (p.label ? ' — ' + p.label : '');
        return;
    }

    // Terminal states: done / cancelled / error
    stopScanPolling();
    setScanControlsRunning(false);

    if (status.status === 'error') {
        showScanStatusMsg('Scan failed: ' + (status.error || 'unknown error'), 'error');
        return;
    }

    applyScanResults(status.results || [], status.threshold_pct, status.status === 'cancelled');
    renderCatalog();
}

// Called once from init(), before the first renderCatalog() — picks up a
// scan that was already running or had already finished before this page
// load (e.g. the page was reloaded mid-scan), so a reload never loses a
// scan's progress or result.
async function checkExistingScanOnLoad() {
    try {
        const res = await fetch('/api/scan/status');
        if (!res.ok) return;
        const status = await res.json();
        if (status.status === 'running') {
            setScanControlsRunning(true);
            const progressWrap = document.getElementById('scanProgressWrap');
            if (progressWrap) progressWrap.style.display = 'flex';
            startScanPolling();
        } else if (status.status === 'done' && status.results && status.results.length > 0) {
            applyScanResults(status.results, status.threshold_pct, false);
        }
        // status === 'idle' -> nothing to restore, defaults already correct.
    } catch (err) {
        console.error('Failed to check existing scan status:', err);
    }
}

function populateCatalog(cols, defaultTarget) {
    const accordion = document.getElementById('catalogAccordion');
    accordion.innerHTML = '';
    let activeCritItem = null;

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

            critHeader.onclick = async (e) => {
                e.stopPropagation();
                if (activeCritItem && activeCritItem !== critItem) {
                    activeCritItem.classList.remove('active');
                }
                critItem.classList.add('active');
                activeCritItem = critItem;
                await runAnalysis(col.id, crit.id);
            };

            critItem.appendChild(critHeader);
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

function showAnalysisError(message) {
    const el = document.getElementById('analysisError');
    if (!el) return;
    if (message) {
        el.textContent = message;
        el.style.display = 'block';
    } else {
        el.textContent = '';
        el.style.display = 'none';
    }
}

async function runAnalysis(targetCol, criterion = null) {
    showLoader(true);
    showAnalysisError(null);
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
            const data = await response.json();
            loadBranchesResponse(data);
        } else {
            let detail = `HTTP ${response.status}`;
            try {
                const errBody = await response.json();
                if (errBody && errBody.error) detail = errBody.error;
            } catch (_) { /* body wasn't JSON, keep status-only detail */ }
            console.error("Error fetching analysis:", detail);
            showAnalysisError(`Analysis failed: ${detail}`);
        }
    } catch (err) {
        console.error("API POST failed:", err);
        showAnalysisError('Analysis request failed: ' + err.message);
    } finally {
        showLoader(false);
    }
}

// (Re-)renders the branch list for the ALREADY-FETCHED
// currentBranchesResponse (no new /api/analyze request) and picks which
// branch becomes active. Called right after a fresh analysis
// (preserveActiveIfPossible=false — always pick the server's own default)
// — there is no client-side branch filter, every discovered branch is
// always shown.
function selectDefaultBranchAndRender(preserveActiveIfPossible) {
    if (!currentBranchesResponse) return;
    const data = currentBranchesResponse;
    const allDims = (data.branch_dims || []).map(String);
    if (allDims.length === 0) return; // handled by loadBranchesResponse's own early return

    renderBranchSelector(data);
    showAnalysisError(null);

    if (preserveActiveIfPossible && activeBranchDim && allDims.includes(activeBranchDim)) {
        return; // still valid — leave the current view exactly as-is
    }

    // Prefer the server's own default_branch (its highest-d branch), else
    // fall back to the highest-d branch available.
    const preferredDefault = data.default_branch != null ? String(data.default_branch) : null;
    const nextDim = (preferredDefault && allDims.includes(preferredDefault))
        ? preferredDefault
        : allDims[allDims.length - 1];
    selectBranch(nextDim, /* fromInitialLoad */ true);
}

// Loads a fresh /api/analyze response: resets ALL per-analysis state (branch
// selection, within-branch dimensionality, 4D slice), then renders the
// branch list and selects the server's default branch.
function loadBranchesResponse(data) {
    currentBranchesResponse = data;
    stopSlicePlayback();
    activeSliceIndex = null;
    currentRenderedDim = null;

    const dims = data.branch_dims || [];
    if (dims.length === 0) {
        currentPayload = null;
        activeBranchDim = null;
        renderBranchSelector(data);
        showAnalysisError('No branches found — this dataset has no usable feature columns for the chosen target.');
        return;
    }

    selectDefaultBranchAndRender(/* preserveActiveIfPossible */ false);
}

function renderBranchSelector(response) {
    const container = document.getElementById('branchList');
    const caption = document.getElementById('branchCaption');
    if (!container) return;
    container.innerHTML = '';

    const allDims = response.branch_dims || [];
    if (allDims.length === 0) {
        container.innerHTML = '<div style="color: var(--text-dim); font-size: 0.85rem; padding: 1rem;">No branches available.</div>';
        if (caption) caption.style.display = 'none';
        return;
    }
    if (caption) caption.style.display = 'block';

    allDims.forEach(dRaw => {
        const dKey = String(dRaw);
        const branch = response.branches ? response.branches[dKey] : null;
        if (!branch || !branch.metrics) return;
        const m = branch.metrics;
        const features = (branch.selected_features || []).join(' + ');

        const card = document.createElement('div');
        card.className = 'branch-card' + (dKey === activeBranchDim ? ' active' : '');
        card.dataset.dim = dKey;
        card.innerHTML = `
            <div class="branch-card-head">
                <span class="branch-dim-badge">${dKey}D</span>
                <span class="branch-mi" title="Raw mutual information I(Z;X_S) — not normalized, no significance test">I = ${m.mi.toFixed(3)}</span>
                <span class="branch-nmi" title="Normalized mutual information of this branch">NMI = ${(m.nmi * 100).toFixed(1)}%</span>
            </div>
            <div class="branch-features">${features || '—'}</div>
        `;
        card.onclick = () => selectBranch(dKey);
        container.appendChild(card);
    });
}

// Branch switch: an INSTANT re-render of the whole scene against a
// DIFFERENT independently-discovered feature set. Never animated — see
// Project_Master_Document.md Section 5.6, case 1. Distinct from
// setDimensionality(), which animates within one already-selected branch.
function selectBranch(dKey, fromInitialLoad = false) {
    dKey = String(dKey);
    if (!currentBranchesResponse || !currentBranchesResponse.branches[dKey]) return;
    if (!fromInitialLoad && dKey === activeBranchDim) return;
    if (isAnimating) return; // don't interrupt an in-flight within-branch collapse/split animation

    stopSlicePlayback();
    activeBranchDim = dKey;
    currentPayload = currentBranchesResponse.branches[dKey];
    activeDimensionality = currentPayload.metrics.d; // start fully expanded to the branch's own dimensionality
    activeSliceIndex = null;
    currentRenderedDim = null; // forces renderPlot() (no-animation path) in setDimensionality()

    document.querySelectorAll('#branchList .branch-card').forEach(card => {
        card.classList.toggle('active', card.dataset.dim === dKey);
    });

    updateDashboard(currentPayload);
}

function showLoader(show) {
    const loader = document.getElementById('loader');
    if (show) loader.classList.add('active');
    else loader.classList.remove('active');
}

// 4-zone Purity Color Scale (Project_Master_Document.md Section 5.3).
// Exhaustive, non-overlapping partition of [0, 1]:
//   [0, 0.25)    -> Red    ("Alternative": target class virtually absent)
//   [0.25, 0.75] -> Brown  ("Murky Zone": classes physically mixed)
//   (0.75, 0.85] -> Yellow ("High": target dominant, but with visible admixture)
//   (0.85, 1.0]  -> Green  ("Target": target class confidently dominant)
const PROB_COLORS = [
    '#ef4444', // Red
    '#92572e', // Brown
    '#eab308', // Yellow
    '#22c55e'  // Green
];
const PALETTE_COUNT = PROB_COLORS.length;

function getColorIndexForPurity(purity) {
    if (purity < 0.25) return 0;   // Red:    [0, 0.25)
    if (purity <= 0.75) return 1;  // Brown:  [0.25, 0.75]
    if (purity <= 0.85) return 2;  // Yellow: (0.75, 0.85]
    return 3;                      // Green:  (0.85, 1.0]
}

function updateDashboard(payload) {
    currentPayload = payload;
    const m = payload.metrics;

    currentRenderedDim = null; // Force full plot re-render with new axis titles

    // Handle 4D slice controller setup
    if (payload.slice_axis) {
        if (activeSliceIndex === null) {
            activeSliceIndex = 0;  // Default to first slice
        }
        renderSliceTabs(payload);
    } else {
        activeSliceIndex = null;
        stopSlicePlayback();
        const sliceCtrl = document.getElementById('slice-controller');
        if (sliceCtrl) sliceCtrl.style.display = 'none';
    }

    document.getElementById('totalSamplesVal').innerText = (payload.total_samples || (payload.x ? payload.x.length : 0)).toLocaleString();

    // Render Dimension Switcher toggle buttons (within-branch collapse/split, 1..branch.metrics.d)
    renderDimensionButtons(payload);

    // Populate Exact Values for Stroke settings
    const exactSelect = document.getElementById('strokeExactVal');
    if (exactSelect) {
        exactSelect.innerHTML = '<option value="">-- Select --</option>';
        let dimStr = (activeDimensionality || m.d).toString();
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

    setDimensionality(activeDimensionality || m.d);
}

function renderDimensionButtons(payload) {
    const container = document.getElementById('dimButtonsGroup');
    if (!container) return;
    container.innerHTML = '';

    const branchD = (payload.metrics && payload.metrics.d) ? payload.metrics.d : 1;
    for (let d = 1; d <= branchD; d++) {
        const btn = document.createElement('button');
        btn.className = 'toggle-btn dim-btn' + (d === activeDimensionality ? ' active' : '');
        btn.dataset.dim = d;
        btn.innerText = `${d}D`;
        btn.title = d === branchD
            ? `This branch's full dimensionality (${d}D)`
            : `Collapse this branch to a ${d}D view (animated, same features)`;
        btn.onclick = () => setDimensionality(d);
        container.appendChild(btn);
    }
}

function updateHUDForDimension(d) {
    if (!currentPayload || !currentPayload.metrics) return;
    const m = currentPayload.metrics;

    // MI/NMI are the BRANCH's own aggregate statistics (computed once, for
    // its full dimensionality m.d) — not recomputed per within-branch
    // collapsed view, so they stay fixed as `d` changes via setDimensionality().
    const miEl = document.getElementById('val-mi');
    if (miEl) miEl.innerText = `${m.mi.toFixed(3)} bits`;

    const nmiEl = document.getElementById('val-nmi');
    if (nmiEl) nmiEl.innerText = `${(m.nmi * 100).toFixed(1)}%`;

    const viewEl = document.getElementById('val-viewdim');
    if (viewEl) {
        viewEl.innerText = (d === m.d) ? `${d}D (full branch)` : `${d}D (collapsed from ${m.d}D)`;
    }

    updateAxesList(d);
}

function updateAxesList(d) {
    const axesContainer = document.getElementById('axesListContainer');
    if (!axesContainer || !currentPayload || !currentPayload.selected_features) return;
    axesContainer.innerHTML = '';
    const labels = ['X-Axis (1D)', 'Y-Axis (2D)', 'Z-Axis (3D)', '4D Slice (Tabs)'];
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

// WITHIN-branch dimensionality collapse/split (Project_Master_Document.md
// Section 5.6, case 2): animates the CURRENTLY selected branch's own axes
// between 1D/2D/3D/4D. Never switches feature sets — see selectBranch()
// for the (unanimated) cross-branch switch.
function setDimensionality(d) {
    if (d === currentRenderedDim && d === activeDimensionality) return;
    if (isAnimating) return;

    const fromDim = currentRenderedDim;
    activeDimensionality = d;

    document.querySelectorAll('#dimButtonsGroup .dim-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.dim) === d);
    });

    updateHUDForDimension(d);

    const sliceCtrl = document.getElementById('slice-controller');
    if (sliceCtrl) {
        if (d >= 4 && currentPayload && currentPayload.slice_axis) {
            sliceCtrl.style.display = 'flex';
        } else {
            sliceCtrl.style.display = 'none';
            stopSlicePlayback();
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
    // "All" tab (marginalizes over the 4th dimension — Project_Master_Document.md Section 5.5)
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

    const branchD = (currentPayload && currentPayload.metrics) ? currentPayload.metrics.d : 4;
    if (activeDimensionality !== branchD) {
        activeDimensionality = branchD;
        document.querySelectorAll('#dimButtonsGroup .dim-btn').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.dim) === branchD);
        });
        updateHUDForDimension(branchD);
    }

    if (currentPayload) {
        if (currentRenderedDim === null) {
            renderPlot(currentPayload);
        } else {
            transitionDimensionality(currentRenderedDim, branchD, window._lastActiveSliceIndex);
        }
    }
}

// --- 4D slice auto-play (Project_Master_Document.md Section 5.5: "Optional
// autoplay (play/pause/speed) available on top of discrete slices") ---
let _sliceIntervalId = null;
let _sliceIsPlaying = false;

function toggleSlicePlayback() {
    if (_sliceIsPlaying) {
        stopSlicePlayback();
    } else {
        startSlicePlayback();
    }
}

function startSlicePlayback() {
    if (!currentPayload || !currentPayload.slice_axis) return;
    const ticks = currentPayload.slice_axis.ticks;
    if (!ticks || ticks.length === 0) return;

    _sliceIsPlaying = true;
    const btn = document.getElementById('slicePlayBtn');
    if (btn) { btn.textContent = '⏸'; btn.title = 'Pause auto-advance'; }

    const speedSelect = document.getElementById('slicePlaySpeed');
    const intervalMs = speedSelect ? parseInt(speedSelect.value, 10) : 1200;

    if (_sliceIntervalId) clearInterval(_sliceIntervalId);
    _sliceIntervalId = setInterval(() => {
        if (isAnimating) return; // skip a beat rather than overlap the in-flight transition
        const count = ticks.length;
        let next = (activeSliceIndex === null) ? 0 : activeSliceIndex + 1;
        if (next >= count) next = 0;
        selectSlice(next);
    }, intervalMs);
}

function stopSlicePlayback() {
    _sliceIsPlaying = false;
    const btn = document.getElementById('slicePlayBtn');
    if (btn) { btn.textContent = '▶'; btn.title = 'Play/Pause auto-advance through slices'; }
    if (_sliceIntervalId) {
        clearInterval(_sliceIntervalId);
        _sliceIntervalId = null;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const speedSelect = document.getElementById('slicePlaySpeed');
    if (speedSelect) {
        speedSelect.addEventListener('change', () => {
            if (_sliceIsPlaying) startSlicePlayback(); // restart timer at new interval
        });
    }
});

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
    const dim = activeDimensionality || (payload.metrics ? payload.metrics.d : 3);
    const data = buildPlotData(payload, dim, activeSliceIndex);

    window._lastFilteredData = {
        purity: data.fPurity,
        opacity: data.fOpacity,
        baseSizes: data.fSizes
    };

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

// Animated within-branch collapse (SPLIT/COLLAPSE) or 4D film-strip
// transition — see setDimensionality()'s doc comment for the branch-switch
// vs. within-branch distinction this implements.
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
    if (id === 'branches') {
        headerId = 'headerBranches';
        contentId = 'contentBranches';
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

window.addEventListener('DOMContentLoaded', init);
