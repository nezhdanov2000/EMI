// VSF v2.0 "Clean Core" frontend — Independent Branch Discovery UI
// (see Project_Master_Document.md for the specification this file
// implements).
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
// {colId: [{value, best_d, best_features, p_value, p_value_familywise, ...}, ...]} once one
// has — the catalog is then rebuilt to show ONLY the columns/values that
// passed. This never touches branch discovery for the currently-analyzed
// target; it only filters which (column, value) pairs are offered as a
// NEW target to pick.
let scanResultsByColumn = null;
let scanPollTimer = null;
let currentDefaultTarget = null;

async function init() {
    initColorScale();
    // Defensive, not load-bearing: index.html already ships btnStartScan
    // with the disabled attribute matching the field's empty default, but
    // deriving it here too means the two never have to be kept in sync by
    // hand, and covers a stale bfcache-restored field value on reload.
    updateScanStartButtonState();
    // No analysis runs until the user picks a target + value (see
    // showWelcomeState()): a target has no interesting class until the user
    // names one, so there is nothing yet for discover_branches to search
    // for. #loader ships without the "active" class in index.html for the
    // same reason -- it is turned on only once runAnalysis() actually
    // starts a request.
    showWelcomeState(true);
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
            // Off the critical path: while the user reads the guide, feed
            // every catalog label through gl3d's text pipeline once, so the
            // first real render does not pay for it (see
            // warmUpPlotlyTextCache()).
            warmUpPlotlyTextCache(allColumnsData);
        } else {
            showAnalysisError(`Failed to load dataset catalog (HTTP ${colRes.status}).`);
        }
    } catch (err) {
        console.error("Initialization error:", err);
        showAnalysisError('Initialization failed: ' + err.message);
    }
}

// -- WebGL text warm-up -------------------------------------------------
// Plotly's gl3d renders axis titles and tick labels as triangulated text
// meshes (vectorize-text), memoised per unique (string, font). Measured on
// adult_census in the desktop browser: the FIRST render of a target pays
// 2.7-3.1 s inside Plotly.newPlot, of which ~2.5 s is that triangulation
// of the tick labels and axis titles; re-rendering the same labels costs
// ~0.1-0.3 s, and a target whose labels are new costs ~0.7 s. Every label
// the renderer can ever show is a catalog label (vsf.vis uses the same
// humanize_* functions for both), so they can all be pushed through the
// pipeline once, in an off-screen plot, while the welcome guide is up.
// Done in small batches with the event loop yielded in between so the page
// never freezes for more than a fraction of a second; abandoned as soon as
// the user starts an analysis, since the real render then triangulates
// what it needs anyway. Measured: first-render 2.7-3.1 s -> 0.26 s.
const _PLOTLY_WARMUP_BATCH = 12;
const _PLOTLY_WARMUP_MAX_LABELS = 400;
let _plotlyWarmupCancelled = false;

async function warmUpPlotlyTextCache(columns) {
    if (typeof Plotly === 'undefined' || !Array.isArray(columns)) return;
    const titles = [];
    const ticks = [];
    const seen = new Set();
    columns.forEach(col => {
        if (col.label && !seen.has('c' + col.label)) { seen.add('c' + col.label); titles.push(col.label); }
        (col.criteria || []).forEach(v => {
            if (v.label && !seen.has('v' + v.label)) { seen.add('v' + v.label); ticks.push(v.label); }
        });
    });
    if (ticks.length > _PLOTLY_WARMUP_MAX_LABELS) ticks.length = _PLOTLY_WARMUP_MAX_LABELS;
    const holder = document.createElement('div');
    holder.setAttribute('aria-hidden', 'true');
    holder.style.cssText = 'position:absolute;left:-9999px;top:0;width:320px;height:320px;overflow:hidden;';
    document.body.appendChild(holder);
    const axis = (labels, title) => ({
        title: { text: title || '', font: { color: '#c084fc', size: 13 } },
        tickvals: labels.map((_, i) => i), ticktext: labels,
        range: [-0.5, Math.max(labels.length, 1) - 0.5],
        tickfont: { color: '#e2e8f0', size: 11 },
        backgroundcolor: '#090d1a', showgrid: false, zeroline: false, showspikes: false
    });
    try {
        let ti = 0, vi = 0;
        while ((vi < ticks.length || ti < titles.length) && !_plotlyWarmupCancelled) {
            const batch = ticks.slice(vi, vi + _PLOTLY_WARMUP_BATCH);
            vi += _PLOTLY_WARMUP_BATCH;
            const third = Math.ceil(batch.length / 3);
            const axes = [batch.slice(0, third), batch.slice(third, 2 * third), batch.slice(2 * third)];
            const t = [titles[ti], titles[ti + 1], titles[ti + 2]];
            ti += 3;
            await Plotly.newPlot(holder, [{
                type: 'scatter3d', mode: 'markers', x: [0], y: [0], z: [0], marker: { size: 2 }
            }], {
                paper_bgcolor: '#070a13', showlegend: false,
                scene: { aspectmode: 'cube', xaxis: axis(axes[0], t[0]), yaxis: axis(axes[1], t[1]), zaxis: axis(axes[2], t[2]) },
                margin: { l: 0, r: 0, b: 0, t: 0 }, font: { family: 'Inter', color: '#94a3b8' }
            }, { displayModeBar: false, staticPlot: true });
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    } catch (err) {
        console.warn('Plotly text warm-up skipped:', err);
    } finally {
        try { Plotly.purge(holder); } catch (_) { /* nothing to purge */ }
        holder.remove();
    }
}

// Toggles between the pre-selection guide (#welcomeState) and the results
// area (#resultsArea, the toolbar + axes bar + plot). Called with true from
// init() and with false the moment the user's first click starts
// runAnalysis() -- never flips back to true afterwards, since lastTargetCol
// (and therefore applyCertificate()'s re-run) only makes sense once a
// target has actually been analysed at least once.
function showWelcomeState(show) {
    const welcome = document.getElementById('welcomeState');
    const results = document.getElementById('resultsArea');
    if (welcome) welcome.style.display = show ? '' : 'none';
    if (results) results.style.display = show ? 'none' : '';
}

// Rebuilds allColumnsData into a catalog containing only the columns/
// values a completed scan kept (a no-op copy when no scan filter is
// active). Surviving criteria get their certified coverage, centre count and
// dimensionality appended to their label so the filtered list still carries
// that information. v2.2: coverage, not U_adj -- the scan now selects on the
// quantity the display delivers, and the label must name the same one.
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
                    label: `${c.label} · ${info.direction === 'absence' ? 'free of value' : 'coverage'} ${(info.coverage * 100).toFixed(1)}% · ${info.n_centers} ${info.direction === 'absence' ? 'free cell' : 'centre'}${info.n_centers === 1 ? '' : 's'} (${info.best_d}D, p=${info.p_value === null || info.p_value === undefined ? 'n/a' : info.p_value.toFixed(3)})`,
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
    const thresholdInput = document.getElementById('scanCoverageThreshold');
    const minSamplesInput = document.getElementById('scanMinSamples');
    if (startBtn) startBtn.style.display = running ? 'none' : '';
    if (cancelBtn) cancelBtn.style.display = running ? '' : 'none';
    if (thresholdInput) thresholdInput.disabled = running;
    if (minSamplesInput) minSamplesInput.disabled = running;
    // Re-derive btnStartScan's disabled state from the threshold field
    // rather than force-enabling it here: a scan that just finished with
    // the field left empty (or emptied while it was running) must NOT
    // silently become startable again.
    if (!running) updateScanStartButtonState();
}

// Coverage threshold starts empty (no implicit default -- see
// index.html's scan-note) and stays empty until the user types a value,
// so btnStartScan is disabled whenever the field is blank or not a valid
// [0, 100] number: running the whole-dataset scan with nothing to filter
// on would just reproduce the unfiltered catalog at real compute cost.
// Wired to the field's oninput (index.html) and called once from init()
// so a page load always reflects the field's actual (empty) content
// rather than a stale server-rendered default.
function updateScanStartButtonState() {
    const startBtn = document.getElementById('btnStartScan');
    const thresholdInput = document.getElementById('scanCoverageThreshold');
    if (!startBtn || !thresholdInput) return;
    const raw = thresholdInput.value.trim();
    const n = Number(raw);
    const valid = raw !== '' && Number.isFinite(n) && n >= 0 && n <= 100;
    startBtn.disabled = !valid;
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

// fdr_q = 1.0 and n_permutations_familywise = 0 are not placeholders: they
// are the values that make the server's Benjamini-Hochberg step a verified
// no-op (see metrics.py::benjamini_hochberg -- q=1 keeps every hypothesis,
// since the largest p-value's threshold is q*m/m = 1.0 and p-values are
// always <= 1) and skip the corrected-p-value computation that would
// otherwise feed it. Coverage threshold plus the scan's own Min. objects
// (scanMinSamples, below) are deliberately the ONLY gates here; see
// index.html's scan-note for what dropping FDR trades away.
const SCAN_FDR_Q_DISABLED = 1.0;
const SCAN_N_PERMUTATIONS_FAMILYWISE_DISABLED = 0;

async function startDatasetScan() {
    const thresholdInput = document.getElementById('scanCoverageThreshold');
    const rawThreshold = thresholdInput ? thresholdInput.value.trim() : '';
    const minSamplesInput = document.getElementById('scanMinSamples');
    const minSamples = minSamplesInput ? Math.round(Number(minSamplesInput.value)) : 1;
    showScanStatusMsg(null);

    // Belt-and-suspenders: btnStartScan is disabled whenever the field is
    // empty (updateScanStartButtonState), so this should be unreachable via
    // a normal click, but startDatasetScan is also called from nowhere else
    // that could bypass the disabled state, and a silent Number('') === 0
    // fallback here would start a real (expensive) scan the user never
    // configured.
    if (rawThreshold === '') {
        showScanStatusMsg('Enter a coverage threshold before scanning.', 'error');
        return;
    }
    const threshold = Number(rawThreshold);
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 100) {
        showScanStatusMsg('Coverage threshold must be a number in [0, 100].', 'error');
        return;
    }
    if (!Number.isFinite(minSamples) || minSamples < 1) {
        showScanStatusMsg('Min. objects must be a whole number of 1 or more.', 'error');
        return;
    }

    try {
        const res = await fetch('/api/scan/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // The scan is certified with the same tau/alpha the viewport is
            // showing, so a pair that survives the scan is a pair whose
            // centres the user will actually see when they click it.
            body: JSON.stringify({
                coverage_threshold: threshold,
                fdr_q: SCAN_FDR_Q_DISABLED,
                tau: readCertTau(),
                alpha: readCertAlpha(),
                rule: readCertRule(),
                direction: readDirection(),
                // The scan's OWN Min. objects, not readCertMinSamples() --
                // tau/alpha ARE shared with the Green/Red sliders
                // above (readCertTau()/readCertAlpha()), but min_samples is
                // deliberately not, so mining across the whole dataset can
                // use a different per-cell floor than the branch you happen
                // to be looking at.
                min_samples: minSamples,
                n_permutations_familywise: SCAN_N_PERMUTATIONS_FAMILYWISE_DISABLED,
            }),
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
        showScanStatusMsg(`${prefix}No column/value cleared U\u2090 > ${thresholdPct}% at the chosen FDR.`, 'empty');
    } else {
        showScanStatusMsg(
            `${prefix}${nVals} value(s) across ${nCols} column(s) cleared U\u2090 > ${thresholdPct}% and FDR control — Target Variable filtered.`,
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

// Last analysed (target, criterion), so a certificate change can re-run the
// same analysis without the user re-picking it from the tree.
let lastTargetCol = null;
let lastCriterion = null;
let lastFeatures = null;

function readCertTau() {
    const el = document.getElementById('certTau');
    const v = el ? Number(el.value) / 100 : 0.90;
    // 1.0 is allowed: "cells that are entirely the target value" is a
    // well-posed request about the observed table (it is rejected only in
    // strict mode, where it would be a request to PROVE exact purity).
    return (Number.isFinite(v) && v > 0 && v <= 1) ? v : 0.90;
}

function readCertAlpha() {
    const el = document.getElementById('certAlpha');
    const v = el ? Number(el.value) : 0.05;
    return (Number.isFinite(v) && v > 0 && v < 1) ? v : 0.05;
}

function readCertMinSamples() {
    const el = document.getElementById('certMinSamples');
    const v = el ? Math.round(Number(el.value)) : 1;
    return (Number.isFinite(v) && v >= 1) ? v : 1;
}

function readCertRule() {
    const el = document.getElementById('certStrict');
    return (el && el.checked) ? 'certified' : 'purity';
}

// The lower colour boundary is cosmetic: it partitions the same cells into
// the same centres and changes only which of the non-centre cells read brown
// rather than red. Re-colour in place; do not re-run the analysis.
function applyColorBoundary() {
    if (!currentPayload) return;
    syncColorScaleFromInputs();
    renderPlot(currentPayload);
}

// Re-runs the current analysis under a new certificate. tau and alpha are
// part of the server's cache key, so this is a genuine recomputation, not a
// client-side re-colouring: the certified set, the coverage and the
// cross-validated coverage all change with tau.
async function applyCertificate() {
    // #certAlpha and #certStrict were removed from Centres & Colour (this
    // view no longer offers the "certified" rule or a tunable alpha --
    // readCertRule()/readCertAlpha() degrade to 'purity'/0.05 with the
    // elements gone), so the strict/alpha validation this function used to
    // do can never fire and was removed with them. Only tau is still a
    // live control here.
    const tauEl = document.getElementById('certTau');
    const tau = tauEl ? Number(tauEl.value) : 90;
    if (!Number.isFinite(tau) || tau <= 0 || tau > 100) {
        showAnalysisError('The green boundary must be in (0, 100] percent.');
        return;
    }
    if (lastTargetCol === null) return;
    await runAnalysis(lastTargetCol, lastCriterion, lastFeatures);
}

function renderCertificateSummary(response) {
    const el = document.getElementById('certSummary');
    if (!el) return;
    el.innerHTML = '';
    if (response && response.schema && response.schema.selected_from_landscape) {
        el.textContent = 'Schema opened from the landscape: '
            + (response.schema.feature_names || []).join(' + ')
            + '. Chosen by looking at the data, so no uncorrected permutation p-value is reported for it; '
            + 'the cross-validated coverage is still out-of-sample.';
    }
}

async function runAnalysis(targetCol, criterion = null, features = null) {
    _plotlyWarmupCancelled = true; // the real render triangulates what it needs
    showWelcomeState(false);
    showLoader(true);
    showAnalysisError(null);
    lastTargetCol = targetCol;
    lastCriterion = criterion;
    lastFeatures = features; // an explicit schema opened from the landscape, or null
    try {
        const reqBody = {
            target: targetCol,
            tau: readCertTau(),
            alpha: readCertAlpha(),
            rule: readCertRule(),
            min_samples: readCertMinSamples(),
            direction: readDirection(),
        };
        if (criterion !== null) {
            reqBody.criterion = criterion;
        }
        if (features !== null) {
            reqBody.features = features;
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
    renderCertificateSummary(data);
    onAnalysisLoadedForLandscape(data);
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

// v2.3: `discover_branches` always ranks by coverage now (see
// vsf.avr's module docstring), so there is no longer a second objective a
// caption would need to distinguish -- #branchCaption was removed from
// index.html rather than left permanently hidden.
function renderBranchSelector(response) {
    const container = document.getElementById('branchList');
    if (!container) return;
    container.innerHTML = '';

    const allDims = response.branch_dims || [];
    if (allDims.length === 0) {
        container.innerHTML = '<div style="color: var(--text-dim); font-size: 0.85rem; padding: 1rem;">No branches available.</div>';
        return;
    }

    allDims.forEach(dRaw => {
        const dKey = String(dRaw);
        const branch = response.branches ? response.branches[dKey] : null;
        if (!branch || !branch.metrics) return;
        const features = (branch.selected_features || []).join(' + ');

        const card = document.createElement('div');
        card.className = 'branch-card' + (dKey === activeBranchDim ? ' active' : '');
        card.dataset.dim = dKey;
        // v2.2: the card leads with what the branch DELIVERS (certified
        // centres and the share of the target they capture) and keeps the
        // information-theoretic pair as a secondary diagnostic line. On a
        // rare target the two disagree by construction -- see
        // vsf.centers' module docstring -- and the card must not lead with
        // the number that reads 41.3% while no cell exceeds 2.42% purity.
        const sc = branch.search_centers;
        const cc = branch.centers || {};
        let headline;
        if (sc && sc.undetermined_reason) {
            headline = `<span class="branch-uadj" style="color:var(--text-dim);" title="${sc.undetermined_reason}">coverage undetermined</span>`;
        } else if (cc.n_centers) {
            headline = (activeDirection === 'absence')
                ? `<span class="branch-uadj" title="Share of ALL rows inside cells certified free of the chosen value, then the share of rows without the value that those cells capture">free ${(cc.mass * 100).toFixed(1)}% of rows · coverage ${(cc.coverage * 100).toFixed(1)}% · ${cc.n_centers} cell${cc.n_centers === 1 ? '' : 's'}</span>`
                : `<span class="branch-uadj" title="Share of all target-value samples inside certified centres">coverage ${(cc.coverage * 100).toFixed(1)}% · ${cc.n_centers} centre${cc.n_centers === 1 ? '' : 's'}</span>`;
        } else {
            const best = (cc.max_purity_lower !== undefined)
                ? ` (best lower bound ${(cc.max_purity_lower * 100).toFixed(1)}%)` : '';
            headline = `<span class="branch-uadj" style="color:var(--text-dim);" title="No cell in this branch reaches the certified purity floor.">no certified centres${best}</span>`;
        }
        // MI/E₀/U_adj and the p-value line were dropped from this card
        // (2026-09, same cleanup pass as the HUD strip's Purity/MI/U_adj/p),
        // and formatSignificance() was deleted from this file along with
        // their last call site here -- as clutter on top of the headline
        // coverage+centres number, which is already what selects and ranks
        // these branches under the coverage-only ranking (vsf.avr.discover_branches,
        // v2.3). Nothing in the UI now
        // shows the per-branch p-value: it was the only signal for whether
        // a coverage/centres number is distinguishable from chance, so a
        // branch shown here with a high coverage is not thereby shown to be
        // a real effect rather than noise -- an explicit, user-accepted
        // trade, not an oversight.
        card.innerHTML = `
            <div class="branch-card-head">
                <span class="branch-dim-badge">${dKey}D</span>
                ${headline}
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
    if (viewMode === 'landscape') refreshLandscape();
}

function showLoader(show) {
    const loader = document.getElementById('loader');
    if (show) loader.classList.add('active');
    else loader.classList.remove('active');
}

// 3-zone Purity Color Scale with USER-MOVABLE boundaries.
//
// History, because the boundaries have now moved twice. v2.0/v2.1 hard-coded
// 0.25 / 0.75 / 0.85 on the point purity, with a yellow band in between and a
// "Noise Reduction (minimum samples)" slider bolted on to hide the cell that
// held one object at 100 %. v2.2's first cut replaced all of that with
// boundaries derived from a confidence bound -- statistically defensible, and
// the wrong product: the user asked to move the boundaries themselves.
//
// What ships: three zones, no yellow, both boundaries owned by the user.
//
//   Green  -> purity >= tau            the discrete centre. tau is the same
//                                      number the panel's Coverage is computed
//                                      from, so what is green on screen and
//                                      what is counted are the same set by
//                                      construction, not by convention.
//   Brown  -> redTo <= purity < tau
//   Red    -> purity < redTo
//
// tau therefore changes the METRIC and requires a recomputation; redTo is
// cosmetic and re-colours instantly. The two are deliberately not the same
// kind of control, and the UI says so.
//
// The confidence interval of every cell is still computed and still shown in
// its hover text. It no longer decides the colour; it is there so that a cell
// which is green on one object is visibly green on one object.
const PROB_COLORS = [
    '#ef4444', // Red   - below the user's lower boundary
    '#92572e', // Brown - between the boundaries (cell colour name; the
               // control that sets this boundary is now called redTo)
    '#22c55e'  // Green - at or above tau: a discrete centre
];
// Absence search (vsf.avr.Direction): the certified zone is drawn RED --
// "certified free of the value" -- and the low zone, rows rich in the value
// but certified for nothing at this tau, slate grey. Never green: a green
// cell is a presence certificate, and the absence search does not make one.
const PROB_COLORS_ABSENCE = [
    '#64748b', // Slate - the value is present here, uncertified either way
    '#92572e', // Brown - between the boundaries
    '#ef4444'  // Red   - certified at least tau free of the value
];
const PALETTE_COUNT = PROB_COLORS.length;
const COLOR_CENTER = 2, COLOR_MIXED = 1, COLOR_LOW = 0;

// Which side of the value the current analysis searches for. Part of the
// server's cache key; switching it re-runs the analysis.
let activeDirection = 'presence';

function readDirection() {
    return activeDirection === 'absence' ? 'absence' : 'presence';
}

function activePalette() {
    return activeDirection === 'absence' ? PROB_COLORS_ABSENCE : PROB_COLORS;
}

// Switches the search direction: relabels the legend, the sliders and the
// HUD, then re-runs the current analysis (the response is a different
// cache entry) or, before any analysis, just arms the choice for the first
// click. The legend text must change here rather than in the render: the
// same slider means "green from" under presence and "certified free from"
// under absence, and a stale label would describe the wrong quantity.
async function setDirection(direction) {
    direction = direction === 'absence' ? 'absence' : 'presence';
    const changed = direction !== activeDirection;
    activeDirection = direction;
    document.body.classList.toggle('absence-mode', direction === 'absence');
    document.querySelectorAll('#directionToggle .direction-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.direction === direction);
    });
    const labelTau = document.getElementById('labelTau');
    const labelRed = document.getElementById('labelRedTo');
    if (labelTau) {
        labelTau.textContent = direction === 'absence' ? '🔴 Free from:' : '🟢 Green from:';
        labelTau.style.color = direction === 'absence' ? '#ef4444' : 'var(--green)';
        labelTau.title = direction === 'absence'
            ? 'A cell is certified FREE of the chosen value — and is drawn red — when the share of rows WITHOUT the value reaches this. Moving this re-runs the analysis.'
            : "A cell is a discrete centre — and is drawn green — when its share of the target value reaches this. Coverage is computed from exactly these cells, so moving this re-runs the analysis. 100% is allowed and means 'only cells that are entirely the target value'.";
    }
    if (labelRed) {
        labelRed.textContent = direction === 'absence' ? '⚪ Grey up to:' : '🔴 Red up to:';
        labelRed.style.color = direction === 'absence' ? '#94a3b8' : '#ef4444';
        labelRed.title = direction === 'absence'
            ? 'Purely a colour boundary: cells whose share of rows without the value is below it are drawn grey (the value is present there); between it and the red boundary, brown. Re-colours immediately, changes no reported number.'
            : 'Purely a colour boundary: cells below it are drawn red; between it and the green boundary, brown. Changing it re-colours immediately and does not affect any reported number.';
    }
    const hudFree = document.getElementById('hud-free');
    if (hudFree) hudFree.style.display = direction === 'absence' ? 'flex' : 'none';
    const labelCov = document.getElementById('label-coverage');
    if (labelCov) labelCov.textContent = direction === 'absence' ? 'Coverage (rows without value):' : 'Coverage:';
    updateBaseRateNote();
    if (changed && lastTargetCol !== null) {
        if (lastCriterion === null) {
            showAnalysisError(direction === 'absence'
                ? 'An absence search needs a specific target value: pick one in the catalog.'
                : null);
            return;
        }
        await runAnalysis(lastTargetCol, lastCriterion, lastFeatures);
    }
}

// The base rate the current search is measured against, and the tau it
// therefore needs. `activePrevalence` is the prevalence of the indicator
// SEARCHED (the value under presence, its complement under absence).
function updateBaseRateNote() {
    const el = document.getElementById('baseRateNote');
    if (!el) return;
    if (!currentPayload || !currentPayload.centers) { el.textContent = ''; return; }
    const pInd = activePrevalence;
    const pValue = activeDirection === 'absence' ? 1 - pInd : pInd;
    const need = activeDirection === 'absence'
        ? `an absence search must use tau above ${(pInd * 100).toFixed(1)}% (the share of rows without it)`
        : `a presence search must use tau above ${(pInd * 100).toFixed(1)}%`;
    el.textContent = `Base rate of the chosen value: ${(pValue * 100).toFixed(1)}% of rows; ${need}.`
        + (pValue > 0.5 && activeDirection === 'presence'
            ? ' The value is more common than not — consider searching for its absence.' : '');
}

// Certificate parameters currently in force, mirrored from the payload so the
// colour function, the legend and the panel can never disagree about tau.
let activeTau = 0.90;
let activeAlpha = 0.05;
let activeRule = 'purity';
let activeMinSamples = 1;
let activePrevalence = 0.0;

function readRedTo() {
    const el = document.getElementById('colorRedTo');
    const v = el ? Number(el.value) / 100 : 0.40;
    return (Number.isFinite(v) && v >= 0 && v <= 1) ? v : 0.40;
}

function getColorIndexForCell(purity, isCenter, tau, redTo) {
    // `isCenter` comes from the server and is authoritative: under the strict
    // rule a cell can sit above tau by point purity and still not be a centre,
    // and the colour must follow the metric rather than re-deriving it here.
    if (isCenter === true) return COLOR_CENTER;
    if (isCenter === false && purity >= tau) return COLOR_MIXED;
    if (purity >= tau) return COLOR_CENTER;
    if (purity >= redTo) return COLOR_MIXED;
    return COLOR_LOW;
}

// ---------------------------------------------------------------------------
// Colour-boundary slider (now part of the Global Pattern Scan panel --
// it used to be its own "Centres & Colour" section, merged in by request
// since tau/alpha are exactly what the scan certifies against too).
//
// Replaces the old renderCertificateLegend()/#purityLegend static legend: the
// gradient track *is* the legend now (drawn straight from the live tau /
// redTo values via CSS custom properties), and its two handles are the
// primary way to move the boundaries. The #certTau / #colorRedTo number
// inputs remain the source of truth read by readCertTau()/readRedTo()
// and by applyCertificate(); the slider only ever reads and writes those
// same inputs, so the two controls can never disagree.
//
// tau (green) and redTo (red) are asymmetric in cost, and the slider
// preserves that asymmetry rather than hiding it:
//   - dragging the RED handle re-colours instantly (cosmetic, no request);
//   - dragging the GREEN handle updates the displayed number live but only
//     commits — i.e. calls applyCertificate() and re-runs the analysis —
//     when the handle is released (mouseup/touchend/blur), because tau is
//     part of the server's cache key and changes the reported metric.
// ---------------------------------------------------------------------------
const COLOR_SCALE_MIN_GAP = 1; // percentage points; keeps the handles from crossing

function clampPct(value, lo, hi) {
    return Math.min(hi, Math.max(lo, value));
}

// Positions the two handles and repaints the gradient from two percentages
// already known to be valid (0 <= redPct < tauPct <= 100). Pure DOM/CSS
// update: never reads or writes the number inputs itself.
function setColorScaleUI(redPct, tauPct) {
    const track = document.getElementById('colorScaleTrack');
    const handleRed = document.getElementById('handleRed');
    const handleGreen = document.getElementById('handleGreen');
    const redValueEl = document.getElementById('handleRedValue');
    const greenValueEl = document.getElementById('handleGreenValue');
    if (!track || !handleRed || !handleGreen) return;
    track.style.setProperty('--red-pct', `${redPct}%`);
    track.style.setProperty('--tau-pct', `${tauPct}%`);
    handleRed.style.left = `${redPct}%`;
    handleGreen.style.left = `${tauPct}%`;
    handleRed.setAttribute('aria-valuenow', String(Math.round(redPct)));
    handleGreen.setAttribute('aria-valuenow', String(Math.round(tauPct)));
    if (redValueEl) redValueEl.textContent = `${Math.round(redPct)}%`;
    if (greenValueEl) greenValueEl.textContent = `${Math.round(tauPct)}%`;
}

// Reflects the current #certTau / #colorRedTo input values onto the
// slider. Call this whenever those inputs change from ANY source (typed by
// hand, a payload reload, dragging the other handle) so the slider can never
// show a stale position.
function syncColorScaleFromInputs() {
    const tauEl = document.getElementById('certTau');
    const redEl = document.getElementById('colorRedTo');
    const tau = tauEl ? clampPct(Number(tauEl.value), 1, 100) : 90;
    const red = redEl ? clampPct(Number(redEl.value), 0, 99) : 40;
    setColorScaleUI(Math.min(red, tau - COLOR_SCALE_MIN_GAP), tau);
}

let colorScaleDragTarget = null; // 'red' | 'green' | null

function colorScalePctFromEvent(evt) {
    const track = document.getElementById('colorScaleTrack');
    if (!track) return 0;
    const rect = track.getBoundingClientRect();
    const point = (evt.touches && evt.touches[0]) ? evt.touches[0] : evt;
    const pct = ((point.clientX - rect.left) / rect.width) * 100;
    return clampPct(Math.round(pct), 0, 100);
}

function onColorScalePointerDown(which) {
    return function (evt) {
        evt.preventDefault();
        colorScaleDragTarget = which;
        document.body.style.userSelect = 'none';
    };
}

function onColorScalePointerMove(evt) {
    if (!colorScaleDragTarget) return;
    evt.preventDefault();
    const tauEl = document.getElementById('certTau');
    const redEl = document.getElementById('colorRedTo');
    const tau = tauEl ? Number(tauEl.value) : 90;
    const red = redEl ? Number(redEl.value) : 40;
    const pct = colorScalePctFromEvent(evt);

    if (colorScaleDragTarget === 'red') {
        const next = clampPct(pct, 0, tau - COLOR_SCALE_MIN_GAP);
        if (redEl) redEl.value = String(next);
        setColorScaleUI(next, tau);
        applyColorBoundary(); // cosmetic: re-colour live while dragging
    } else if (colorScaleDragTarget === 'green') {
        const next = clampPct(pct, Math.max(1, red + COLOR_SCALE_MIN_GAP), 100);
        if (tauEl) tauEl.value = String(next);
        setColorScaleUI(red, next);
        // Deliberately NOT recomputed while dragging — see file header.
    }
}

function onColorScalePointerUp() {
    if (!colorScaleDragTarget) return;
    const wasGreen = colorScaleDragTarget === 'green';
    colorScaleDragTarget = null;
    document.body.style.userSelect = '';
    if (wasGreen) applyCertificate(); // commit: re-run the analysis at the new tau
}

// Arrow-key nudge for the focused handle (1 point; Shift = 5 points), so the
// boundaries stay operable without a mouse or touch.
function onColorScaleKeyDown(which) {
    return function (evt) {
        const step = evt.shiftKey ? 5 : 1;
        let delta = 0;
        if (evt.key === 'ArrowLeft' || evt.key === 'ArrowDown') delta = -step;
        else if (evt.key === 'ArrowRight' || evt.key === 'ArrowUp') delta = step;
        else return;
        evt.preventDefault();

        const tauEl = document.getElementById('certTau');
        const redEl = document.getElementById('colorRedTo');
        const tau = tauEl ? Number(tauEl.value) : 90;
        const red = redEl ? Number(redEl.value) : 40;

        if (which === 'red') {
            const next = clampPct(red + delta, 0, tau - COLOR_SCALE_MIN_GAP);
            if (redEl) redEl.value = String(next);
            setColorScaleUI(next, tau);
            applyColorBoundary();
        } else {
            const next = clampPct(tau + delta, Math.max(1, red + COLOR_SCALE_MIN_GAP), 100);
            if (tauEl) tauEl.value = String(next);
            setColorScaleUI(red, next);
            applyCertificate();
        }
    };
}

// Wires the drag/keyboard handlers once at startup and paints the initial
// handle positions from whatever the number inputs already hold.
function initColorScale() {
    const handleRed = document.getElementById('handleRed');
    const handleGreen = document.getElementById('handleGreen');
    if (!handleRed || !handleGreen) return;

    handleRed.addEventListener('mousedown', onColorScalePointerDown('red'));
    handleRed.addEventListener('touchstart', onColorScalePointerDown('red'), { passive: false });
    handleGreen.addEventListener('mousedown', onColorScalePointerDown('green'));
    handleGreen.addEventListener('touchstart', onColorScalePointerDown('green'), { passive: false });

    window.addEventListener('mousemove', onColorScalePointerMove);
    window.addEventListener('touchmove', onColorScalePointerMove, { passive: false });
    window.addEventListener('mouseup', onColorScalePointerUp);
    window.addEventListener('touchend', onColorScalePointerUp);

    handleRed.addEventListener('keydown', onColorScaleKeyDown('red'));
    handleGreen.addEventListener('keydown', onColorScaleKeyDown('green'));

    syncColorScaleFromInputs();
}

function formatCoverage(payload, d) {
    // Centre statistics for the CURRENTLY VIEWED dimensionality, from the
    // displayed partition itself (vsf.vis `view_metrics.*_by_d`). Falls back
    // to the branch-level block for a payload that predates v2.2.
    const vm = payload.view_metrics || {};
    const key = String(d);
    const pick = (obj, fallback) =>
        (obj && obj[key] !== undefined && obj[key] !== null) ? obj[key] : fallback;
    const c = payload.centers || {};
    return {
        coverage: pick(vm.coverage_by_d, c.coverage),
        n_centers: pick(vm.n_centers_by_d, c.n_centers),
        purity: pick(vm.purity_by_d, c.purity_pooled),
        max_lower: pick(vm.max_purity_lower_by_d, c.max_purity_lower),
        mass: pick(vm.mass_by_d, c.mass)
    };
}

function updateDashboard(payload) {
    currentPayload = payload;
    const m = payload.metrics;
    const cert = payload.certificate || {};
    activeTau = (cert.tau !== undefined && cert.tau !== null) ? cert.tau : 0.90;
    activeAlpha = (cert.alpha !== undefined && cert.alpha !== null) ? cert.alpha : 0.05;
    activeRule = cert.rule || 'purity';
    activeMinSamples = (cert.min_samples !== undefined && cert.min_samples !== null)
        ? cert.min_samples : 1;
    activePrevalence = (payload.centers && payload.centers.prevalence !== undefined)
        ? payload.centers.prevalence : 0.0;
    if (currentBranchesResponse && currentBranchesResponse.direction
        && currentBranchesResponse.direction !== activeDirection) {
        // A response computed under the other direction (e.g. loaded from a
        // stale state): make the legend describe what is on screen.
        activeDirection = currentBranchesResponse.direction;
        setDirection(activeDirection);
    }
    updateBaseRateNote();
    syncColorScaleFromInputs();

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

    // v2.2: `total_samples` is the N every reported statistic is computed on;
    // `rendered_samples` is how many of those rows are drawn as spheres. The
    // two were silently conflated before, so the panel showed 10 000 beside
    // numbers computed on 32 561 rows.
    const nStat = payload.total_samples || (payload.x ? payload.x.length : 0);
    const nDrawn = (payload.rendered_samples !== undefined && payload.rendered_samples !== null)
        ? payload.rendered_samples : nStat;
    const samplesEl = document.getElementById('totalSamplesVal');
    if (samplesEl) {
        samplesEl.innerText = (nDrawn < nStat)
            ? `${nStat.toLocaleString()} (${nDrawn.toLocaleString()} drawn)`
            : nStat.toLocaleString();
        samplesEl.title = (nDrawn < nStat)
            ? `All ${nStat.toLocaleString()} rows are used for every statistic and every cell colour; ${nDrawn.toLocaleString()} of them are drawn as individual spheres.`
            : 'Every row is both counted and drawn.';
    }

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

    // HUD strip intentionally shows only Coverage and Centres (trimmed from
    // 7 chips on user request: Purity, MI raw/E₀, U_adj, the raw p-value
    // chip and the "Viewing" dimensionality label were judged clutter for
    // this always-visible strip). renderBranchSelector's per-branch cards
    // went through the same cleanup in the same pass, so MI/U_adj/p are no
    // longer shown there either -- there is no remaining place in this UI
    // that surfaces them. Which `d` Coverage/Centres refer to is no longer
    // stated in text next to them; it is still visible from
    // dimButtonsGroup's active-button state, just not textually paired
    // with the values anymore — a deliberate trade the user accepted.
    const cov = formatCoverage(currentPayload, d);
    const sc = currentPayload.search_centers;
    const freeEl = document.getElementById('val-free');
    if (freeEl) {
        freeEl.innerText = (cov.mass === undefined || cov.mass === null)
            ? 'n/a' : `${(cov.mass * 100).toFixed(1)}%`;
        freeEl.style.color = (cov.mass > 0) ? '#ef4444' : 'var(--text-dim)';
    }
    const covEl = document.getElementById('val-coverage');
    if (covEl) {
        if (cov.coverage === undefined || cov.coverage === null) {
            covEl.innerText = 'n/a';
            covEl.style.color = 'var(--text-dim)';
            covEl.title = 'The target has more than two values and no positive value was declared, so cell purity is undefined.';
        } else if (sc && sc.undetermined_reason) {
            covEl.innerText = 'undetermined';
            covEl.style.color = 'var(--text-dim)';
            covEl.title = sc.undetermined_reason;
        } else {
            covEl.innerText = `${(cov.coverage * 100).toFixed(1)}%`;
            covEl.style.color = cov.coverage > 0 ? 'var(--green)' : 'var(--text-dim)';
            covEl.title = activeDirection === 'absence'
                ? 'Share of all rows WITHOUT the chosen value that fall inside cells certified free of it.'
                : 'Share of all target-value samples that fall inside certified centres.';
        }
    }
    const kEl = document.getElementById('val-centers');
    if (kEl) {
        kEl.innerText = (cov.n_centers === undefined || cov.n_centers === null)
            ? 'n/a' : String(cov.n_centers);
        kEl.title = (cov.n_centers === 0 && cov.max_lower !== undefined && cov.max_lower !== null)
            ? `No cell reaches the certified purity floor. The highest lower bound anywhere in this view is ${(cov.max_lower * 100).toFixed(1)}%, against tau = ${(activeTau * 100).toFixed(0)}%.`
            : 'Number of cells certified to be at least tau pure in the target value. Fewer is better at equal coverage.';
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
    let currentCertified = g ? g.certified : payload.grid_certified;
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
        const isCenter = (currentCertified && currentCertified[i] !== undefined)
            ? currentCertified[i] : undefined;

        const colorIndex = getColorIndexForCell(p, isCenter, activeTau, readRedTo());

        fx.push(xCoords[i]);
        fy.push(yCoords[i]);
        fz.push(zCoords[i]);

        const hex = activePalette()[colorIndex];
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
        header.setAttribute('aria-expanded', 'false');
    } else {
        header.classList.add('open');
        header.setAttribute('aria-expanded', 'true');
        content.classList.add('open');
    }
}

window.addEventListener('DOMContentLoaded', init);


// ===========================================================================
// Solution landscape (Project_Master_Document.md Section 4.9)
// ===========================================================================
// Every schema the exhaustive search scored, drawn as a 10 x 10 lattice of
// categories: x = coverage of the value (absence: mass of certified-free
// cells) in (0,10], (10,20], ..., (90,100] percent; y = the schema's centre
// count as a share of K max, the largest centre count among the schemas
// shown, in the same categories. A cell's colour is how many schemas fall
// in it, under a three-zone scale the user moves; schemas certifying no
// centre are excluded from the lattice and counted in the header. Clicking
// a cell lists its schemas (most concentrated first); "open" renders one.
let viewMode = 'lattice';
let landscapeState = {
    key: null,          // JSON of the analyze parameters the landscape belongs to
    bins: null,         // /api/landscape response for the current d selection
    dSelection: null,   // the d requested (number) or null for all
    cell: null,         // {ix, iy} of the open cell
    cellData: null,     // /api/landscape/cell response
};

function landscapeParams() {
    if (!currentBranchesResponse) return null;
    const p = {
        target: currentBranchesResponse.target,
        criterion: currentBranchesResponse.criterion,
        tau: readCertTau(), alpha: readCertAlpha(), rule: readCertRule(),
        min_samples: readCertMinSamples(), direction: readDirection(),
    };
    return p;
}

function onAnalysisLoadedForLandscape(data) {
    // A new analysis of a different target/certificate invalidates the
    // cached landscape; opening a schema of the same target keeps it, so
    // the user can return to the same cell.
    const p = landscapeParams();
    const key = p ? JSON.stringify(p) : null;
    if (key !== landscapeState.key) {
        landscapeState = { key, bins: null, dSelection: null, cell: null, cellData: null };
    }
    if (data && data.schema && data.schema.selected_from_landscape) {
        setViewMode('lattice');
    } else if (viewMode === 'landscape') {
        refreshLandscape();
    }
}

function setViewMode(mode) {
    viewMode = mode === 'landscape' ? 'landscape' : 'lattice';
    document.querySelectorAll('.view-mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === viewMode);
    });
    const plot = document.getElementById('plot-container');
    const area = document.getElementById('landscapeArea');
    const slices = document.getElementById('slice-controller');
    if (plot) plot.style.display = viewMode === 'landscape' ? 'none' : '';
    if (area) area.style.display = viewMode === 'landscape' ? 'flex' : 'none';
    if (slices && viewMode === 'landscape') slices.style.display = 'none';
    if (viewMode === 'landscape') {
        refreshLandscape();
    } else if (currentPayload) {
        currentRenderedDim = null;
        renderPlot(currentPayload);
        if (currentPayload.slice_axis && slices) slices.style.display = '';
    }
}

function landscapeDimRequested() {
    const sel = document.getElementById('landscapeDim');
    if (sel && sel.value === 'all') return null;
    const d = currentPayload && currentPayload.metrics ? Number(currentPayload.metrics.d) : null;
    return Number.isFinite(d) ? d : null;
}

function onLandscapeDimChange() {
    landscapeState.cell = null;
    landscapeState.cellData = null;
    closeLandscapeCell();
    refreshLandscape();
}

async function refreshLandscape() {
    const p = landscapeParams();
    if (!p) return;
    const d = landscapeDimRequested();
    if (landscapeState.bins && landscapeState.dSelection === d && landscapeState.key === JSON.stringify(p)) {
        renderLandscape();
        return;
    }
    try {
        const res = await fetch('/api/landscape', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.assign({ d }, p)),
        });
        const data = await res.json();
        if (!res.ok) { showAnalysisError('Landscape: ' + (data.error || res.status)); return; }
        landscapeState.key = JSON.stringify(p);
        landscapeState.bins = data;
        landscapeState.dSelection = d;
        renderLandscape();
        if (landscapeState.cell) openLandscapeCell(landscapeState.cell.ix, landscapeState.cell.iy, 0);
    } catch (err) {
        showAnalysisError('Landscape request failed: ' + err.message);
    }
}

// Category index of a fraction in (0, 1] under (0,1/n], ..., ((n-1)/n, 1]:
// the same rule as vsf.avr.Landscape.bin_index, so the winner marker lands
// in the cell the server counted it in.
function landscapeBinIndex(fraction, n) {
    const idx = Math.ceil(fraction * n - 1e-9) - 1;
    return Math.max(0, Math.min(n - 1, idx));
}

function densityBoundaries() {
    const lowEl = document.getElementById('densityLow');
    const highEl = document.getElementById('densityHigh');
    let low = lowEl ? Math.max(0, Math.round(Number(lowEl.value))) : 10;
    let high = highEl ? Math.max(0, Math.round(Number(highEl.value))) : 40;
    if (!Number.isFinite(low)) low = 10;
    if (!Number.isFinite(high)) high = 40;
    if (high < low) high = low;
    return { low, high };
}

const DENSITY_COLORS = ['#334155', '#6366f1', '#e0e7ff']; // faint, mid, dark(bright)

function renderLandscape() {
    const bins = landscapeState.bins;
    const container = document.getElementById('landscape-plot');
    if (!bins || !container) return;
    const n = bins.n_bins;
    const { low, high } = densityBoundaries();
    const maxCount = bins.max_count || 0;
    const bar = document.getElementById('densityBar');
    if (bar) {
        const pct = (v) => maxCount > 0 ? Math.max(0, Math.min(100, 100 * v / maxCount)) : 0;
        bar.style.setProperty('--low-pct', `${pct(low)}%`);
        bar.style.setProperty('--high-pct', `${pct(high)}%`);
    }
    const maxEl = document.getElementById('densityMax');
    if (maxEl) maxEl.textContent = `max ${maxCount}`;

    const summary = document.getElementById('landscapeSummary');
    if (summary) {
        const shown = bins.n_total - bins.n_zero;
        const scope = bins.d === null || bins.d === undefined ? 'all dimensionalities' : `${bins.d}D`;
        summary.innerHTML = `<strong>${shown}</strong> of <strong>${bins.n_total}</strong> schemas (${scope}) certify at least one `
            + (bins.direction === 'absence' ? 'value-free cell' : 'centre')
            + ` at tau = ${(activeTau * 100).toFixed(0)}%; <strong>${bins.n_zero}</strong> certify none and are not drawn. K max = <strong>${bins.k_max}</strong>.`;
    }

    const xLabels = [], yLabels = [];
    for (let i = 0; i < n; i++) {
        xLabels.push(`(${i * 100 / n},${(i + 1) * 100 / n}]%`);
        const lo = Math.floor(i * bins.k_max / n), hi = Math.ceil((i + 1) * bins.k_max / n);
        yLabels.push(`(${lo},${hi}]`);
    }
    const xs = [], ys = [], colors = [], texts = [], hovers = [], customs = [];
    for (let iy = 0; iy < n; iy++) {
        for (let ix = 0; ix < n; ix++) {
            const c = bins.counts[iy][ix];
            if (!c) continue;
            xs.push(ix); ys.push(iy);
            colors.push(c > high ? DENSITY_COLORS[2] : (c > low ? DENSITY_COLORS[1] : DENSITY_COLORS[0]));
            texts.push(String(c));
            hovers.push(`${c} schema${c === 1 ? '' : 's'}<br>${bins.x}: ${xLabels[ix]}<br>centres: ${yLabels[iy]} of K max ${bins.k_max}<br><i>click to list</i>`);
            customs.push([ix, iy]);
        }
    }
    const isBright = colors.map(col => col === DENSITY_COLORS[2]);
    const traces = [{
        type: 'scatter', mode: 'markers+text', x: xs, y: ys, text: texts, textposition: 'middle center',
        textfont: { size: 11, color: isBright.map(b => b ? '#0f172a' : '#f8fafc') },
        marker: { size: 34, color: colors, line: { width: 1, color: 'rgba(255,255,255,0.25)' } },
        hovertext: hovers, hoverinfo: 'text', customdata: customs, name: 'schemas',
    }];
    // Winner of the selected branch, from the SEARCH partition's exact
    // numbers (`search_centers`): the landscape is the search's family, and
    // a capacity-coarsened winner can differ from the drawn (raw) lattice's
    // `centers` block.
    const cc = currentPayload && (currentPayload.search_centers || currentPayload.centers);
    if (cc && cc.n_centers > 0 && bins.k_max > 0
        && (bins.d === null || bins.d === undefined || Number(bins.d) === Number(currentPayload.metrics.d))) {
        const xFrac = bins.x === 'mass' ? cc.mass : cc.coverage;
        const wx = landscapeBinIndex(xFrac, n), wy = landscapeBinIndex(cc.n_centers / bins.k_max, n);
        traces.push({
            type: 'scatter', mode: 'markers', x: [wx], y: [wy],
            marker: { symbol: 'star', size: 16, color: '#facc15', line: { width: 1, color: '#0f172a' } },
            hovertext: [`search winner (${currentPayload.metrics.d}D): ${(currentPayload.selected_features || []).join(' + ')}<br>${bins.x} ${(xFrac * 100).toFixed(1)}% · ${cc.n_centers} centres`],
            hoverinfo: 'text', name: 'winner',
        });
    }
    if (landscapeState.cell) {
        traces.push({
            type: 'scatter', mode: 'markers', x: [landscapeState.cell.ix], y: [landscapeState.cell.iy],
            marker: { symbol: 'circle-open', size: 44, color: '#f8fafc', line: { width: 2 } },
            hoverinfo: 'skip', name: 'selected',
        });
    }
    const layout = {
        paper_bgcolor: '#070a13', plot_bgcolor: '#090d1a', showlegend: false,
        margin: { l: 90, r: 20, t: 20, b: 70 }, font: { family: 'Inter', color: '#94a3b8' },
        xaxis: {
            title: { text: bins.x === 'mass' ? 'Mass of certified value-free cells' : 'Coverage of the value', font: { color: '#c084fc', size: 13 } },
            tickvals: xLabels.map((_, i) => i), ticktext: xLabels, range: [-0.6, n - 0.4],
            tickfont: { size: 10 }, showgrid: true, gridcolor: 'rgba(255,255,255,0.06)', zeroline: false, fixedrange: true,
        },
        yaxis: {
            title: { text: `Certified centres (of K max = ${bins.k_max})`, font: { color: '#c084fc', size: 13 } },
            tickvals: yLabels.map((_, i) => i), ticktext: yLabels, range: [-0.6, n - 0.4],
            tickfont: { size: 10 }, showgrid: true, gridcolor: 'rgba(255,255,255,0.06)', zeroline: false, fixedrange: true,
        },
    };
    Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false });
    container.removeAllListeners && container.removeAllListeners('plotly_click');
    container.on('plotly_click', (ev) => {
        const pt = ev.points && ev.points[0];
        if (!pt || !pt.customdata) return;
        openLandscapeCell(pt.customdata[0], pt.customdata[1], 0);
    });
}

async function openLandscapeCell(ix, iy, offset) {
    const p = landscapeParams();
    if (!p || !landscapeState.bins) return;
    const limit = 50;
    try {
        const res = await fetch('/api/landscape/cell', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.assign({ d: landscapeState.dSelection, ix, iy, limit, offset }, p)),
        });
        const data = await res.json();
        if (!res.ok) { showAnalysisError('Landscape cell: ' + (data.error || res.status)); return; }
        landscapeState.cell = { ix, iy };
        landscapeState.cellData = data;
        renderLandscapeCell();
        renderLandscape();
    } catch (err) {
        showAnalysisError('Landscape cell request failed: ' + err.message);
    }
}

function closeLandscapeCell() {
    landscapeState.cell = null;
    landscapeState.cellData = null;
    const panel = document.getElementById('landscapeCellPanel');
    if (panel) panel.style.display = 'none';
    if (viewMode === 'landscape' && landscapeState.bins) renderLandscape();
}

function renderLandscapeCell() {
    const panel = document.getElementById('landscapeCellPanel');
    const list = document.getElementById('landscapeCellList');
    const more = document.getElementById('landscapeCellMore');
    const title = document.getElementById('landscapeCellTitle');
    const data = landscapeState.cellData;
    const bins = landscapeState.bins;
    if (!panel || !list || !data || !bins) return;
    panel.style.display = '';
    const n = bins.n_bins;
    const ix = landscapeState.cell.ix, iy = landscapeState.cell.iy;
    if (title) {
        title.textContent = `${data.total} schema${data.total === 1 ? '' : 's'} with ${bins.x} in (${ix * 100 / n},${(ix + 1) * 100 / n}]% and `
            + `${Math.floor(iy * bins.k_max / n)} < centres ≤ ${Math.ceil((iy + 1) * bins.k_max / n)} — most concentrated first`;
    }
    const currentFeats = (currentBranchesResponse && currentBranchesResponse.schema)
        ? currentBranchesResponse.schema.features.slice().sort().join(',')
        : (currentPayload && currentPayload.selected_feature_indices ? currentPayload.selected_feature_indices.slice().sort().join(',') : null);
    list.innerHTML = '';
    data.schemas.forEach(sc => {
        const row = document.createElement('div');
        row.className = 'landscape-row' + (currentFeats === sc.features.slice().sort().join(',') ? ' current' : '');
        const xv = bins.x === 'mass' ? sc.mass : sc.coverage;
        row.innerHTML = `
            <span class="axes">${sc.feature_names.join(' + ')}</span>
            <span class="num">${bins.x} ${(xv * 100).toFixed(2)}%</span>
            <span class="num">${sc.n_centers} centre${sc.n_centers === 1 ? '' : 's'}</span>
            <span class="num">mass ${(sc.mass * 100).toFixed(2)}%</span>
            <button type="button" class="landscape-open">open</button>`;
        row.querySelector('.landscape-open').onclick = () => openSchemaFromLandscape(sc.features);
        list.appendChild(row);
    });
    if (more) {
        more.innerHTML = '';
        const shown = data.offset + data.schemas.length;
        more.textContent = `${data.offset + 1}–${shown} of ${data.total}`;
        if (data.offset > 0) {
            const b = document.createElement('button'); b.className = 'landscape-open'; b.textContent = '← previous';
            b.onclick = () => openLandscapeCell(ix, iy, Math.max(0, data.offset - data.limit)); more.appendChild(b);
        }
        if (shown < data.total) {
            const b = document.createElement('button'); b.className = 'landscape-open'; b.textContent = 'next →';
            b.onclick = () => openLandscapeCell(ix, iy, data.offset + data.limit); more.appendChild(b);
        }
    }
}

async function openSchemaFromLandscape(features) {
    if (lastTargetCol === null) return;
    await runAnalysis(lastTargetCol, lastCriterion, features.slice());
}
