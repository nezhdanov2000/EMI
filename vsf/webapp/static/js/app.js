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
            // The dataset screen (Section 4.12) belongs on the guide screen:
            // it is read before a target is chosen.
            refreshScreen(false);
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
// that information: the scan selects on the quantity the display delivers
// (coverage), and the label names the same one.
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
            body: JSON.stringify(screenRequestOptions({
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
            })),
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

        // Values arrive ordered by share (server, catalog_from_dataframe),
        // so the donuts read as one decreasing series, and the tau divider
        // (updateCatalogTauDividers) splits the list into "absence only"
        // above and "either direction" below.
        col.criteria.forEach(crit => {
            const critItem = document.createElement('div');
            critItem.className = 'crit-item';
            if (typeof crit.share === 'number') critItem.dataset.share = String(crit.share);

            const critHeader = document.createElement('div');
            critHeader.className = 'crit-header';
            const share = (typeof crit.share === 'number') ? crit.share : null;
            critItem.dataset.col = String(col.id);
            critItem.dataset.val = String(crit.id);
            critHeader.innerHTML = `<span class="crit-label">${crit.label}</span>`
                + (share === null ? '' : `<span class="crit-share" title="${crit.count.toLocaleString()} rows — ${(share * 100).toFixed(1)}% of all rows">${shareDonutSVG(share)}<span class="crit-share-pct">${formatSharePct(share)}</span></span>`)
                + `<button type="button" class="crit-add" title="Add to the target: objects must have this value AND the target's other values (the column leaves the feature space)">+</button>`;

            critHeader.onclick = async (e) => {
                e.stopPropagation();
                await pickPrimaryTarget(col.id, crit.id);
            };
            critHeader.querySelector('.crit-add').onclick = async (e) => {
                e.stopPropagation();
                await addTargetConjunct(col.id, crit.id);
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
    updateCatalogTauDividers();
    markCatalogTarget();
}

// ---------------------------------------------------------------------------
// Composite target: a conjunction of (column, value) pairs (Section 4.10)
// ---------------------------------------------------------------------------
// The primary pair is (lastTargetCol, lastCriterion); `targetAlso` holds
// the further conjuncts. The indicator searched is the AND of all pairs -
// one more 0/1 column to the search - and every column of the target is
// removed from the feature space by the server (its own columns would
// otherwise be "found" as the schema). At most three pairs: base rates
// fall multiplicatively, and beyond that certified cells are single
// objects. No automatic enumeration of conjunctions exists, by design: the
// user names the target, the search finds where it concentrates.
let targetAlso = [];          // [{col, val}] beyond the primary pair
let targetInfoCache = null;   // last /api/target response for the current conjunction
const MAX_TARGET_CONJUNCTS = 3;

function targetAlsoPairs() {
    return targetAlso.map(p => [p.col, p.val]);
}

function catalogLabelOf(colId, critId) {
    const col = (allColumnsData || []).find(c => c.id === colId);
    const crit = col && (col.criteria || []).find(c => String(c.id) === String(critId));
    return { col: col ? col.label : String(colId), val: crit ? crit.label : String(critId) };
}

async function pickPrimaryTarget(colId, critId) {
    // A column can appear once in the target: picking a value of a column
    // that is currently a conjunct replaces that conjunct.
    targetAlso = targetAlso.filter(p => p.col !== colId);
    await runAnalysis(colId, critId);
}

async function addTargetConjunct(colId, critId) {
    if (lastTargetCol === null || lastCriterion === null) {
        await pickPrimaryTarget(colId, critId);   // the first pick is the primary
        return;
    }
    if (colId === lastTargetCol) {
        showAnalysisError(`${catalogLabelOf(colId, critId).col} is already the target's column: pick a value of it to replace the primary pair.`);
        return;
    }
    const existing = targetAlso.find(p => p.col === colId);
    if (existing && String(existing.val) === String(critId)) {
        showAnalysisError('This pair is already part of the target.');
        return;
    }
    if (!existing && targetAlso.length + 1 >= MAX_TARGET_CONJUNCTS) {
        showAnalysisError(`A target may have at most ${MAX_TARGET_CONJUNCTS} (column, value) pairs.`);
        return;
    }
    targetAlso = targetAlso.filter(p => p.col !== colId).concat([{ col: colId, val: String(critId) }]);
    await runAnalysis(lastTargetCol, lastCriterion);
}

async function removeTargetConjunct(colId) {
    targetAlso = targetAlso.filter(p => p.col !== colId);
    if (lastTargetCol !== null) await runAnalysis(lastTargetCol, lastCriterion);
}

async function removePrimaryTarget() {
    // The first conjunct becomes the primary; with none left, the target is
    // cleared and the catalog waits for a pick.
    if (targetAlso.length === 0) return;
    const next = targetAlso.shift();
    await runAnalysis(next.col, next.val);
}

// The catalog's highlighting: the primary pair is `active`, the further
// conjuncts are `conjunct`.
function markCatalogTarget() {
    document.querySelectorAll('#catalogAccordion .crit-item').forEach(item => {
        const isPrimary = lastTargetCol !== null && item.dataset.col === String(lastTargetCol)
            && String(item.dataset.val) === String(lastCriterion);
        const isConjunct = targetAlso.some(p => p.col === item.dataset.col && String(p.val) === String(item.dataset.val));
        item.classList.toggle('active', isPrimary);
        item.classList.toggle('conjunct', isConjunct);
    });
}

// What the conjunction IS before the search: rows, share, features left.
// Cheap on the server (one boolean mask); called on every change.
async function fetchTargetInfo(targetCol, criterion) {
    const body = screenRequestOptions({ target: targetCol, criterion });
    if (targetAlso.length) body.also = targetAlsoPairs();
    const res = await fetch('/api/target', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || String(res.status));
    return data;
}

function renderTargetPanel(info) {
    const panel = document.getElementById('targetPanel');
    const chips = document.getElementById('targetChips');
    const infoEl = document.getElementById('targetInfo');
    if (!panel || !chips || !infoEl) return;
    if (lastTargetCol === null || lastCriterion === null) { panel.style.display = 'none'; return; }
    panel.style.display = '';
    chips.innerHTML = '';
    const pairs = [{ col: lastTargetCol, val: lastCriterion, primary: true }].concat(targetAlso.map(p => Object.assign({}, p, { primary: false })));
    pairs.forEach((p, i) => {
        if (i > 0) {
            const and = document.createElement('span'); and.className = 'target-and'; and.textContent = '∧'; chips.appendChild(and);
        }
        const lab = catalogLabelOf(p.col, p.val);
        const chip = document.createElement('span');
        chip.className = 'target-chip';
        chip.innerHTML = `<span class="chip-col">${lab.col}</span> = <span class="chip-val">${lab.val}</span>`;
        if (pairs.length > 1) {
            const b = document.createElement('button');
            b.type = 'button'; b.textContent = '✕';
            b.title = p.primary ? 'Remove this pair (the next one becomes the primary)' : 'Remove this pair from the target';
            b.onclick = () => (p.primary ? removePrimaryTarget() : removeTargetConjunct(p.col));
            chip.appendChild(b);
        }
        chips.appendChild(chip);
    });
    if (!info) { infoEl.textContent = '…'; return; }
    const pct = info.share === null ? '—' : `${(info.share * 100).toFixed(1)}%`;
    let text = `${info.n_positive === null ? '—' : info.n_positive.toLocaleString()} of ${info.n_samples.toLocaleString()} rows (${pct}). `
        + `${info.n_features} feature${info.n_features === 1 ? '' : 's'}`
        + (info.target_columns.length > 1 ? ` — ${info.target_columns.join(', ')} are the target and leave the feature space.` : '.');
    let warn = '';
    if (info.n_positive !== null && info.n_positive < 30) {
        warn = ` Only ${info.n_positive} objects match: any certified cell will be a handful of objects — raise Min. objects before trusting a centre.`;
    }
    infoEl.innerHTML = text + (warn ? `<span class="warn">${warn}</span>` : '');
}

// A 14 px donut showing a value's share of all rows. Neutral indigo: the
// greens and reds of the interface are certificates, and a share is not
// one. The exact percentage sits beside it; on a donut this small, 3 % and
// 8 % are the same picture.
function shareDonutSVG(share) {
    const r = 5.5, c = 2 * Math.PI * r;
    const filled = Math.max(0, Math.min(1, share)) * c;
    return `<svg class="share-donut" width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">`
        + `<circle cx="7" cy="7" r="${r}" fill="none" stroke="rgba(148,163,184,0.25)" stroke-width="3"/>`
        + `<circle cx="7" cy="7" r="${r}" fill="none" stroke="#818cf8" stroke-width="3" `
        + `stroke-dasharray="${filled.toFixed(3)} ${(c - filled).toFixed(3)}" transform="rotate(-90 7 7)"/></svg>`;
}

function formatSharePct(share) {
    const pct = share * 100;
    if (pct >= 10) return `${pct.toFixed(0)}%`;
    if (pct >= 1) return `${pct.toFixed(1)}%`;
    return pct > 0 ? `${pct.toFixed(2)}%` : '0%';
}

// The base-rate invariant, drawn: within each column's (share-ordered)
// list, a divider is placed after the last value whose share is at least
// the current tau. Above it a presence search is void (tau is not above
// the value's base rate) and only Absence is available; below it either
// direction works. Re-run whenever tau changes.
// With an anchored scale a value can always be analysed (the boundary is
// re-anchored on click), so the divider now says what WILL happen: values
// whose base rate lies on the wrong side of the current certified boundary
// will have that boundary moved when picked.
function updateCatalogTauDividers() {
    const cert = readCertBoundaryPct() / 100;
    const absence = activeDirection === 'absence';
    document.querySelectorAll('#catalogAccordion .crit-item').forEach(item => {
        const share = Number(item.dataset.share);
        // Under absence the axis is the complement, whose base rate is 1 - share.
        const flagged = Number.isFinite(share) && (absence ? (1 - share) >= cert : share >= cert);
        item.classList.toggle('above-tau', flagged);
        item.title = flagged
            ? (absence
                ? `Rows without this value are ${((1 - share) * 100).toFixed(1)}% of all rows, not below the red boundary (${(cert * 100).toFixed(0)}%): picking it raises the boundary above that.`
                : `This value fills ${(share * 100).toFixed(1)}% of the rows, not below the green boundary (${(cert * 100).toFixed(0)}%): picking it raises the boundary above its base rate.`)
            : '';
    });
    document.querySelectorAll('#catalogAccordion .char-content').forEach(content => {
        const old = content.querySelector('.tau-divider');
        if (old) old.remove();
        const items = Array.from(content.querySelectorAll('.crit-item'));
        const flags = items.map(i => i.classList.contains('above-tau'));
        const divider = document.createElement('div');
        divider.className = 'tau-divider';
        if (absence) {
            // Rare values are at the bottom of the share-ordered list, and
            // those are the ones whose complement base rate is high.
            const first = flags.indexOf(true);
            if (first < 0) return;
            divider.innerHTML = `<span>below: rows without the value ≥ ${(cert * 100).toFixed(0)}% — boundary moves when picked</span>`;
            items[first].before(divider);
        } else {
            const last = flags.lastIndexOf(true);
            if (last < 0) return;
            divider.innerHTML = `<span>above: base rate ≥ ${(cert * 100).toFixed(0)}% — boundary moves when picked</span>`;
            items[last].after(divider);
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

// The certificate boundary the SERVER receives, as a purity of the
// indicator searched. The scale is always drawn in shares of the chosen
// VALUE (see the anchored-scale block below): under presence the certified
// boundary is that share itself; under absence it is "value share <= c",
// i.e. a complement purity of 1 - c.
function readCertTau() {
    // Under absence the scale's axis is the share of rows WITHOUT the
    // value, so the boundary is already the complement purity.
    const tau = readCertBoundaryPct() / 100;
    // 1.0 is allowed: "cells that are entirely the target value" is a
    // well-posed request about the observed table (it is rejected only in
    // strict mode, where it would be a request to PROVE exact purity).
    return (Number.isFinite(tau) && tau > 0 && tau <= 1) ? tau : 0.90;
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
    // In the landscape the user is reading how the chosen branch's
    // dimensionality moves along the Coverage-vs-purity-floor curves, so a
    // tau change must NOT jump to the server's default branch. Same target,
    // same criterion, same schema: keep the active d if it still exists.
    await runAnalysis(lastTargetCol, lastCriterion, lastFeatures,
        { preserveBranch: viewMode === 'landscape' || viewMode === 'redundancy' });
}

function renderCertificateSummary(response) {
    const el = document.getElementById('certSummary');
    if (!el) return;
    el.innerHTML = '';
    if (response && response.schema && response.schema.selected_from_landscape) {
        let note = 'Schema opened from the landscape: '
            + (response.schema.feature_names || []).join(' + ')
            + '. Chosen by looking at the data, so no uncorrected permutation p-value is reported for it; '
            + 'the cross-validated coverage is still out-of-sample.';
        const dKey = String(response.default_branch);
        const b = response.branches && response.branches[dKey];
        const cert = b && b.certificate;
        if (b && cert && cert.partition_matches_search === false && b.search_centers && b.centers) {
            note += ` The search scored this schema on a capacity-coarsened partition (coverage ${(b.search_centers.coverage * 100).toFixed(1)}%, ${b.search_centers.n_centers} centres — the numbers in the landscape); the lattice shows the full-resolution partition (${(b.centers.coverage * 100).toFixed(1)}%, ${b.centers.n_centers} centres).`;
        }
        el.textContent = note;
    }
}

// Monotone id of the latest /api/analyze request. Arrow-key nudges and
// repeated drags fire overlapping requests; a response that is not the
// latest is dropped so an earlier tau can never overwrite a later one.
let _analysisRequestSeq = 0;

async function runAnalysis(targetCol, criterion = null, features = null, options = {}) {
    const preserveBranch = Boolean(options && options.preserveBranch);
    const requestId = ++_analysisRequestSeq;
    _plotlyWarmupCancelled = true; // the real render triangulates what it needs
    showWelcomeState(false);
    showLoader(true);
    showAnalysisError(null);
    lastTargetCol = targetCol;
    lastCriterion = criterion;
    lastFeatures = features; // an explicit schema opened from the landscape, or null
    if (criterion === null) targetAlso = [];   // a raw-column target has no conjuncts
    markCatalogTarget();
    renderTargetPanel(null);
    try {
        // Re-anchor the colour scale to the target's base rate BEFORE asking,
        // so the certified boundary is always admissible (the server refuses a
        // boundary on the wrong side of the base rate); a forced move is shown.
        // The share comes from /api/target (one boolean mask on the server),
        // which also fills the target panel; the catalog's share is the
        // fallback for a plain value.
        if (criterion !== null) {
            let share = catalogShareOf(targetCol, criterion);
            try {
                targetInfoCache = await fetchTargetInfo(targetCol, criterion);
                if (requestId !== _analysisRequestSeq) return; // superseded
                share = targetInfoCache.share;
            } catch (err) {
                // The catalog's share still anchors a plain value; a
                // conjunction without its share cannot be anchored honestly.
                targetInfoCache = null;
                if (targetAlso.length) throw err;
            }
            renderTargetPanel(targetInfoCache);
            if (share !== null) {
                const moved = setScaleAnchor(share);
                showBaseRateMove(moved);
            }
        }
        const reqBody = screenRequestOptions({
            target: targetCol,
            tau: readCertTau(),
            alpha: readCertAlpha(),
            rule: readCertRule(),
            min_samples: readCertMinSamples(),
            direction: readDirection(),
        });
        if (criterion !== null) {
            reqBody.criterion = criterion;
        }
        if (targetAlso.length) {
            reqBody.also = targetAlsoPairs();
        }
        if (features !== null) {
            reqBody.features = features;
        }

        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqBody)
        });

        if (requestId !== _analysisRequestSeq) return; // superseded
        if (response.ok) {
            const data = await response.json();
            if (requestId !== _analysisRequestSeq) return; // superseded while parsing
            loadBranchesResponse(data, preserveBranch);
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
        // Only the latest request owns the loader; a superseded one must not
        // hide it while the newer request is still in flight.
        if (requestId === _analysisRequestSeq) showLoader(false);
    }
}

// (Re-)renders the branch list for the ALREADY-FETCHED
// currentBranchesResponse (no new /api/analyze request) and picks which
// branch becomes active. Called right after a fresh analysis
// (preserveActiveIfPossible=false — always pick the server's own default)
// — there is no client-side branch filter, every discovered branch is
// always shown.
//
// `carryOverDim` (a fresh response only): the branch key that was active
// before this response arrived. If the new response still has a branch of
// that dimensionality it is re-selected against the NEW payload (the old
// currentPayload belongs to the previous response and must be replaced);
// otherwise the server default is used as usual.
function selectDefaultBranchAndRender(preserveActiveIfPossible, carryOverDim = null) {
    if (!currentBranchesResponse) return;
    const data = currentBranchesResponse;
    const allDims = (data.branch_dims || []).map(String);
    if (allDims.length === 0) return; // handled by loadBranchesResponse's own early return

    renderBranchSelector(data);
    showAnalysisError(null);

    if (carryOverDim !== null && allDims.includes(carryOverDim)) {
        selectBranch(carryOverDim, /* fromInitialLoad */ true);
        return;
    }

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
//
// preserveBranch: keep the previously active branch dimensionality when the
// new response still offers it (used for tau changes in the landscape).
function loadBranchesResponse(data, preserveBranch = false) {
    const carryOverDim = (preserveBranch && activeBranchDim !== null) ? String(activeBranchDim) : null;
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

    selectDefaultBranchAndRender(/* preserveActiveIfPossible */ false, carryOverDim);
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
        // The card carries only the headline coverage + centres number,
        // which is already what selects and ranks
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
    if (viewMode === 'redundancy') refreshCenterGroups(0);
    // The frontier is drawn at the current floor and marks the selected branch,
    // so a change of branch changes it; the tau-curves do not depend on tau.
    if (viewMode === 'tradeoffs') { renderTauCurves(); refreshFrontier(false); }
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
        labelTau.textContent = direction === 'absence' ? '🔴 Red from:' : '🟢 Green from:';
        labelTau.style.color = direction === 'absence' ? '#ef4444' : 'var(--green)';
        labelTau.title = direction === 'absence'
            ? 'A cell is certified FREE of the chosen value — and is drawn red — when its share of rows WITHOUT the value reaches this. Must lie above the base rate of rows without the value (the 1−p₀ tick). Moving this re-runs the analysis.'
            : "A cell is a discrete centre — and is drawn green — when its share of the target value reaches this. Must lie above the value's base rate (the p₀ tick). Coverage is computed from exactly these cells, so moving this re-runs the analysis. 100% is allowed and means 'only cells that are entirely the target value'.";
    }
    if (labelRed) {
        labelRed.textContent = direction === 'absence' ? '⚪ Grey up to:' : '🔴 Red up to:';
        labelRed.style.color = direction === 'absence' ? '#94a3b8' : '#ef4444';
        labelRed.title = direction === 'absence'
            ? 'Purely a colour boundary, at or below the base rate: cells whose share of rows without the value is below it are drawn grey (the value is present there at least as often as in the data as a whole); between it and the red boundary, brown. Re-colours immediately, changes no reported number.'
            : 'Purely a colour boundary, at or below the base rate: cells below it are drawn red (the value is rarer there than in the data as a whole); between it and the green boundary, brown. Re-colours immediately, changes no reported number.';
    }
    if (changed) {
        // Defaults for the new direction (the same purity floor of 90 % on
        // its own axis), then clamp to the anchor.
        const tauEl = document.getElementById('certTau');
        const decoEl = document.getElementById('colorRedTo');
        if (tauEl) tauEl.value = '90';
        const a = axisAnchorPct();
        if (decoEl) { decoEl.value = String(a === null ? 40 : a); decoEl.dataset.userSet = ''; }
        syncColorScaleFromInputs();
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

let _baseRateMoveNote = null;
function showBaseRateMove(note) {
    _baseRateMoveNote = note || null;
    updateBaseRateNote();
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
        ? `rows without it are ${((1 - pValue) * 100).toFixed(1)}%, and a certificate of absence must lie above that on the "without" axis`
        : `a certificate of presence must lie above it`;
    el.textContent = `Base rate of the chosen value: ${(pValue * 100).toFixed(1)}% of rows; ${need}.`
        + (pValue > 0.5 && activeDirection === 'presence'
            ? ' The value is more common than not — consider searching for its absence.' : '')
        + (_baseRateMoveNote ? ' ' + _baseRateMoveNote : '');
}

// Certificate parameters currently in force, mirrored from the payload so the
// colour function, the legend and the panel can never disagree about tau.
let activeTau = 0.90;
let activeAlpha = 0.05;
let activeRule = 'purity';
let activeMinSamples = 1;
let activePrevalence = 0.0;

// The decorative boundary in INDICATOR terms (what getColorIndexForCell
// compares cell purities against): under presence the value share itself;
// under absence the complement of the "grey from" value share.
function readRedTo() {
    return readDecoBoundaryPct() / 100;
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
// ---------------------------------------------------------------------------
// The anchored colour scale.
//
// The axis is the share, in a cell, of what is being searched for: the
// chosen VALUE under presence, the rows WITHOUT it under absence. The base
// rate of that quantity -- p0 under presence, 1 - p0 under absence -- is a
// fixed tick on it (the anchor), and the certified boundary starts one
// point beyond the anchor and can only move further right, never across
// it: a certificate at or below the base rate would call cells "centres"
// that say nothing (Project_Master_Document.md Section 4.8, base-rate
// invariant). The second boundary is decorative -- it partitions the
// UNCERTIFIED cells into "above the base rate but short of the certificate"
// (brown) and "below the base rate" (red under presence: the value is
// rarer there than overall; grey under absence: the value is present at
// least as often as overall) -- and defaults to the anchor so that brown
// never covers a cell that sits below the base rate.
//
//   presence (axis = share of the value):
//              red [0, deco)   brown [deco, cert)   green [cert, 100]
//   absence  (axis = share of rows without the value):
//              grey [0, deco)  brown [deco, cert)   red [cert, 100]
//   in both:   deco <= anchor < cert
//
// `cert` is what #certTau shows, on the axis of the current mode, and is
// exactly the purity the server certifies (readCertTau()).
let scaleAnchorPct = null;   // round(p0 * 100) of the chosen VALUE, or null

// The anchor on the axis of the current mode.
function axisAnchorPct() {
    if (scaleAnchorPct === null) return null;
    return activeDirection === 'absence' ? 100 - scaleAnchorPct : scaleAnchorPct;
}

function readCertBoundaryPct() {
    const el = document.getElementById('certTau');
    const v = el ? Number(el.value) : NaN;
    if (Number.isFinite(v)) return clampPct(Math.round(v), 0, 100);
    return activeDirection === 'absence' ? 10 : 90;
}

function readDecoBoundaryPct() {
    const el = document.getElementById('colorRedTo');
    const v = el ? Number(el.value) : NaN;
    if (Number.isFinite(v)) return clampPct(Math.round(v), 0, 100);
    const a = axisAnchorPct();
    return a !== null ? a : 40;
}

// Admissible range of the certified boundary: strictly above the anchor
// (one point past it at the closest), up to 100.
function certBoundaryRange() {
    const a = axisAnchorPct();
    return { lo: a === null ? 1 : Math.min(100, a + 1), hi: 100 };
}

// Admissible range of the decorative boundary: at or below the anchor and
// at least one point below the certified boundary.
function decoBoundaryRange(certPct) {
    const a = axisAnchorPct();
    const hi = Math.min(a === null ? 100 : a, certPct - COLOR_SCALE_MIN_GAP);
    return { lo: 0, hi: Math.max(hi, 0) };
}

function clampCertBoundary(pct) {
    const r = certBoundaryRange();
    return clampPct(Math.round(pct), r.lo, r.hi);
}

function clampDecoBoundary(pct, certPct) {
    const r = decoBoundaryRange(certPct);
    return clampPct(Math.round(pct), r.lo, r.hi);
}

// Paints the track, the two handles and the anchor tick from percentages
// on the value-share axis.
function setColorScaleUI(decoPct, certPct) {
    const track = document.getElementById('colorScaleTrack');
    const handleDeco = document.getElementById('handleRed');
    const handleCert = document.getElementById('handleGreen');
    const decoValueEl = document.getElementById('handleRedValue');
    const certValueEl = document.getElementById('handleGreenValue');
    if (!track || !handleDeco || !handleCert) return;
    track.style.setProperty('--red-pct', `${decoPct}%`);
    track.style.setProperty('--tau-pct', `${certPct}%`);
    handleDeco.style.left = `${decoPct}%`;
    handleCert.style.left = `${certPct}%`;
    handleDeco.setAttribute('aria-valuenow', String(Math.round(decoPct)));
    handleCert.setAttribute('aria-valuenow', String(Math.round(certPct)));
    if (decoValueEl) decoValueEl.textContent = `${Math.round(decoPct)}%`;
    if (certValueEl) certValueEl.textContent = `${Math.round(certPct)}%`;
    const tick = document.getElementById('anchorTick');
    const a = axisAnchorPct();
    if (tick) {
        if (a === null) {
            tick.style.display = 'none';
        } else {
            tick.style.display = '';
            tick.style.left = `${a}%`;
            tick.title = activeDirection === 'absence'
                ? `Base rate of rows WITHOUT the chosen value: ${a}% (the value itself fills ${scaleAnchorPct}%). The certified boundary cannot cross it.`
                : `Base rate of the chosen value: ${a}% of rows. The certified boundary cannot cross it.`;
            const label = tick.querySelector('.anchor-label');
            if (label) label.textContent = (activeDirection === 'absence' ? '1−p₀ ' : 'p₀ ') + `${a}%`;
        }
    }
    const caption = document.getElementById('scaleAxisCaption');
    if (caption) {
        const name = currentValueLabel();
        caption.textContent = activeDirection === 'absence'
            ? (name ? `axis: share of rows WITHOUT «${name}» in a cell` : 'axis: share of rows without the chosen value in a cell')
            : (name ? `axis: share of «${name}» in a cell` : 'axis: share of the chosen value in a cell');
    }
}

function currentValueLabel() {
    if (lastTargetCol === null || lastCriterion === null) return null;
    const col = (allColumnsData || []).find(c => c.id === lastTargetCol);
    const crit = col && (col.criteria || []).find(c => String(c.id) === String(lastCriterion));
    const primary = crit ? crit.label : String(lastCriterion);
    if (!targetAlso.length) return primary;
    // A conjunction is named in full: "yes ∧ sex = female" would be ambiguous.
    const pl = catalogLabelOf(lastTargetCol, lastCriterion);
    return [`${pl.col} = ${pl.val}`].concat(targetAlso.map(p => { const l = catalogLabelOf(p.col, p.val); return `${l.col} = ${l.val}`; })).join(' ∧ ');
}

// Reads the two number inputs, clamps them to the admissible ranges (writing
// the clamped values back), and paints the slider. Call this whenever the
// inputs change from ANY source so the slider can never show a stale or
// inadmissible position.
function syncColorScaleFromInputs() {
    updateCatalogTauDividers();
    const tauEl = document.getElementById('certTau');
    const decoEl = document.getElementById('colorRedTo');
    const cert = clampCertBoundary(readCertBoundaryPct());
    const deco = clampDecoBoundary(readDecoBoundaryPct(), cert);
    if (tauEl && String(cert) !== tauEl.value) tauEl.value = String(cert);
    if (decoEl && String(deco) !== decoEl.value) decoEl.value = String(deco);
    setColorScaleUI(deco, cert);
}

// Re-anchors the scale to a value's base rate (p0 as a share of all rows).
// If the certified boundary now sits on the wrong side of the anchor, it is
// moved to the nearest admissible position and the move is reported; the
// decorative boundary is reset to the anchor unless it is still admissible.
// Returns a note describing any forced move, or null.
function setScaleAnchor(p0) {
    scaleAnchorPct = (typeof p0 === 'number' && Number.isFinite(p0)) ? clampPct(Math.round(p0 * 100), 0, 100) : null;
    const tauEl = document.getElementById('certTau');
    const decoEl = document.getElementById('colorRedTo');
    const before = readCertBoundaryPct();
    const cert = clampCertBoundary(before);
    let note = null;
    if (cert !== before) {
        note = activeDirection === 'absence'
            ? `The red boundary was raised from ${before}% to ${cert}%: rows without the value are ${axisAnchorPct()}% of all rows, and a certificate of absence must lie above that.`
            : `The green boundary was raised from ${before}% to ${cert}%: the value fills ${scaleAnchorPct}% of the rows, and a certificate of presence must lie above that.`;
    }
    if (tauEl) tauEl.value = String(cert);
    if (decoEl) {
        const decoRange = decoBoundaryRange(cert);
        const current = Number(decoEl.value);
        const keep = Number.isFinite(current) && current >= decoRange.lo && current <= decoRange.hi && decoEl.dataset.userSet === '1';
        const a = axisAnchorPct();
        decoEl.value = String(keep ? Math.round(current) : clampDecoBoundary(a === null ? current : a, cert));
    }
    syncColorScaleFromInputs();
    return note;
}

// The base rate of a (column, value) from the catalog, if the server sent
// shares (catalog_from_dataframe), else null.
function catalogShareOf(colId, critId) {
    const col = (allColumnsData || []).find(c => c.id === colId);
    if (!col) return null;
    const crit = (col.criteria || []).find(c => String(c.id) === String(critId));
    return (crit && typeof crit.share === 'number') ? crit.share : null;
}

let colorScaleDragTarget = null; // 'red' (decorative) | 'green' (certified) | null

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

function moveDecoBoundary(pct) {
    const decoEl = document.getElementById('colorRedTo');
    const cert = clampCertBoundary(readCertBoundaryPct());
    const next = clampDecoBoundary(pct, cert);
    if (decoEl) { decoEl.value = String(next); decoEl.dataset.userSet = '1'; }
    setColorScaleUI(next, cert);
    applyColorBoundary(); // cosmetic: re-colour live
}

function moveCertBoundary(pct) {
    const tauEl = document.getElementById('certTau');
    const next = clampCertBoundary(pct);
    if (tauEl) tauEl.value = String(next);
    const deco = clampDecoBoundary(readDecoBoundaryPct(), next);
    const decoEl = document.getElementById('colorRedTo');
    if (decoEl) decoEl.value = String(deco);
    setColorScaleUI(deco, next);
}

function onColorScalePointerMove(evt) {
    if (!colorScaleDragTarget) return;
    evt.preventDefault();
    const pct = colorScalePctFromEvent(evt);
    if (colorScaleDragTarget === 'red') moveDecoBoundary(pct);
    else moveCertBoundary(pct); // deliberately NOT recomputed while dragging
}

function onColorScalePointerUp() {
    if (!colorScaleDragTarget) return;
    const wasCert = colorScaleDragTarget === 'green';
    colorScaleDragTarget = null;
    document.body.style.userSelect = '';
    if (wasCert) applyCertificate(); // commit: re-run the analysis at the new boundary
}

// Arrow-key nudge for the focused handle (1 point; Shift = 5 points).
function onColorScaleKeyDown(which) {
    return function (evt) {
        const step = evt.shiftKey ? 5 : 1;
        let delta = 0;
        if (evt.key === 'ArrowLeft' || evt.key === 'ArrowDown') delta = -step;
        else if (evt.key === 'ArrowRight' || evt.key === 'ArrowUp') delta = step;
        else return;
        evt.preventDefault();
        if (which === 'red') {
            moveDecoBoundary(readDecoBoundaryPct() + delta);
        } else {
            moveCertBoundary(readCertBoundaryPct() + delta);
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
    const decoEl = document.getElementById('colorRedTo');
    if (decoEl) decoEl.addEventListener('input', () => { decoEl.dataset.userSet = '1'; });

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
    // Anchor the scale on the value's base rate as the server measured it
    // (the indicator's prevalence, complemented under absence).
    if (payload.centers && payload.centers.prevalence !== undefined && (currentBranchesResponse && currentBranchesResponse.criterion !== null)) {
        setScaleAnchor(activeDirection === 'absence' ? 1 - activePrevalence : activePrevalence);
    }
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

    // HUD strip intentionally shows only Coverage and Centres (plus the
    // certified-free mass under absence); purity and the raw p-value were
    // judged clutter for this always-visible strip, and the per-branch
    // cards follow the same rule. Which `d` Coverage/Centres refer to is no longer
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
    if (m) return [parseInt(m[1]), parseInt(m[2]), parseInt(m[3])];
    return [0, 0, 0];
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
    highlight: null,    // lattice cells of the expanded duplicate-centre group
};

function landscapeParams() {
    if (!currentBranchesResponse) return null;
    const p = screenRequestOptions({
        target: currentBranchesResponse.target,
        criterion: currentBranchesResponse.criterion,
        tau: readCertTau(), alpha: readCertAlpha(), rule: readCertRule(),
        min_samples: readCertMinSamples(), direction: readDirection(),
    });
    if (currentBranchesResponse.also && currentBranchesResponse.also.length) p.also = currentBranchesResponse.also;
    // The feature space the BRANCHES were computed in, not the screen's
    // live exclusions: a column excluded after the analysis (and not yet
    // re-run) must not shift the feature indices the branches refer to.
    if (Array.isArray(currentBranchesResponse.dropped_columns)) p.drop = currentBranchesResponse.dropped_columns.slice();
    if (typeof currentBranchesResponse.prune_dependent === 'boolean') p.prune = currentBranchesResponse.prune_dependent;
    return p;
}

function onAnalysisLoadedForLandscape(data) {
    // A new analysis of a different target/certificate invalidates the
    // cached landscape; opening a schema of the same target keeps it, so
    // the user can return to the same cell.
    const p = landscapeParams();
    const key = p ? JSON.stringify(p) : null;
    if (key !== landscapeState.key) {
        landscapeState = { key, bins: null, dSelection: null, cell: null, cellData: null, highlight: null };
        resetCenterGroupsState();
    }
    const ck = curvesKey();
    if (ck !== curvesState.key) {
        curvesState = { key: ck, data: null, point: null, pointData: null };
    }
    const rk = rulesKey();
    if (rk !== rulesState.key) {
        rulesState.key = rk; rulesState.data = null; rulesState.bar = null; rulesState.page = 0;
        rulesState.erased = []; rulesState.history = []; rulesState.notice = null;
        rulesCurves = { key: null, data: null };
    }
    if (data && data.schema && data.schema.selected_from_landscape) {
        setViewMode('lattice');
    } else if (viewMode === 'rules') {
        refreshRules();   // fetches the curves first, then the rules
    } else if (viewMode === 'landscape') {
        refreshLandscape();
    } else if (viewMode === 'tradeoffs') {
        refreshTauCurves();
        refreshFrontier(false);
    } else if (viewMode === 'redundancy') {
        refreshCenterGroups(0);
    } else if (viewMode === 'screen') {
        refreshScreen(false);
    }
}

function setViewMode(mode) {
    viewMode = (mode === 'landscape' || mode === 'redundancy' || mode === 'screen' || mode === 'tradeoffs' || mode === 'rules')
        ? mode : 'lattice';
    document.querySelectorAll('.view-mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === viewMode);
    });
    const lattice = viewMode === 'lattice';
    const plot = document.getElementById('plot-container');
    const area = document.getElementById('landscapeArea');
    const dup = document.getElementById('redundancyArea');
    const scr = document.getElementById('screenArea');
    const tro = document.getElementById('tradeoffsArea');
    const rul = document.getElementById('rulesArea');
    const slices = document.getElementById('slice-controller');
    if (plot) plot.style.display = lattice ? '' : 'none';
    if (rul) rul.style.display = viewMode === 'rules' ? 'flex' : 'none';
    if (area) area.style.display = viewMode === 'landscape' ? 'flex' : 'none';
    if (dup) dup.style.display = viewMode === 'redundancy' ? 'flex' : 'none';
    if (scr) scr.style.display = viewMode === 'screen' ? 'flex' : 'none';
    if (tro) tro.style.display = viewMode === 'tradeoffs' ? 'flex' : 'none';
    if (slices && !lattice) slices.style.display = 'none';
    // Display Settings: the scan rows and the contour panel belong to the
    // lattice; every other tab keeps only the certificate controls.
    const scanRows = document.getElementById('scanOnlyRows');
    const contour = document.getElementById('contourPanel');
    const scanBlock = document.getElementById('scanBlock');
    const rulesSide = document.getElementById('rulesSidePanel');
    if (scanRows) scanRows.style.display = lattice ? 'contents' : 'none';
    if (contour) contour.style.display = lattice ? '' : 'none';
    // Rules: the certificate controls step aside for the erasing curves.
    if (scanBlock) scanBlock.style.display = viewMode === 'rules' ? 'none' : '';
    if (rulesSide) rulesSide.style.display = viewMode === 'rules' ? '' : 'none';
    if (viewMode === 'landscape') {
        refreshLandscape();
    } else if (viewMode === 'tradeoffs') {
        refreshTauCurves();
        refreshFrontier(false);
    } else if (viewMode === 'redundancy') {
        refreshCenterGroups(0);
    } else if (viewMode === 'rules') {
        refreshRules();   // fetches the curves first, then the rules
    } else if (viewMode === 'screen') {
        refreshScreen(false);
        renderScreen();
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
        rerenderLandscapeFromCache();
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
        return;
    } catch (err) {
        showAnalysisError('Landscape request failed: ' + err.message);
        return;
    }
}

// Cached bins: redraw, and redraw the open cell's list so the "current"
// highlight follows whichever schema is on the lattice now.
function rerenderLandscapeFromCache() {
    renderLandscape();
    if (landscapeState.cellData) renderLandscapeCell();
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
    // Now that the grid lines are cell boundaries, a circle must fit INSIDE
    // its cell: diameter = 80 % of the smaller cell side in pixels (margins
    // l+r = 110, t+b = 90), capped at the previous fixed 34 px.
    const plotW = Math.max(1, (container.clientWidth || 800) - 110);
    const plotH = Math.max(1, (container.clientHeight || 420) - 90);
    const cellPx = Math.min(plotW, plotH) / n;
    const markerPx = Math.max(16, Math.min(34, Math.floor(0.8 * cellPx)));
    const traces = [{
        type: 'scatter', mode: 'markers+text', x: xs, y: ys, text: texts, textposition: 'middle center',
        textfont: { size: 11, color: isBright.map(b => b ? '#0f172a' : '#f8fafc') },
        marker: { size: markerPx, color: colors, line: { width: 1, color: 'rgba(255,255,255,0.25)' } },
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
        const opened = !!(currentBranchesResponse && currentBranchesResponse.schema);
        traces.push({
            type: 'scatter', mode: 'markers', x: [wx], y: [wy],
            marker: { symbol: opened ? 'diamond' : 'star', size: Math.min(16, markerPx), color: '#facc15', line: { width: 1, color: '#0f172a' } },
            hovertext: [`${opened ? 'opened schema' : 'search winner'} (${currentPayload.metrics.d}D): ${(currentPayload.selected_features || []).join(' + ')}<br>${bins.x} ${(xFrac * 100).toFixed(1)}% · ${cc.n_centers} centres (search partition)`],
            hoverinfo: 'text', name: opened ? 'opened' : 'winner',
        });
    }
    if (landscapeState.cell) {
        traces.push({
            type: 'scatter', mode: 'markers', x: [landscapeState.cell.ix], y: [landscapeState.cell.iy],
            marker: { symbol: 'circle-open', size: markerPx + 8, color: '#f8fafc', line: { width: 2 } },
            hoverinfo: 'skip', name: 'selected',
        });
    }
    const hl = landscapeHighlightPoints(bins);
    if (hl && hl.xs.length) {
        traces.push({
            type: 'scatter', mode: 'markers', x: hl.xs, y: hl.ys,
            marker: { symbol: 'circle-open', size: markerPx + 8, color: '#f59e0b', line: { width: 3 } },
            hovertext: hl.hov, hoverinfo: 'text', name: 'group',
            customdata: hl.xs.map((x, i) => [x, hl.ys[i]]),
        });
    }
    // Cell boundaries, as in the main lattice (buildPlotData draws its grid at
    // k - 0.5): a category is the open interval BETWEEN two lines, so its
    // label and its circle sit at the integer centre and the lines at the
    // half-integers. Plotly's own grid is drawn at the tick values, i.e.
    // through the centres, which read as if the lines were the categories.
    const gridShapes = [];
    for (let k = 0; k <= n; k++) {
        const b = k - 0.5;
        const line = { color: 'rgba(255,255,255,0.10)', width: 1 };
        gridShapes.push({ type: 'line', layer: 'below', xref: 'x', yref: 'y', x0: b, x1: b, y0: -0.5, y1: n - 0.5, line });
        gridShapes.push({ type: 'line', layer: 'below', xref: 'x', yref: 'y', x0: -0.5, x1: n - 0.5, y0: b, y1: b, line });
    }
    const layout = {
        paper_bgcolor: '#070a13', plot_bgcolor: '#090d1a', showlegend: false,
        shapes: gridShapes,
        margin: { l: 90, r: 20, t: 20, b: 70 }, font: { family: 'Inter', color: '#94a3b8' },
        xaxis: {
            title: { text: bins.x === 'mass' ? 'Mass of certified value-free cells' : 'Coverage of the value', font: { color: '#c084fc', size: 13 } },
            tickvals: xLabels.map((_, i) => i), ticktext: xLabels, range: [-0.5, n - 0.5],
            tickfont: { size: 10 }, showgrid: false, ticks: '', zeroline: false, fixedrange: true,
        },
        yaxis: {
            title: { text: `Certified centres (of K max = ${bins.k_max})`, font: { color: '#c084fc', size: 13 } },
            tickvals: yLabels.map((_, i) => i), ticktext: yLabels, range: [-0.5, n - 0.5],
            tickfont: { size: 10 }, showgrid: false, ticks: '', zeroline: false, fixedrange: true,
        },
    };
    Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false });
    // Plotly.react() keeps the size autosize measured on the FIRST draw; it
    // does not re-measure the container. Opening/closing the cell panel
    // below changes this div's flex height, so without a resize the SVG kept
    // its old height, overflowed the div and painted over the top of the
    // panel (the title and the first schema row were hidden).
    fitLandscapePlot();
    container.removeAllListeners && container.removeAllListeners('plotly_click');
    container.on('plotly_click', (ev) => {
        const pt = ev.points && ev.points[0];
        if (!pt || !pt.customdata) return;
        openLandscapeCell(pt.customdata[0], pt.customdata[1], 0);
    });
}

// Re-measures #landscape-plot after layout has settled (next frame), so the
// Plotly SVG always matches the div's current flex height.
function fitLandscapePlot() {
    const container = document.getElementById('landscape-plot');
    if (!container || typeof Plotly === 'undefined' || !Plotly.Plots) return;
    requestAnimationFrame(() => {
        if (container.offsetParent === null || !container._fullLayout) return; // hidden or not drawn
        Plotly.Plots.resize(container);
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

// ---------------------------------------------------------------------------
// Rules: the cells of the winning schemas as conjunctive rules (Section 4.15)
// ---------------------------------------------------------------------------
// Every occupied cell of the winning 1D-4D schemas of the current analysis
// is a rule "col_1 = v_1 ∧ ... ∧ col_d = v_d" with its rows, its share of
// the value (purity) and whether the certificate makes it a centre. The
// server (`/api/rules`) lists them once per analysis; everything below is
// client-side: the dimensionality checkboxes, the cascading filter over
// conditions (characteristic → value → next characteristic), the purity
// bars split by dimensionality, and the cards. A rule "contains" a fixed
// condition when one of its own conditions is that (column, value); a
// partly filled filter therefore lists the bare rule and every refinement
// of it. No rule is hidden for being a refinement whose purity fell: the
// card shows the parents' purity so the reader sees the change.
let rulesState = {
    key: null,          // JSON of the analysis parameters + schemas the rules belong to
    data: null,         // /api/rules response
    fixed: [],          // conditions being composed in the filter [{feature, value, column_label, value_label}]
    applied: [],        // conditions in force for the bars and cards ("Show")
    facet: null,        // feature whose values are open in the filter, or null
    bar: null,          // selected purity interval (0..9) or null
    page: 0,
    erased: [],         // schemas erased from the view: [{features, d, names, n_rules, intervals}]
    history: [],        // erase operations, for Undo (each entry = one erased schema)
    notice: null,       // text of the last erase, shown under the bars
};
const RULES_PAGE = 30;
const RULES_N_BINS = 10;

// The schemas whose cells are the rules: every schema on the tau-curves
// (the best of its dimensionality at some floor - the "systems" the reader
// sees and erases there), plus the four winners of the current analysis
// (which are the curves' schemas at the current boundary). Sorted by key
// so the request, and its cache key, do not depend on discovery order.
function rulesSchemas() {
    const seen = new Map();
    const add = f => { if (Array.isArray(f) && f.length) seen.set(rulesSchemaKey(f), f.slice().sort((a, b) => a - b)); };
    if (currentBranchesResponse && currentBranchesResponse.branches) {
        (currentBranchesResponse.branch_dims || []).forEach(d => {
            const b = currentBranchesResponse.branches[String(d)];
            if (b) add(b.selected_feature_indices);
        });
    }
    const cd = rulesCurves.data;
    if (cd && cd.curves) Object.values(cd.curves).forEach(c => (c.features || []).forEach(add));
    return Array.from(seen.keys()).sort().map(k => seen.get(k));
}

function rulesMinRows() {
    const el = document.getElementById('ruMinRows');
    const v = el ? Math.round(Number(el.value)) : 1;
    return (Number.isFinite(v) && v >= 1) ? v : 1;
}

function rulesKey() {
    const p = landscapeParams();
    if (!p) return null;
    return JSON.stringify(Object.assign({}, p, { schemas: rulesSchemas(), min_rows: rulesMinRows() }));
}

function rulesDims() {
    const out = {};
    document.querySelectorAll('#ruDims input[type="checkbox"]').forEach(cb => { out[Number(cb.value)] = cb.checked; });
    return out;
}

function onRulesControl() {
    rulesState.bar = null;
    rulesState.page = 0;
    // A dimensionality change can strand a fixed condition (its column no
    // longer occurs); keep it - the counts say 0 and the user sees why.
    renderRules();
}

function onRulesMinRowsChange() {
    rulesState.bar = null;
    rulesState.page = 0;
    refreshRules();
}

async function refreshRules() {
    const p = landscapeParams();
    if (!p || !currentBranchesResponse || currentBranchesResponse.criterion === null) {
        const note = document.getElementById('ruNote');
        if (note) note.textContent = 'The rules view needs a specific target value: pick one in the catalog.';
        return;
    }
    // The curves name the schemas; without them only the winners would be
    // listed, so they are fetched first (once per analysis).
    if (!rulesCurves.data || rulesCurves.key !== rulesCurvesKey()) {
        await refreshRulesCurves(true);
        if (!rulesCurves.data) return;
    }
    const schemas = rulesSchemas();
    if (!schemas.length) return;
    const key = rulesKey();
    if (rulesState.data && rulesState.key === key) {
        renderRules();
        return;
    }
    const note = document.getElementById('ruNote');
    if (note) note.textContent = 'Listing the rules…';
    try {
        const res = await fetch('/api/rules', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.assign({}, p, { schemas, min_rows: rulesMinRows() })),
        });
        const data = await res.json();
        if (!res.ok) { showAnalysisError('Rules: ' + (data.error || res.status)); return; }
        rulesState.key = key;
        rulesState.data = data;
        rulesState.bar = null;
        rulesState.page = 0;
        renderRules();
    } catch (err) {
        showAnalysisError('Rules request failed: ' + err.message);
    }
}

// The rules of the checked dimensionalities.
function rulesPool() {
    const data = rulesState.data;
    if (!data) return [];
    const dims = rulesDims();
    return data.rules.filter(r => dims[r.d] && !rulesIsErased(r.features));
}

function rulesSchemaKey(features) {
    return features.slice().sort((a, b) => a - b).join(',');
}

function rulesIsErased(features) {
    const k = rulesSchemaKey(features);
    return rulesState.erased.some(e => e.key === k);
}

// Erasing a schema from the Rules view: clicking a segment of a purity bar
// removes the SCHEMA the segment's rules belong to - not just the rules in
// that interval - from every interval, the cards and the facet counts. A
// schema is one description; its cells at 30 % and at 100 % are the same
// description read at different places, so it leaves whole. The notice
// under the bars says where else it was. Undo takes back the last erase;
// Reset restores every schema. Nothing statistical changes: this is what
// is shown, not what was certified.
function eraseRulesSchema(features, d) {
    if (rulesIsErased(features)) return;
    const data = rulesState.data;
    const cdata = rulesCurves.data;
    const featNames = (data && data.feature_names) || (cdata && cdata.curves && cdata.curves[String(d)] && cdata.feature_names) || null;
    const key = rulesSchemaKey(features);
    const names = featNames ? features.map(j => featNames[j]) : features.map(String);
    const rules = data ? data.rules.filter(r => rulesSchemaKey(r.features) === key) : [];
    const intervals = Array.from(new Set(rules.map(r => rulesBinIndex(r.purity)))).sort((a, b) => a - b);
    // Where the schema stood on the envelope of its dimensionality.
    const stretches = rulesEnvelopeStretches(features, d);
    const entry = { key, features: features.slice(), d, names, n_rules: rules.length, intervals, stretches };
    rulesState.erased.push(entry);
    rulesState.history.push(entry);
    const parts = [`Erased the ${d}D schema ${names.join(' + ')}.`];
    if (stretches.length) {
        parts.push(`Its points at ${stretches.map(s => `${s.lo}–${s.hi}%`).join(', ')} are gone from the curves.`);
    }
    if (rules.length) {
        const where = intervals.map(ix => `${ix === 0 ? '[' : '('}${ix * 10},${(ix + 1) * 10}]%`).join(', ');
        parts.push(`Its ${rules.length} rule${rules.length === 1 ? '' : 's'} left ${intervals.length === 1 ? 'the interval' : 'the intervals'} ${where}.`);
    }
    rulesState.notice = parts.join(' ');
    if (rulesState.bar !== null) rulesState.page = 0;
    renderRules();
    renderRulesCurves();
}

function rulesEnvelopeStretches(features, d) {
    const cdata = rulesCurves.data;
    const c = cdata && cdata.curves && cdata.curves[String(d)];
    if (!c) return [];
    const key = rulesSchemaKey(features);
    const out = [];
    for (let i = 0; i < cdata.taus.length; i++) {
        const here = c.features[i] && c.features[i].length && rulesSchemaKey(c.features[i]) === key;
        const pct = Math.round(cdata.taus[i] * 100);
        if (here) {
            const last = out[out.length - 1];
            if (last && last.iEnd === i - 1) { last.hi = pct; last.iEnd = i; }
            else out.push({ lo: pct, hi: pct, iEnd: i });
        }
    }
    return out;
}

function undoRulesErase() {
    const last = rulesState.history.pop();
    if (!last) return;
    rulesState.erased = rulesState.erased.filter(e => e !== last);
    rulesState.notice = `Restored the ${last.d}D schema ${last.names.join(' + ')}${last.n_rules ? ` (${last.n_rules} rule${last.n_rules === 1 ? '' : 's'})` : ''}.`;
    renderRules();
    renderRulesCurves();
}

function resetRulesErase() {
    if (!rulesState.erased.length) return;
    const n = rulesState.erased.length;
    rulesState.erased = [];
    rulesState.history = [];
    rulesState.notice = `Restored ${n} erased schema${n === 1 ? '' : 's'}.`;
    renderRules();
    renderRulesCurves();
}

// ---- The erasing curves in the sidebar ---------------------------------
// The tau-curves of Trade-offs (the best coverage per dimensionality at every
// floor) as a map of the schemas - each stretch of a line is one schema. A
// click on a point erases the schema that the chosen branch (radio above the
// plots) has at that floor: its points become GAPS in the curves - nothing
// takes their place, the curves are not recomputed - and its rules leave
// the Rules tab. The branch is chosen first because the four lines overlap
// and a click on the plot alone could not say which schema was meant.
let rulesCurves = { key: null, data: null };

function rulesBranchPick() {
    const el = document.querySelector('input[name="ruBranch"]:checked');
    const d = el ? Number(el.value) : 2;
    return Number.isFinite(d) ? d : 2;
}

// Whether the branches that are NOT the click target are drawn at full
// strength. Off by default: the dimming is what makes the clickable line
// unmistakable, and a misdirected click erases a schema. On, the four lines
// can be read against each other; the click target is unchanged either way,
// and the picked line stays thicker with larger markers so it is still
// identifiable without the opacity cue.
function rulesShowAll() {
    const el = document.getElementById('ruShowAll');
    return !!(el && el.checked);
}

// Whether the points erased so far are drawn back in, as ghosts. The masked
// curves leave gaps where a schema was erased; this fills exactly those gaps
// from the UNMASKED data, dashed and hollow, so the reader sees what was
// removed and where without it re-entering the solid line. It changes nothing
// about the state: those schemas stay erased, their rules stay out of the
// list, and a click on a ghost still answers "already erased".
function rulesShowGone() {
    const el = document.getElementById('ruShowGone');
    return !!(el && el.checked);
}

function rulesCurvesKey() {
    const p = landscapeParams();
    if (!p) return null;
    const q = Object.assign({}, p);
    delete q.tau;
    return JSON.stringify(q);
}

async function refreshRulesCurves(quiet) {
    if (viewMode !== 'rules') return;
    const p = landscapeParams();
    if (!p || !currentBranchesResponse || currentBranchesResponse.criterion === null) return;
    const key = rulesCurvesKey();
    if (rulesCurves.data && rulesCurves.key === key) { renderRulesCurves(); return; }
    const note = document.getElementById('ruCurveNote');
    if (note) note.textContent = 'Computing the curves…';
    try {
        const body = Object.assign({}, p);
        const res = await fetch('/api/landscape/curves', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) { showAnalysisError('Curves: ' + (data.error || res.status)); return; }
        data.feature_names = (rulesState.data && rulesState.data.feature_names)
            || (currentBranchesResponse.branches && Object.values(currentBranchesResponse.branches)[0] && Object.values(currentBranchesResponse.branches)[0].feature_names) || null;
        rulesCurves = { key, data };
        renderRulesCurves();
        if (!quiet) refreshRules();
    } catch (err) {
        showAnalysisError('Curves request failed: ' + err.message);
    }
}

// The curves with the erased schemas' points taken out: null on the plots
// (a gap), zero and nameless for the leader strip.
function rulesMaskedCurves(data) {
    if (!rulesState.erased.length) return data;
    const erasedKeys = new Set(rulesState.erased.map(e => e.key));
    const curves = {};
    Object.keys(data.curves).forEach(d => {
        const c = data.curves[d];
        const gone = c.features.map(f => f && f.length && erasedKeys.has(rulesSchemaKey(f)));
        curves[d] = Object.assign({}, c, {
            x: c.x.map((v, i) => gone[i] ? null : v),
            n_centers: c.n_centers.map((v, i) => gone[i] ? null : v),
            features: c.features.map((f, i) => gone[i] ? [] : f),
            feature_names: c.feature_names.map((f, i) => gone[i] ? [] : f),
            gone,
        });
    });
    return Object.assign({}, data, { curves });
}

function renderRulesCurves() {
    const data = rulesCurves.data ? rulesMaskedCurves(rulesCurves.data) : null;
    const covEl = document.getElementById('ru-coverage-plot');
    const cenEl = document.getElementById('ru-centres-plot');
    const strip = document.getElementById('ruLeadStrip');
    const note = document.getElementById('ruCurveNote');
    if (!data || !covEl || !cenEl || typeof Plotly === 'undefined') return;
    const taus = data.taus;
    const dims = Object.keys(data.curves).map(Number).sort((a, b) => a - b);
    const raw = rulesCurves.data;
    const pick = rulesBranchPick();
    const showAll = rulesShowAll();
    const showGone = rulesShowGone() && rulesState.erased.length > 0;
    const cursor = readCertTau() * 100;
    const xmin = taus.length ? Math.floor(taus[0] * 100) - 1 : 0;
    const cursorShape = { type: 'line', xref: 'x', yref: 'paper', x0: cursor, x1: cursor, y0: 0, y1: 1, line: { color: '#f8fafc', width: 1, dash: 'dot' } };
    const covTraces = [], cenTraces = [];
    // The erased points, if asked for: exactly the gaps of the masked curves,
    // filled from the unmasked data. Drawn first so the surviving lines stay
    // on top of them, dashed and hollow so they never read as live points.
    if (showGone) {
        dims.forEach(d => {
            const c = data.curves[String(d)], rc = raw.curves[String(d)];
            if (!rc || !c.gone || !c.gone.some(Boolean)) return;   // nothing erased in this d
            const ghost = {
                x: taus.map(t => t * 100), name: `${d}D erased`, showlegend: false,
                hoverinfo: 'skip', opacity: 0.55, connectgaps: false,
                line: { color: TAU_CURVE_COLORS[d] || '#e2e8f0', width: 1, dash: 'dot' },
                marker: { size: 5, color: 'rgba(0,0,0,0)', line: { color: TAU_CURVE_COLORS[d] || '#e2e8f0', width: 1 } },
            };
            covTraces.push(Object.assign({}, ghost, {
                type: 'scatter', mode: 'lines+markers',
                y: taus.map((t, i) => (c.gone[i] ? rc.x[i] * 100 : null)),
            }));
            cenTraces.push(Object.assign({}, ghost, {
                type: 'scatter', mode: 'lines+markers',
                line: Object.assign({}, ghost.line, { shape: 'hv' }),
                y: taus.map((t, i) => (c.gone[i] ? rc.n_centers[i] : null)),
            }));
        });
    }
    // The picked branch is drawn last (on top), full opacity, larger
    // markers; the others are context and take no clicks.
    dims.filter(d => d !== pick).concat(dims.includes(pick) ? [pick] : []).forEach(d => {
        const c = data.curves[String(d)];
        const colour = TAU_CURVE_COLORS[d] || '#e2e8f0';
        const active = d === pick;
        const custom = taus.map((t, i) => [c.n_centers[i], c.feature_names[i].join(' + ')]);
        // No hover boxes: they covered the very points to click. The
        // picked branch reports hover events (hoverinfo 'none' keeps the
        // events, drops the label) and a readout line under the plots
        // says what is under the cursor; the others are silent context.
        const common = {
            x: taus.map(t => t * 100), customdata: custom, name: `${d}D`,
            opacity: (active || showAll) ? 1 : 0.3,
            hoverinfo: active ? 'none' : 'skip',
        };
        covTraces.push(Object.assign({}, common, {
            type: 'scatter', mode: 'lines+markers', y: c.x.map(v => v === null ? null : v * 100), connectgaps: false,
            line: { color: colour, width: active ? 2 : 1 }, marker: { size: active ? 6 : 3, color: colour },
        }));
        cenTraces.push(Object.assign({}, common, {
            type: 'scatter', mode: 'lines+markers', y: c.n_centers, showlegend: false, connectgaps: false,
            line: { color: colour, width: active ? 2 : 1, shape: 'hv' }, marker: { size: active ? 6 : 3, color: colour },
        }));
    });
    const layoutCov = toLayout(data, { ytitle: data.x === 'mass' ? 'mass free, %' : 'coverage, %', xrange: [xmin, 101], yrange: [0, 102], shapes: [cursorShape], legend: true });
    layoutCov.margin = { l: 40, r: 6, t: 22, b: 24 }; layoutCov.font.size = 9;
    const layoutCen = toLayout(data, { ytitle: 'certified centres', xrange: [xmin, 101], shapes: [cursorShape] });
    layoutCen.margin = { l: 40, r: 6, t: 6, b: 24 }; layoutCen.font.size = 9;
    // An eraser, not a chart to explore: no zoom or pan on drag (a drag
    // used to zoom in with no visible way back), and a click anywhere in
    // the plot picks the nearest floor on the x axis - the points are
    // 1 % apart and too small to be hit one by one.
    // Plotly's own hover is off: its per-move work made the panel lag.
    // The pointer is read by our handlers below (floor from the axis
    // scale), the readout, the cursor line and the rings are drawn on an
    // overlay of our own.
    [layoutCov, layoutCen].forEach(l => {
        l.dragmode = false;
        l.hovermode = false;
        l.xaxis.fixedrange = true; l.yaxis.fixedrange = true;
    });
    const cfg = { displayModeBar: false, responsive: true, scrollZoom: false, doubleClick: false };
    Plotly.react(covEl, covTraces, layoutCov, cfg);
    Plotly.react(cenEl, cenTraces, layoutCen, cfg);
    covEl.style.cursor = 'crosshair'; cenEl.style.cursor = 'crosshair';
    const floorIndexOf = pt => {
        // Under hovermode 'x' every trace reports a point at the same
        // floor; take the floor from the x value, not a trace's index.
        let ti = pt.pointNumber;
        if (typeof pt.x === 'number') {
            let best = 0;
            for (let i = 1; i < taus.length; i++) if (Math.abs(taus[i] * 100 - pt.x) < Math.abs(taus[best] * 100 - pt.x)) best = i;
            ti = best;
        }
        return ti;
    };
    const readout = document.getElementById('ruCurveHover');
    const describe = ti => {
        const c = data.curves[String(pick)];
        if (!c) return '';
        const pct = Math.round(taus[ti] * 100);
        if (c.gone && c.gone[ti]) return `${pick}D at ${pct}%: erased.`;
        if (!c.features[ti] || !c.features[ti].length) return `${pick}D at ${pct}%: no schema certifies a centre.`;
        const key = rulesSchemaKey(c.features[ti]);
        let nPts = 0;
        for (let i = 0; i < taus.length; i++) if (c.features[i] && c.features[i].length && rulesSchemaKey(c.features[i]) === key) nPts += 1;
        return `${pick}D at ${pct}%: ${data.x} ${(c.x[ti] * 100).toFixed(1)}%, ${c.n_centers[ti]} centres — ${c.feature_names[ti].join(' + ')}. Click erases its ${nPts} blinking point${nPts === 1 ? '' : 's'}.`;
    };
    // Blinking highlight of the hovered schema's points on both plots: SVG
    // rings drawn on an overlay of our own, placed through Plotly's axis
    // scales - Plotly itself is not touched, so the highlight costs nothing
    // to move and the blink is a CSS animation.
    let hiKey = null;
    const overlayOf = el => {
        let ov = el.querySelector(':scope > svg.ru-overlay');
        if (!ov) {
            ov = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            ov.setAttribute('class', 'ru-overlay');
            el.appendChild(ov);
        }
        return ov;
    };
    const setCursorLine = (el, ov, ti) => {
        const fl = el._fullLayout, xa = fl && fl.xaxis, ya = fl && fl.yaxis;
        let line = ov.querySelector('line.ru-cursor');
        if (ti === null || !xa || !ya) { if (line) line.remove(); return; }
        if (!line) {
            line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('class', 'ru-cursor');
            ov.appendChild(line);
        }
        const px = (xa._offset + xa.d2p(taus[ti] * 100)).toFixed(1);
        line.setAttribute('x1', px); line.setAttribute('x2', px);
        line.setAttribute('y1', String(ya._offset)); line.setAttribute('y2', String(ya._offset + ya._length));
    };
    const drawRings = (el, ov, xs, ys) => {
        const fl = el._fullLayout, xa = fl && fl.xaxis, ya = fl && fl.yaxis;
        Array.from(ov.querySelectorAll('circle')).forEach(c => c.remove());
        if (!xa || !ya) return;
        xs.forEach((x, i) => {
            if (ys[i] === null || ys[i] === undefined) return;
            const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            c.setAttribute('cx', (xa._offset + xa.d2p(x)).toFixed(1));
            c.setAttribute('cy', (ya._offset + ya.d2p(ys[i])).toFixed(1));
            c.setAttribute('r', '6');
            ov.appendChild(c);
        });
    };
    const ovCov = overlayOf(covEl), ovCen = overlayOf(cenEl);
    const clearHighlight = () => {
        hiKey = null;
        [ovCov, ovCen].forEach(ov => Array.from(ov.querySelectorAll('circle')).forEach(c => c.remove()));
    };
    clearHighlight();
    setCursorLine(covEl, ovCov, null); setCursorLine(cenEl, ovCen, null);
    const highlightSchema = ti => {
        const c = data.curves[String(pick)];
        const f = c && c.features[ti];
        const key = f && f.length ? rulesSchemaKey(f) : null;
        if (key === hiKey) return;
        if (key === null) { clearHighlight(); return; }
        hiKey = key;
        const idx = [];
        for (let i = 0; i < taus.length; i++) if (c.features[i] && c.features[i].length && rulesSchemaKey(c.features[i]) === key) idx.push(i);
        const xs = idx.map(i => taus[i] * 100);
        drawRings(covEl, ovCov, xs, idx.map(i => c.x[i] * 100));
        drawRings(cenEl, ovCen, xs, idx.map(i => c.n_centers[i]));
    };
    let lastTi = null;
    const onMoveAt = (el, clientX) => {
        const ti = floorAtPixel(el, clientX);
        if (ti === lastTi) return;   // same floor: nothing to update
        lastTi = ti;
        if (readout) readout.textContent = ti === null ? '' : describe(ti);
        setCursorLine(covEl, ovCov, ti); setCursorLine(cenEl, ovCen, ti);
        if (ti === null) clearHighlight(); else highlightSchema(ti);
    };
    const onLeave = () => { lastTi = null; if (readout) readout.textContent = ''; clearHighlight(); setCursorLine(covEl, ovCov, null); setCursorLine(cenEl, ovCen, null); };
    // Erase on OUR mouse-up, not on Plotly's click: with dragging off,
    // Plotly reports its click on mouse-DOWN, so a drag could not be told
    // from a click. The floor comes from the pointer's x through the axis
    // (nearest grid point), so a click anywhere in the plot area works.
    const floorAtPixel = (el, clientX) => {
        const xa = el._fullLayout && el._fullLayout.xaxis;
        if (!xa || typeof xa.p2d !== 'function') return null;
        const bb = el.getBoundingClientRect();
        const px = clientX - bb.left - xa._offset;
        if (px < -4 || px > xa._length + 4) return null;
        const xval = xa.p2d(px);
        if (!Number.isFinite(xval)) return null;
        let best = 0;
        for (let i = 1; i < taus.length; i++) if (Math.abs(taus[i] * 100 - xval) < Math.abs(taus[best] * 100 - xval)) best = i;
        return best;
    };
    const eraseAtFloor = ti => {
        const c = data.curves[String(pick)];
        if (!c || !c.features[ti] || !c.features[ti].length) {
            rulesState.notice = (c && c.gone && c.gone[ti])
                ? `${pick}D at ${Math.round(taus[ti] * 100)}% is already erased.`
                : `No ${pick}D schema certifies anything at ${Math.round(taus[ti] * 100)}%: nothing to erase.`;
            renderRulesErased();
            return;
        }
        eraseRulesSchema(c.features[ti].slice(), pick);
    };
    [covEl, cenEl].forEach(el => {
        if (el.removeAllListeners) { el.removeAllListeners('plotly_click'); el.removeAllListeners('plotly_hover'); el.removeAllListeners('plotly_unhover'); }
        el.onmousemove = e => onMoveAt(el, e.clientX);
        el.onmouseleave = onLeave;
        // Plotly lays a cover over the page while the button is down, so
        // the move and the release are heard on the document, not on the
        // plot; the press position is what tells a click from a drag.
        el.onmousedown = e => {
            if (e.button !== 0) return;
            const press = [e.clientX, e.clientY];
            let moved = false;
            const onMove = ev => { if (Math.hypot(ev.clientX - press[0], ev.clientY - press[1]) > 4) moved = true; };
            const onUp = ev => {
                document.removeEventListener('mousemove', onMove, true);
                document.removeEventListener('mouseup', onUp, true);
                if (moved || ev.button !== 0) return;
                const ti = floorAtPixel(el, ev.clientX);
                if (ti !== null) eraseAtFloor(ti);
            };
            document.addEventListener('mousemove', onMove, true);
            document.addEventListener('mouseup', onUp, true);
        };
    });
    // The lead strip: which d has the best coverage at each floor.
    if (strip) {
        const segs = leadSegments(data, taus, dims);
        const lo = xmin, span = 101 - xmin;
        const stops = segs.map(sg => {
            const half = taus.length > 1 ? (taus[1] - taus[0]) * 50 : 0.5;
            const a = ((taus[sg.i0] * 100 - half - lo) / span * 100).toFixed(2);
            const b = ((taus[sg.i1] * 100 + half - lo) / span * 100).toFixed(2);
            return `${TAU_CURVE_COLORS[sg.d]} ${a}% ${b}%`;
        });
        strip.style.background = stops.length ? `linear-gradient(90deg, ${stops.join(', ')})` : '#334155';
        strip.innerHTML = `<div class="ru-lead-cursor" style="left:${((cursor - lo) / span * 100).toFixed(2)}%"></div>`;
        strip.title = segs.map(sg => `${sg.d}D leads ${Math.round(taus[sg.i0] * 100)}–${Math.round(taus[sg.i1] * 100)}%`).join('; ') || 'no schema certifies anything';
    }
    if (note) {
        const nEx = rulesState.erased.length;
        note.textContent = `Pick a branch, click a curve at a floor: the schema there is erased — its points become gaps, its rules leave the list.`
            + (showAll ? ` All four branches are at full strength; a click still erases from ${pick}D.` : '')
            + (nEx ? ` ${nEx} erased.` : '')
            + (showGone ? ' Dotted hollow points are the erased ones, drawn back in for reference only — they stay out of the rules.' : '');
    }
    renderRulesErased();
}

function ruleContains(rule, cond) {
    return rule.conditions.some(c => c.feature === cond.feature && c.value === cond.value);
}

function rulesMatching(pool, conds) {
    if (!conds.length) return pool;
    return pool.filter(r => conds.every(c => ruleContains(r, c)));
}

function condLabel(c) {
    return `${c.column_label} = ${c.value_label}`;
}

function renderRules() {
    renderRulesFilter();
    renderRulesResults();
}

// --- the cascading filter ----------------------------------------------
function renderRulesFilter() {
    const chips = document.getElementById('ruChips');
    const facets = document.getElementById('ruFacets');
    const data = rulesState.data;
    if (!chips || !facets || !data) return;
    const pool = rulesPool();
    const fixed = rulesState.fixed;
    const matching = rulesMatching(pool, fixed);

    chips.innerHTML = '';
    if (!fixed.length) {
        const e = document.createElement('span'); e.className = 'ru-facet-title'; e.textContent = 'No condition fixed — pick a characteristic, then a value.';
        chips.appendChild(e);
    }
    fixed.forEach((c, i) => {
        if (i > 0) { const and = document.createElement('span'); and.className = 'target-and'; and.textContent = '∧'; chips.appendChild(and); }
        const chip = document.createElement('span');
        chip.className = 'target-chip';
        chip.innerHTML = `<span class="chip-col">${c.column_label}</span> = <span>${c.value_label}</span>`;
        const b = document.createElement('button'); b.type = 'button'; b.textContent = '✕'; b.title = 'Remove this condition';
        b.onclick = () => { rulesState.fixed = rulesState.fixed.filter(x => x !== c); rulesState.facet = null; renderRulesFilter(); };
        chip.appendChild(b);
        chips.appendChild(chip);
    });

    // Characteristics: every column that occurs in a matching rule, with
    // the number of matching rules that carry a condition on it.
    const byFeature = new Map();
    matching.forEach(r => r.conditions.forEach(c => {
        const e = byFeature.get(c.feature) || { feature: c.feature, column_label: c.column_label, n: 0, values: new Map() };
        e.n += 1;
        const v = e.values.get(c.value) || { value: c.value, value_label: c.value_label, n: 0 };
        v.n += 1; e.values.set(c.value, v);
        byFeature.set(c.feature, e);
    }));
    const fixedFeatures = new Set(fixed.map(c => c.feature));
    const features = Array.from(byFeature.values()).sort((a, b) => b.n - a.n || a.column_label.localeCompare(b.column_label));

    facets.innerHTML = '';
    const f1 = document.createElement('div'); f1.className = 'ru-facet';
    f1.innerHTML = `<div class="ru-facet-title">Characteristic · rules containing a condition on it (${matching.length} rule${matching.length === 1 ? '' : 's'} match so far)</div>`;
    if (!features.length) {
        const e = document.createElement('div'); e.className = 'ru-facet-title'; e.textContent = 'No rule contains all fixed conditions.'; f1.appendChild(e);
    }
    features.forEach(f => {
        const row = document.createElement('div');
        const isFixed = fixedFeatures.has(f.feature);
        row.className = 'ru-row' + (isFixed ? ' fixed' : '') + (rulesState.facet === f.feature ? ' selected' : '');
        row.innerHTML = `<span>${f.column_label}</span><span class="ru-count">${f.n} rule${f.n === 1 ? '' : 's'}</span>`;
        if (!isFixed) row.onclick = () => { rulesState.facet = (rulesState.facet === f.feature) ? null : f.feature; renderRulesFilter(); };
        f1.appendChild(row);
    });
    facets.appendChild(f1);

    if (rulesState.facet !== null && byFeature.has(rulesState.facet) && !fixedFeatures.has(rulesState.facet)) {
        const f = byFeature.get(rulesState.facet);
        const f2 = document.createElement('div'); f2.className = 'ru-facet';
        f2.innerHTML = `<div class="ru-facet-title">Value of ${f.column_label} · rules containing it</div>`;
        Array.from(f.values.values()).sort((a, b) => b.n - a.n || a.value_label.localeCompare(b.value_label)).forEach(v => {
            const row = document.createElement('div');
            row.className = 'ru-row';
            row.innerHTML = `<span>${v.value_label}</span><span class="ru-count">${v.n}</span>`;
            row.onclick = () => {
                rulesState.fixed = rulesState.fixed.concat([{ feature: f.feature, value: v.value, column_label: f.column_label, value_label: v.value_label }]);
                rulesState.facet = null;
                renderRulesFilter();
            };
            f2.appendChild(row);
        });
        facets.appendChild(f2);
    }
}

function applyRulesFilter() {
    rulesState.applied = rulesState.fixed.slice();
    rulesState.bar = null;
    rulesState.page = 0;
    renderRulesResults();
}

function clearRulesFilter() {
    rulesState.fixed = [];
    rulesState.applied = [];
    rulesState.facet = null;
    rulesState.bar = null;
    rulesState.page = 0;
    renderRules();
}

// --- bars and cards -------------------------------------------------------
function rulesBinIndex(purity) {
    return landscapeBinIndex(Math.max(purity, 1e-12), RULES_N_BINS);
}

function renderRulesResults() {
    const data = rulesState.data;
    const bars = document.getElementById('ruBars');
    const total = document.getElementById('ruTotal');
    const legend = document.getElementById('ruLegend');
    const note = document.getElementById('ruNote');
    const summary = document.getElementById('ruSummary');
    const title = document.getElementById('ruBarsTitle');
    if (!data || !bars) return;
    const absence = data.direction === 'absence';
    const pool = rulesPool();
    const applied = rulesState.applied;
    const shown = rulesMatching(pool, applied);
    const dims = rulesDims();
    const dimList = [1, 2, 3, 4].filter(d => dims[d]);

    if (title) title.textContent = absence ? 'Rules by share of rows WITHOUT the value' : 'Rules by share of the value';
    if (legend) {
        legend.innerHTML = dimList.map(d => `<span><i style="background:${TAU_CURVE_COLORS[d]}"></i>${d}D</span>`).join('');
    }
    if (summary) {
        const perD = [1, 2, 3, 4].filter(d => dims[d]).map(d => `${d}D: ${pool.filter(r => r.d === d).length}`).join(', ');
        const nSchemas = new Set(pool.map(r => rulesSchemaKey(r.features))).size;
        summary.textContent = `${pool.length} rule${pool.length === 1 ? '' : 's'} from ${nSchemas} schema${nSchemas === 1 ? '' : 's'} on the curves (${perD}); min rows ${data.min_rows}`;
    }
    if (note) {
        const filt = applied.length ? `Showing the ${shown.length} that contain ${applied.map(condLabel).join(' ∧ ')} — with or without further conditions.` : `Showing all ${shown.length}.`;
        note.innerHTML = `A rule is one cell of a schema on the curves (the best of its dimensionality at some purity floor; the four winners among them): its conditions, the rows that satisfy them and the share of «${currentValueLabel() || 'the value'}» among those rows${absence ? ' (under Absence: the share of rows without it)' : ''}. `
            + `Certified at the current boundary (${Math.round(data.tau * 100)}%) marks a discrete centre. ${filt} `
            + `A refinement whose share is lower than its parent's is kept: learning more changed the probability, and the card says by how much.`;
    }

    // Purity intervals (0,10], ..., (90,100]; counts per d.
    const counts = Array.from({ length: RULES_N_BINS }, () => ({}));
    shown.forEach(r => { const ix = rulesBinIndex(r.purity); counts[ix][r.d] = (counts[ix][r.d] || 0) + 1; });
    const totals = counts.map(c => Object.values(c).reduce((a, b) => a + b, 0));
    const maxCount = Math.max(1, ...totals);
    const anchor = axisAnchorPct();
    const tauPct = Math.round(data.tau * 100);
    bars.innerHTML = '';
    for (let ix = RULES_N_BINS - 1; ix >= 0; ix--) {
        const lo = ix * 10, hi = (ix + 1) * 10;
        const label = document.createElement('div');
        label.className = 'ru-bar-label' + (anchor !== null && hi <= anchor ? ' below-anchor' : '') + (tauPct > lo && tauPct <= hi ? ' at-tau' : '') + (rulesState.bar === ix ? ' selected' : '');
        label.textContent = `${ix === 0 ? '[' : '('}${lo},${hi}]%`;
        label.title = ((tauPct > lo && tauPct <= hi) ? `The certified boundary (${tauPct}%) lies in this interval. ` : (anchor !== null && hi <= anchor ? `Below the base rate (${anchor}%). ` : ''))
            + (totals[ix] ? 'Click: list only this interval\'s rules.' : '');
        if (totals[ix]) label.onclick = () => { rulesState.bar = (rulesState.bar === ix) ? null : ix; rulesState.page = 0; renderRulesResults(); };
        const bar = document.createElement('div');
        bar.className = 'ru-bar' + (totals[ix] === 0 ? ' empty' : '') + (rulesState.bar === ix ? ' selected' : '');
        bar.style.width = `${(100 * totals[ix] / maxCount).toFixed(2)}%`;
        dimList.forEach(d => {
            const n = counts[ix][d] || 0;
            if (!n) return;
            // The rules of one d in one interval; the schemas behind them
            // (one per d today - the winners - but written for any number).
            const schemas = new Map();
            shown.filter(r => r.d === d && rulesBinIndex(r.purity) === ix).forEach(r => {
                const k = rulesSchemaKey(r.features);
                const e = schemas.get(k) || { features: r.features, n: 0 };
                e.n += 1; schemas.set(k, e);
            });
            const seg = document.createElement('div');
            seg.className = 'ru-seg';
            seg.style.width = `${(100 * n / totals[ix]).toFixed(3)}%`;
            seg.style.background = TAU_CURVE_COLORS[d];
            const desc = Array.from(schemas.values()).map(e => `${e.features.map(j => data.feature_names[j]).join(' + ')} (${e.n})`).join('; ');
            seg.title = `${d}D: ${n} rule${n === 1 ? '' : 's'} — ${desc}. Click: erase this schema from every interval.`;
            seg.onclick = (ev) => {
                ev.stopPropagation();
                const list = Array.from(schemas.values());
                if (list.length === 1) { eraseRulesSchema(list[0].features, d); return; }
                // Several schemas of one d in one segment: erase the one
                // with most rules here, and say so.
                list.sort((a, b) => b.n - a.n);
                eraseRulesSchema(list[0].features, d);
            };
            bar.appendChild(seg);
        });
        const count = document.createElement('div');
        count.className = 'ru-bar-count';
        count.textContent = totals[ix] ? String(totals[ix]) : '';
        bars.appendChild(label); bars.appendChild(bar); bars.appendChild(count);
    }

    const listed = rulesState.bar === null ? shown : shown.filter(r => rulesBinIndex(r.purity) === rulesState.bar);
    if (total) {
        total.textContent = `Total centres: ${shown.length}` + (rulesState.bar !== null
            ? ` · ${listed.length} in (${rulesState.bar * 10},${(rulesState.bar + 1) * 10}]%` : '')
            + ` · certified: ${listed.filter(r => r.certified).length}`;
    }
    renderRulesErased();
    renderRulesCards(listed, absence);
}

// The erased schemas as chips (each restorable), Undo / Reset, and the
// notice of the last erase.
function renderRulesErased() {
    ['ruErased', 'ruErasedSide'].forEach(id => renderRulesErasedInto(document.getElementById(id)));
}

function renderRulesErasedInto(box) {
    if (!box) return;
    box.innerHTML = '';
    const data = rulesState.data;
    const erased = rulesState.erased;
    if ((!erased.length && !rulesState.notice) || (!data && !rulesCurves.data)) { box.style.display = 'none'; return; }
    box.style.display = '';
    const head = document.createElement('div');
    head.className = 'ru-erased-head';
    const nRules = erased.reduce((a, e) => a + e.n_rules, 0);
    head.innerHTML = `<span>${erased.length ? `Erased: ${erased.length} schema${erased.length === 1 ? '' : 's'} · ${nRules} rule${nRules === 1 ? '' : 's'}` : 'Nothing erased'}</span>`;
    // The buttons on their own row: beside the text they were pushed out
    // of the narrow panel whenever the text wrapped.
    const actions = document.createElement('div');
    actions.className = 'ru-filter-actions';
    const undo = document.createElement('button'); undo.className = 'landscape-open'; undo.textContent = 'Undo'; undo.disabled = !rulesState.history.length; undo.onclick = undoRulesErase;
    const reset = document.createElement('button'); reset.className = 'landscape-open'; reset.textContent = 'Reset'; reset.disabled = !erased.length; reset.onclick = resetRulesErase;
    actions.appendChild(undo); actions.appendChild(reset);
    box.appendChild(head);
    if (erased.length) box.appendChild(actions);
    if (erased.length) {
        const chips = document.createElement('div'); chips.className = 'ru-chips';
        erased.forEach(e => {
            const chip = document.createElement('span'); chip.className = 'target-chip';
            chip.innerHTML = `<span class="chip-col" style="color:${TAU_CURVE_COLORS[e.d]}">${e.d}D</span> ${e.names.join(' + ')} <span class="ru-count">(${e.n_rules})</span>`;
            const b = document.createElement('button'); b.type = 'button'; b.textContent = '✕'; b.title = 'Restore this schema';
            b.onclick = () => {
                rulesState.erased = rulesState.erased.filter(x => x !== e);
                rulesState.history = rulesState.history.filter(x => x !== e);
                rulesState.notice = `Restored the ${e.d}D schema ${e.names.join(' + ')}.`;
                renderRules();
                renderRulesCurves();
            };
            chip.appendChild(b); chips.appendChild(chip);
        });
        box.appendChild(chips);
    }
    if (rulesState.notice) {
        const n = document.createElement('div'); n.className = 'ru-notice'; n.textContent = rulesState.notice; box.appendChild(n);
    }
}

function renderRulesCards(listed, absence) {
    const cards = document.getElementById('ruCards');
    const more = document.getElementById('ruMore');
    if (!cards) return;
    cards.innerHTML = '';
    const start = rulesState.page * RULES_PAGE;
    const page = listed.slice(start, start + RULES_PAGE);
    page.forEach(r => {
        const card = document.createElement('div');
        card.className = 'cg-card ru-card';
        const cond = r.conditions.map(c => `<b>${c.column_label}</b> = ${c.value_label}`).join(' ∧ ');
        const parents = (r.parents || []).map((g, p) => {
            const delta = (r.purity - g.purity) * 100;
            const cls = delta > 0 ? 'up' : (delta < 0 ? 'down' : '');
            const sign = delta > 0 ? '+' : '';
            return `<div>without ${condLabel(r.conditions[p])}: ${(g.purity * 100).toFixed(1)}% of ${g.n.toLocaleString()} rows <span class="${cls}">(${sign}${delta.toFixed(1)} pp)</span></div>`;
        }).join('');
        card.innerHTML = `
            <div class="ru-purity"><span>${(r.purity * 100).toFixed(1)}%</span><span class="ru-d" style="background:${TAU_CURVE_COLORS[r.d]}">${r.d}D</span></div>
            <div class="ru-cond">${cond}</div>
            <div class="ru-meta">schema ${r.features.map(j => (rulesState.data.feature_names || [])[j]).join(' + ')}</div>
            <div class="ru-meta">${r.n.toLocaleString()} rows · ${r.k.toLocaleString()} with the value${r.purity_lower > 0 ? ` · lower bound ${(r.purity_lower * 100).toFixed(1)}%` : ''}</div>
            <div class="ru-meta">${r.certified ? `<span class="ru-cert${absence ? ' absence' : ''}">● certified centre at the current boundary</span>` : 'not a centre at the current boundary'}</div>
            ${parents ? `<div class="ru-parents">${parents}</div>` : ''}
            <div class="ru-actions"><button type="button" class="landscape-open" title="Render this rule's schema as a lattice">open schema</button></div>`;
        card.querySelector('.landscape-open').onclick = () => openSchemaFromLandscape(r.features);
        cards.appendChild(card);
    });
    if (more) {
        more.innerHTML = '';
        if (listed.length) {
            more.textContent = `${start + 1}–${start + page.length} of ${listed.length}`;
            if (start > 0) {
                const b = document.createElement('button'); b.className = 'landscape-open'; b.textContent = '← previous';
                b.onclick = () => { rulesState.page -= 1; renderRulesResults(); }; more.appendChild(b);
            }
            if (start + page.length < listed.length) {
                const b = document.createElement('button'); b.className = 'landscape-open'; b.textContent = 'next →';
                b.onclick = () => { rulesState.page += 1; renderRulesResults(); }; more.appendChild(b);
            }
        } else {
            more.textContent = 'No rule matches.';
        }
    }
}

// ---------------------------------------------------------------------------
// Dataset screen (Project_Master_Document.md Section 4.12)
// ---------------------------------------------------------------------------
// What the framework will do with each column, and which columns (nearly)
// determine each other - computed from the columns alone, so the reader can
// read it and exclude a column BEFORE choosing a target and before seeing a
// single result. Exclusions travel as `drop` on every request, are part of
// the analysis identity and are echoed in the response; `prune` additionally
// skips candidates that an exact dependency makes renamings of a smaller
// schema (`/api/screen`, `vsf.screen`).
let screenState = { data: null, key: null, pending: false, error: null };
let excludedColumns = [];      // column names the reader excluded
let pruneDependent = false;    // skip candidates that are renamings
let screenDirty = false;       // exclusions changed after the current analysis

function screenRequestOptions(body) {
    // `drop`/`prune` on every analysis request: the feature space and the
    // candidate family are part of what a cached response is keyed by.
    body.drop = excludedColumns.slice();
    body.prune = pruneDependent;
    return body;
}

function screenMinStrength() {
    const el = document.getElementById('screenStrength');
    let v = el ? Number(el.value) : 90;
    if (!Number.isFinite(v)) v = 90;
    return Math.max(0, Math.min(100, Math.round(v))) / 100;
}

// Exactly one container is filled at a time - the guide screen's panel
// before the first analysis, the Data tab's panel after it - so the controls
// inside it keep unique ids.
function screenContainers() {
    const welcome = document.getElementById('welcomeState');
    const inWelcome = !!(welcome && welcome.style.display !== 'none');
    const el = document.getElementById(inWelcome ? 'screenWelcome' : 'screenPanel');
    const other = document.getElementById(inWelcome ? 'screenPanel' : 'screenWelcome');
    if (other) other.innerHTML = '';
    return el ? [el] : [];
}

async function refreshScreen(force) {
    const body = { min_strength: screenMinStrength() };
    if (lastTargetCol !== null && lastCriterion !== null) {
        body.target = lastTargetCol;
        body.criterion = lastCriterion;
        if (targetAlso.length) body.also = targetAlsoPairs();
    }
    const key = JSON.stringify(body);
    if (!force && screenState.key === key && screenState.data) { renderScreen(); return; }
    screenState.pending = true;
    screenState.error = null;
    renderScreen();
    try {
        const res = await fetch('/api/screen', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        // A running server older than these assets has no /api/screen route
        // and answers the POST with its HTML 404 page: the static files are
        // read from disk per request, the routes are not.
        const ctype = res.headers.get('content-type') || '';
        if (!ctype.includes('json')) {
            screenState.pending = false;
            screenState.error = 'this page is newer than the server process: /api/screen answered with '
                + (res.status === 404 ? 'a 404 page' : `content-type ${ctype || 'unknown'}`)
                + '. Restart run.py (the assets reload from disk, the routes do not).';
            renderScreen();
            return;
        }
        const data = await res.json();
        screenState.pending = false;
        if (!res.ok) { screenState.error = data.error || String(res.status); renderScreen(); return; }
        screenState.key = key;
        screenState.data = data;
        renderScreen();
    } catch (err) {
        screenState.pending = false;
        screenState.error = err.message;
        renderScreen();
    }
}

function toggleExcludedColumn(name) {
    const i = excludedColumns.indexOf(name);
    if (i >= 0) excludedColumns.splice(i, 1);
    else excludedColumns.push(name);
    screenDirty = currentBranchesResponse !== null;
    renderScreen();
}

function onPruneToggle(el) {
    pruneDependent = !!(el && el.checked);
    screenDirty = currentBranchesResponse !== null;
    renderScreen();
}

async function applyScreenChanges() {
    if (lastTargetCol === null) return;
    screenDirty = false;
    await runAnalysis(lastTargetCol, lastCriterion, null, { preserveBranch: true });
    refreshScreen(true);
}

const SCREEN_FLAG_TEXT = {
    constant: 'one category only — every schema holding it has the partition of the schema without it',
    exceeds_capacity: 'more categories than the grid capacity — even its own 1D grid is coarsened',
    dominant_level: 'one category covers almost everything — every other cell is small',
    mostly_missing: 'more than half the rows are missing',
    placeholder_level: 'a category whose NAME reads as “no value recorded” covers a large share of the rows; a centre built on it describes how the data were collected, unless that name means a real value here',
};

function screenPairHtml(p, dataTarget) {
    const cls = { equivalent: 'cg-tag', exact: 'cg-tag cg-tag-ref', approximate: 'cg-tag cg-tag-warn' }[p.kind] || 'cg-tag';
    const dir = (a, b, delta, strength, exceptions) => `
        <div class="sc-dir"><strong>${escHtml(a)}</strong> → ${escHtml(b)}:
            <span class="sc-num">${pct(delta, 1)}</span> of rows
            <span class="cg-feat">(${exceptions} exception${exceptions === 1 ? '' : 's'}; ${pct(strength, 0)} above the majority rule)</span></div>`;
    const meaning = {
        equivalent: 'The two columns split the rows identically — one is a renaming of the other. Excluding either changes nothing except the size of the search.',
        exact: 'One column fixes the other. A schema holding both has exactly the partition of the schema without the determined one; those candidates are renamings (switch on “skip renaming schemas” below). Excluding the determined column is NOT free: on its own it is a coarser grid the other cannot reproduce.',
        approximate: 'Nothing is proved: the exception rows below break the dependency. Decide yourself whether the two columns say the same thing.',
    }[p.kind];
    return `<div class="sc-pair">
        <div class="sc-pair-head"><span class="${cls}">${escHtml(p.kind)}</span>
            <strong>${escHtml(p.a)}</strong> ↔ <strong>${escHtml(p.b)}</strong></div>
        ${dir(p.a, p.b, p.delta_ab, p.strength_ab, p.exceptions_ab)}
        ${dir(p.b, p.a, p.delta_ba, p.strength_ba, p.exceptions_ba)}
        <div class="sc-meaning">${meaning}</div>
        <div class="sc-actions">
            <button type="button" class="landscape-open" data-exclude="${escHtml(p.a)}">${excludedColumns.includes(p.a) ? 'keep' : 'exclude'} ${escHtml(p.a)}</button>
            <button type="button" class="landscape-open" data-exclude="${escHtml(p.b)}">${excludedColumns.includes(p.b) ? 'keep' : 'exclude'} ${escHtml(p.b)}</button>
        </div></div>`;
}

function renderScreen() {
    const els = screenContainers();
    if (!els.length) return;
    const d = screenState.data;
    let html;
    if (screenState.error) {
        html = `<div class="cg-warn">Data screen: ${escHtml(screenState.error)}</div>`;
    } else if (!d) {
        html = `<div class="cg-busy">${screenState.pending ? 'Screening the columns…' : 'Not screened yet.'}</div>`;
    } else {
        const targetCols = new Set(d.target_columns || []);
        const flagged = d.columns.filter(c => c.flags.length && !targetCols.has(c.name));
        const pairs = d.pairs || [];
        const rows = d.columns.map(c => {
            const isTarget = targetCols.has(c.name);
            const excluded = excludedColumns.includes(c.name);
            const flags = c.flags.map(f => `<span class="cg-tag cg-tag-warn" title="${escHtml(SCREEN_FLAG_TEXT[f] || '')}">${escHtml(f.replace(/_/g, ' '))}</span>`).join(' ');
            return `<tr class="${excluded ? 'sc-excluded' : ''}">
                <td>${isTarget
                    ? '<span class="cg-feat" title="A column of the target is out of the feature space already">target</span>'
                    : `<input type="checkbox" data-exclude-box="${escHtml(c.name)}" ${excluded ? '' : 'checked'} title="Unchecked columns are excluded from the feature space of every search">`}</td>
                <td class="sc-name">${escHtml(c.name)}</td>
                <td class="sc-num">${c.n_levels}</td>
                <td>${escHtml(c.largest_level)} <span class="cg-feat">${pct(c.largest_level_share, 0)}</span></td>
                <td>${c.placeholder_level ? `<span class="cg-warn" title="Its name reads as a placeholder for a missing value">${escHtml(c.placeholder_level)}</span> <span class="cg-feat">${pct(c.placeholder_share, 0)}</span>` : ''}</td>
                <td class="sc-num">${c.n_missing || 0}</td>
                <td class="sc-num">${c.n_singleton_levels}</td>
                <td>${flags}</td></tr>`;
        }).join('');
        const leak = (d.target_report || []).slice(0, 3).map(e =>
            `<li><strong>${escHtml(e.feature)}</strong>: fixes the target on ${pct(e.delta, 1)} of rows
              <span class="cg-feat">(${e.exceptions} exceptions; ${pct(e.strength, 0)} above the majority rule)</span>${e.exact ? ' — <span class="cg-warn">exact: this column IS the target under another name</span>' : ''}</li>`).join('');
        html = `
            <div class="sc-head">
                <div class="sc-title">Data screen</div>
                <div class="sc-summary"><strong>${d.n_rows}</strong> rows · <strong>${d.n_columns}</strong> columns ·
                    grid capacity <strong>${d.grid_capacity}</strong> occupied cells per schema ·
                    <strong>${pairs.length}</strong> dependent column pair${pairs.length === 1 ? '' : 's'} ·
                    <strong>${flagged.length}</strong> flagged column${flagged.length === 1 ? '' : 's'}</div>
            </div>
            <div class="cg-explain">
                Everything here is computed from the columns alone, before any target and any search, so excluding a
                column cannot be a reaction to a result. Exclusions are part of the analysis: they change the candidate
                family and the multiplicity correction, and they are reported with every number.
                ${d.pairs_skipped ? `<span class="cg-warn">${escHtml(d.pairs_skipped)}</span>` : ''}
            </div>
            <div class="sc-controls">
                <label title="An inexact pair is listed when the determination is at least this far above what the majority rule alone achieves.">Report pairs from
                    <input type="number" id="screenStrength" min="0" max="100" step="1" value="${Math.round(d.min_strength * 100)}" onchange="refreshScreen(true)"> % above chance</label>
                <label title="Skip candidate schemas that an exact dependency makes a renaming of a smaller schema: the partition, the coverage and the centre count are identical, only the dimensionality is higher. Never applied where the grid-capacity rule would coarsen the two differently.">
                    <input type="checkbox" id="screenPrune" ${pruneDependent ? 'checked' : ''} onchange="onPruneToggle(this)"> skip renaming schemas in the search</label>
                <span class="sc-excluded-list">Excluded: ${excludedColumns.length ? excludedColumns.map(escHtml).join(', ') : 'none'}</span>
                ${screenDirty ? '<button type="button" class="landscape-open sc-apply" onclick="applyScreenChanges()">re-run the analysis</button>' : ''}
            </div>
            ${leak ? `<div class="sc-section"><div class="sc-sub">How well a single column fixes the target (leakage)</div><ul class="sc-list">${leak}</ul></div>` : ''}
            <div class="sc-section">
                <div class="sc-sub">Columns that (nearly) determine each other</div>
                ${pairs.length ? `<div class="sc-pairs">${pairs.map(screenPairHtml).join('')}</div>`
                : '<div class="cg-busy">No pair reaches the reporting floor.</div>'}
            </div>
            <div class="sc-section">
                <div class="sc-sub">Columns</div>
                <div class="cg-table-wrap"><table class="cg-table sc-table">
                    <thead><tr><th title="Unchecked columns are excluded from every search">use</th><th>column</th>
                        <th title="Distinct categories, missing values counted as one">categories</th>
                        <th>largest category</th>
                        <th title="A category whose name reads as a placeholder for a missing value, and its share">placeholder?</th>
                        <th title="Rows with a missing value">missing</th>
                        <th title="Categories holding exactly one row: they can only ever be single-object cells">singletons</th>
                        <th>flags</th></tr></thead>
                    <tbody>${rows}</tbody></table></div>
            </div>`;
    }
    els.forEach(el => {
        el.innerHTML = html;
        el.querySelectorAll('[data-exclude]').forEach(b => {
            b.onclick = () => toggleExcludedColumn(b.dataset.exclude);
        });
        el.querySelectorAll('[data-exclude-box]').forEach(b => {
            b.onchange = () => toggleExcludedColumn(b.dataset.excludeBox);
        });
    });
}

// ---------------------------------------------------------------------------
// Redundancy tab (Project_Master_Document.md Section 4.11)
// ---------------------------------------------------------------------------
// Different characteristics can describe the same objects. Two centres are
// "the same" at threshold t when EACH holds at least t of the other's rows
// (mutual containment, |A ∩ B| / max(|A|, |B|) >= t).
//
// Two scopes:
//   * Selected branch (default): the centres of the branch (or opened
//     schema) on screen; for each, every centre of every other scored schema
//     that is t-similar to IT directly (`/api/centers/branch`).
//   * All centres: every centre of every scored schema, grouped by leader
//     clustering around a representative - fewest characteristics, then the
//     highest certified purity bound (`/api/centers/groups`, `/group`).
// The server builds the centre catalogue once per (certificate, min rows);
// moving the threshold only regroups. Descriptive only: no centre, coverage
// or p-value changes.
let centerGroupsState = {
    key: null,          // JSON of the request that produced the data (without paging)
    data: null,         // /api/centers/groups or /api/centers/branch response
    openGroup: null,    // all-scope: representative set id of the expanded card
    openAnchor: null,   // branch-scope: cell code of the expanded branch centre
    detail: null,       // the expanded card's member list (either scope)
    seq: 0,             // request counter: stale responses are dropped
    timer: null,        // debounce of the threshold input
};
const CG_PAGE = 12;
const CG_MEMBER_PAGE = 50;

function escHtml(value) {
    return String(value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

function pct(x, digits = 1) {
    return (x === null || x === undefined || !Number.isFinite(x)) ? '—' : `${(x * 100).toFixed(digits)}%`;
}

function resetCenterGroupsState() {
    centerGroupsState.data = null;
    centerGroupsState.key = null;
    centerGroupsState.openGroup = null;
    centerGroupsState.openAnchor = null;
    centerGroupsState.detail = null;
}

function centerGroupsScope() {
    const active = document.querySelector('.cg-scope-btn.active');
    return active && active.dataset.scope === 'all' ? 'all' : 'branch';
}

function setCenterGroupsScope(scope) {
    document.querySelectorAll('.cg-scope-btn').forEach(b => b.classList.toggle('active', b.dataset.scope === scope));
    const allOnly = document.getElementById('cgAllOnly');
    if (allOnly) allOnly.style.display = scope === 'all' ? 'contents' : 'none';
    resetCenterGroupsState();
    setCenterGroupHighlight(null);
    refreshCenterGroups(0);
}

function centerGroupsControls() {
    const tEl = document.getElementById('cgThreshold');
    const mEl = document.getElementById('cgMinRows');
    const sEl = document.getElementById('cgSort');
    const fEl = document.getElementById('cgFilter');
    const kEl = document.getElementById('cgKinship');
    let t = tEl ? Number(tEl.value) : 80;
    if (!Number.isFinite(t)) t = 80;
    t = Math.max(50, Math.min(100, t));
    let m = mEl ? Math.round(Number(mEl.value)) : 10;
    if (!Number.isFinite(m) || m < 1) m = 1;
    const k = kEl ? kEl.value : 'all';
    return {
        threshold: Math.round(t) / 100,
        min_rows: m,
        sort: sEl && sEl.value === 'members' ? 'members' : 'coverage',
        filter: fEl ? fEl.value : 'all',
        kinship: (k === 'related' || k === 'unrelated') ? k : 'all',
        scope: centerGroupsScope(),
    };
}

// The value the counts refer to: the chosen value under presence, its
// complement under absence (the indicator the search certified).
function centerGroupsValueLabel() {
    const crit = currentBranchesResponse ? currentBranchesResponse.criterion : null;
    const name = crit === null || crit === undefined ? 'value' : `«${crit}»`;
    return readDirection() === 'absence' ? `not ${name}` : name;
}

// Feature indices of the schema on screen: the selected branch, or the
// schema opened from the landscape / this tab.
function centerGroupsBranchFeatures() {
    if (currentBranchesResponse && currentBranchesResponse.schema) return currentBranchesResponse.schema.features.slice();
    if (currentPayload && currentPayload.selected_feature_indices) return currentPayload.selected_feature_indices.slice();
    return null;
}

function centerGroupsRequest() {
    const p = landscapeParams();
    if (!p) return null;
    const c = centerGroupsControls();
    const body = Object.assign({}, p, { threshold: c.threshold, min_rows: c.min_rows, kinship: c.kinship });
    if (c.scope === 'branch') {
        const feats = centerGroupsBranchFeatures();
        if (!feats || !feats.length) return null;
        body.features = feats;
        return { url: '/api/centers/branch', body, scope: 'branch' };
    }
    body.sort = c.sort;
    if (c.filter === 'd') {
        const d = currentPayload && currentPayload.metrics ? Number(currentPayload.metrics.d) : null;
        if (Number.isFinite(d)) body.filter_d = d;
    }
    return { url: '/api/centers/groups', body, scope: 'all' };
}

function onCenterGroupsControl(debounce) {
    if (centerGroupsState.timer) clearTimeout(centerGroupsState.timer);
    if (debounce) {
        centerGroupsState.timer = setTimeout(() => refreshCenterGroups(0), 300);
    } else {
        refreshCenterGroups(0);
    }
}

async function refreshCenterGroups(offset) {
    if (viewMode !== 'redundancy') return;
    const list = document.getElementById('cgList');
    const summary = document.getElementById('cgSummary');
    const req = centerGroupsRequest();
    if (!req || !list) return;
    const body = Object.assign({ limit: req.scope === 'all' ? CG_PAGE : CG_MEMBER_PAGE, offset: req.scope === 'all' ? (offset || 0) : 0 }, req.body);
    const key = JSON.stringify([req.url, req.body]);
    const seq = ++centerGroupsState.seq;
    if (!centerGroupsState.data || centerGroupsState.key !== key) {
        list.innerHTML = '<div class="cg-busy">Comparing the centres of every schema… (computed once for these settings)</div>';
        if (summary) summary.textContent = '';
    }
    try {
        const res = await fetch(req.url, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const data = await res.json();
        if (seq !== centerGroupsState.seq) return;
        if (!res.ok) {
            list.innerHTML = `<div class="cg-warn">${escHtml(data.error || res.status)}</div>`;
            return;
        }
        const same = centerGroupsState.key === key;
        centerGroupsState.key = key;
        centerGroupsState.data = data;
        data._scope = req.scope;
        if (req.scope === 'all') {
            const stillThere = same && data.groups.some(g => g.group === centerGroupsState.openGroup);
            if (!stillThere) { centerGroupsState.openGroup = null; centerGroupsState.detail = null; setCenterGroupHighlight(null); }
            renderCenterGroups();
        } else {
            const a = same && data.anchors.find(x => x.cell === centerGroupsState.openAnchor);
            if (!a) { centerGroupsState.openAnchor = null; centerGroupsState.detail = null; setCenterGroupHighlight(null); }
            else centerGroupsState.detail = { anchor: a };
            renderBranchRedundancy();
        }
    } catch (err) {
        if (seq === centerGroupsState.seq) list.innerHTML = `<div class="cg-warn">Request failed: ${escHtml(err.message)}</div>`;
    }
}

// A centre's conjunction as a two-column list, one characteristic per line,
// characteristics in alphabetical order so that two centres built on the
// same characteristics line up the same way wherever they are shown. A
// characteristic whose categories the grid-capacity rule merged in this
// schema lists all of them (the cell holds any of them).
//
// `ref` (optional): the conjunction this one is compared with. Lines that the
// reference does not have - another characteristic, or the same one with
// other categories - are highlighted, so a table of alternatives shows at a
// glance what each one describes differently.
function conditionsHtml(conditions, ref) {
    const sorted = conditions.slice().sort((a, b) =>
        String(a.feature).localeCompare(String(b.feature), undefined, { sensitivity: 'base', numeric: true }));
    const refKey = ref ? new Map(ref.map(c => [c.feature, c.values.join('\u0000')])) : null;
    const rows = sorted.map(c => {
        const merged = c.values.length > 1;
        const vals = c.values.map(escHtml).join(', ');
        const differs = refKey !== null && refKey.get(c.feature) !== c.values.join('\u0000');
        const cls = [merged ? 'cg-merged' : '', differs ? 'cg-diff' : ''].filter(Boolean).join(' ');
        const tip = [
            merged ? 'Any of these categories: the grid-capacity rule merged them in this schema.' : '',
            differs ? 'Not in the centre this one is compared with.' : '',
        ].filter(Boolean).join(' ');
        return `<dt${differs ? ' class="cg-diff"' : ''}>${escHtml(c.feature)}</dt><dd${cls ? ` class="${cls}"` : ''}${tip ? ` title="${escHtml(tip)}"` : ''}>${vals}</dd>`;
    }).join('');
    return `<dl class="cg-conds">${rows}</dl>`;
}

// A card's numbers as a two-column list, one fact per line: label left,
// value right, aligned with the conjunction above it. Entries whose value is
// null are dropped, so a card never shows an empty row.
function statsHtml(entries) {
    const rows = entries.filter(e => e && e[1] !== null && e[1] !== undefined).map(e => {
        const [label, value, title] = e;
        return `<dt${title ? ` title="${escHtml(title)}"` : ''}>${escHtml(label)}</dt><dd>${value}</dd>`;
    }).join('');
    return rows ? `<dl class="cg-stats">${rows}</dl>` : '';
}

// Dimensionality badge + conjunction (+ optional tags under it).
function centreHtml(d, conditions, tags, ref) {
    return `<div class="cg-centre"><span class="cg-d">${d}D</span><div class="cg-centre-body">${conditionsHtml(conditions, ref)}${tags ? `<div class="cg-tags">${tags}</div>` : ''}</div></div>`;
}

// ---- Similarity histograms -------------------------------------------------
// Three scopes, all drawn by the same routine and all reading the same axis
// (mutual containment, from the pair-store floor to 1, identity on its own
// bar at the right):
//   "all"    — every centre of every scored schema, one value each: how close
//              its nearest other centre is. Shown in the All-centres scope.
//   "branch" — the same value, but only for the centres of the selected
//              branch. Shown above the cards of the Selected-branch scope.
//   "center" — ONE centre against every other centre of the run (not just its
//              nearest). Shown inside that centre's card: the bars at or above
//              the dashed line are exactly its "other descriptions".
function histogramBars(h, t) {
    const bars = h.counts.map((c, i) => ({ lo: h.edges[i], hi: h.edges[i + 1], c }));
    bars.push({ lo: 1, hi: 1, c: h.identical, identical: true });
    let atOrAbove = h.identical;
    bars.forEach(b => { if (!b.identical && b.lo >= t - 1e-9) atOrAbove += b.c; });
    return { bars, atOrAbove };
}

// `cfg`: bar width, gap, height, and whether the floor/1.0 axis labels are
// drawn. `noun` names what one bar counts in the tooltips. `cfg.part` is a
// second histogram on the same bins, a subset of `h`: it is drawn solid at the
// foot of each bar and the remainder — what `h` has and `part` does not —
// faded above it, so a stacked bar reads as "this much is on screen, this much
// is hidden" without changing the totals.
function histogramSvg(h, t, cfg) {
    const { bars } = histogramBars(h, t);
    const part = cfg.part ? histogramBars(cfg.part, t).bars : null;
    const bw = cfg.bw, gap = cfg.gap, H = cfg.h, padB = cfg.labels ? 16 : 2;
    const noun = cfg.noun || 'centres';
    const maxC = Math.max(1, ...bars.map(b => b.c));
    const W = bars.length * (bw + gap);
    const rx = bw >= 10 ? 2 : 1;
    let svg = `<svg width="${W + 4}" height="${H + padB}" role="img" aria-label="${escHtml(cfg.aria || 'Similarity histogram')}">`;
    bars.forEach((b, i) => {
        const x = i * (bw + gap);
        const hgt = b.c > 0 ? Math.max(2, Math.round(H * b.c / maxC)) : 0;
        const on = b.identical || b.lo >= t - 1e-9;
        const fill = b.identical ? '#10b981' : (on ? '#f59e0b' : '#475569');
        const label = b.identical ? 'identical rows (similarity = 1)' : `similarity ${b.lo.toFixed(2)}–${b.hi.toFixed(2)}`;
        const solid = part ? part[i].c : b.c;
        const tip = part && solid !== b.c
            ? `${b.c} ${noun} at ${label} — ${solid} in the cards shown, ${b.c - solid} in the centres the filter hides`
            : `${b.c} ${noun} at ${label}`;
        // whole bar faded, then the listed part solid on top of it
        svg += `<rect x="${x}" y="${H - hgt}" width="${bw}" height="${hgt}" rx="${rx}" fill="${fill}"`
            + `${part ? ' opacity="0.32"' : ''}><title>${escHtml(tip)}</title></rect>`;
        if (part && solid > 0) {
            const sh = Math.max(2, Math.round(H * solid / maxC));
            svg += `<rect x="${x}" y="${H - sh}" width="${bw}" height="${sh}" rx="${rx}" fill="${fill}">`
                + `<title>${escHtml(tip)}</title></rect>`;
        }
    });
    const tx = Math.max(0, Math.min(h.counts.length, (t - h.floor) / h.step)) * (bw + gap) - gap / 2;
    svg += `<line x1="${tx}" x2="${tx}" y1="0" y2="${H}" stroke="#f8fafc" stroke-dasharray="3,2" stroke-width="1"/>`;
    if (cfg.labels) {
        svg += `<text x="0" y="${H + 12}" font-size="9" fill="#64748b">${h.floor.toFixed(1)}</text>`;
        svg += `<text x="${h.counts.length * (bw + gap) - 14}" y="${H + 12}" font-size="9" fill="#64748b">1.0</text>`;
        svg += `<text x="${h.counts.length * (bw + gap) + 4}" y="${H + 12}" font-size="9" fill="#10b981">=</text>`;
    }
    return svg + '</svg>';
}

// What each centre of a picture is compared against, under the active filter.
function kinshipAgainst(kinship) {
    return kinship === 'related'
        ? 'every centre whose characteristics extend or shorten its own'
        : kinship === 'unrelated'
            ? 'every centre built on characteristics that neither contain nor are contained in its own'
            : 'every centre of every scored schema';
}

// One of the two pictures above the cards, as a block: the drawing, a bold
// caption naming what one observation is, and the counts either side of the
// threshold. The long reading of it lives in the block's hover.
function histogramBlock(h, t, spec) {
    const { atOrAbove } = histogramBars(h, t);
    const middle = h.n_centers - atOrAbove - h.below_floor;
    const svg = histogramSvg(h, t, {
        bw: 13, gap: 3, h: 48, labels: true, noun: spec.noun, aria: spec.aria, part: spec.part,
    });
    const hidden = spec.part ? h.n_centers - spec.part.n_centers : 0;
    const rows = [
        [`At or above ${pct(t, 0)}`, `<strong>${atOrAbove}</strong>`,
        `${escHtml(spec.noun)} at or above the threshold — the ones the lists call the same`],
        h.identical
            ? ['— of them, identical rows', `<strong>${h.identical}</strong>`,
                'Mutual containment exactly 1: the two hold the same rows, not almost the same']
            : null,
        t > h.floor + 1e-9
            ? [`${pct(h.floor, 0)} to ${pct(t, 0)}`, `<strong>${middle}</strong>`,
                `Below the threshold but still drawn: what raising or lowering it would move`]
            : null,
        [`Below ${pct(h.floor, 0)}`, `<strong>${h.below_floor}</strong>`,
        `Not drawn at all: the pair store keeps nothing below ${pct(h.floor, 0)}, where two centres share less than half of the larger one`],
        hidden > 0 ? ['In the cards shown', `<strong>${spec.part.n_centers}</strong>`,
            'The solid part of each bar: the sum of the strips the reader can see'] : null,
        hidden > 0 ? ['In the centres the filter hides', `<strong>${hidden}</strong>`,
            'The faded part above it'] : null,
        ['Total', `<span class="cg-feat">${h.n_centers} ${escHtml(spec.noun)}</span>`, null],
    ];
    return `<div class="cg-hist-block" title="${escHtml(spec.title)}">${svg}
        <div class="cg-hist-cap"><strong>${escHtml(spec.caption)}</strong></div>
        ${statsHtml(rows).replace('cg-stats', 'cg-stats cg-hist-tab')}</div>`;
}

// The pictures above the cards. Both read the same axis (mutual containment,
// floor to 1, identity on the green bar, the threshold dashed), and they
// differ in what ONE observation is: the left one takes each centre's nearest
// other centre — one value per centre, so it sits at the right edge by
// construction; the right one takes every pair, which is exactly the sum of
// the strips inside the cards below. The branch scope draws both; the
// All-centres scope has no pair-wise counterpart it could afford.
function renderCenterGroupsHistogram(data) {
    const el = document.getElementById('cgHistogram');
    if (!el) return;
    const h = data.histogram;
    const t = data.threshold;
    if (!h || !h.n_centers) {
        el.innerHTML = h
            ? `<div class="cg-hist-note">No centre of this branch reaches ${data.min_rows} rows, so there is nothing to compare here.</div>`
            : '';
        return;
    }
    const against = kinshipAgainst(data.kinship);
    const branch = h.scope === 'branch';
    const who = branch
        ? `the ${h.n_centers} compared centre${h.n_centers === 1 ? '' : 's'} of this branch`
        : `all ${h.n_centers} centres of all scored schemas`;
    let out = histogramBlock(h, t, {
        caption: 'Per centre: its nearest other centre',
        noun: 'centres',
        aria: 'Closest other centre, per centre',
        title: `One bar counts CENTRES: for each of ${who}, the mutual containment with its single closest other centre, `
            + `taken over ${against}. Every centre contributes its best match only, so this picture is pressed against `
            + `the right edge by construction and a bar left of the dashed line is a centre with no alternative at the `
            + `current threshold — including centres the Descriptions filter hides from the list below. Pairs below `
            + `${pct(h.floor, 0)} are counted in the text, not drawn.`,
    });
    const hp = data.histogram_pairs;
    const hl = data.histogram_pairs_listed;
    if (hp && hp.n_centers) {
        const hidden = hl ? hp.n_centers - hl.n_centers : 0;
        out += histogramBlock(hp, t, {
            caption: hidden > 0 ? 'All pairs: solid = the sum of the cards' : 'All pairs: this is the sum of the cards',
            noun: 'pairs',
            aria: 'Similarity of every pair',
            part: hidden > 0 ? hl : null,
            title: `One bar counts PAIRS: every (centre of this branch, other centre) pair, over ${against}. `
                + (hidden > 0
                    ? `The solid part of each bar is the elementwise sum of the strips inside the cards on screen; the `
                    + `faded part above it belongs to the ${h.n_centers - (data.n_centers_listed || 0)} centres the `
                    + `Descriptions filter drops from the list, which have no alternative of that kind at the current `
                    + `threshold but do have nearer ones below it. Solid + faded is every pair of the branch.`
                    : `This is exactly the elementwise sum of the strips inside the cards below, so the bars at or above `
                    + `the dashed line are the alternatives those cards list, added up.`)
                + ` Unlike the picture on the left it keeps the whole left tail, which is where a gap — if the data had `
                + `one — would show. Pairs below ${pct(hp.floor, 0)} are counted in the text, not drawn.`,
        });
    }
    el.innerHTML = out;
}

// The strip inside one card: this centre against every other centre of the
// run. Everything at or above the dashed line is listed when the card is
// opened, so the strip shows what moving the threshold would add or drop.
function cardHistogramHtml(h, t) {
    if (!h || !h.n_centers) return '';
    const { atOrAbove } = histogramBars(h, t);
    const svg = histogramSvg(h, t, { bw: 7, gap: 1, h: 20, labels: false, noun: 'other centres', aria: 'Similarity to every other centre' });
    const middle = h.n_centers - atOrAbove - h.below_floor;
    const title = `This centre against each of the ${h.n_centers} other centres it is compared with, by mutual containment `
        + `(${pct(h.floor, 0)} to 100%, exactly the same rows on the green bar; pairs below ${pct(h.floor, 0)} are not drawn). `
        + `Dashed line: the current threshold — everything at or above it is listed when the card is opened.`;
    const parts = [`<strong>${atOrAbove}</strong> ≥ ${pct(t, 0)}`];
    if (t > h.floor + 1e-9) parts.push(`${middle} in ${pct(h.floor, 0)}–${pct(t, 0)}`);
    parts.push(`${h.below_floor} below ${pct(h.floor, 0)}`);
    return `<div class="cg-hist-mini" title="${escHtml(title)}">${svg}`
        + `<span class="cg-hist-mini-note">${parts.join(' · ')}</span></div>`;
}

function pagerInto(el, info, go) {
    if (!el) return;
    el.innerHTML = '';
    if (!info.total) return;
    const shown = info.offset + info.count;
    el.textContent = `${info.offset + 1}–${shown} of ${info.total}${info.noun ? ' ' + info.noun : ''}`;
    if (info.offset > 0) {
        const b = document.createElement('button'); b.className = 'landscape-open'; b.textContent = '← previous';
        b.onclick = (ev) => { ev.stopPropagation(); go(Math.max(0, info.offset - info.limit)); }; el.appendChild(b);
    }
    if (shown < info.total) {
        const b = document.createElement('button'); b.className = 'landscape-open'; b.textContent = 'next →';
        b.onclick = (ev) => { ev.stopPropagation(); go(info.offset + info.limit); }; el.appendChild(b);
    }
}

// ---- Selected-branch scope ------------------------------------------------
function renderBranchRedundancy() {
    const data = centerGroupsState.data;
    const list = document.getElementById('cgList');
    const summary = document.getElementById('cgSummary');
    const more = document.getElementById('cgMore');
    if (!data || !list) return;
    const value = centerGroupsValueLabel();
    const compared = data.anchors.filter(a => a.compared);
    const withAlt = compared.filter(a => a.total > 0);
    const simpler = compared.filter(a => a.simplest && a.simplest.d < data.d);
    // Under a Descriptions filter the list is an answer to "which centres have
    // a description of this kind", so centres that have none are dropped rather
    // than shown empty. The branch's own totals above stay whole-branch.
    const filtered = data.kinship && data.kinship !== 'all';
    const shown = filtered ? withAlt : data.anchors;
    if (summary) {
        const tot = compared.reduce((s, a) => {
            const k = a.kinship_counts;
            return k ? { r: s.r + k.related, u: s.u + k.unrelated, w: s.w + k.inconsistent } : s;
        }, { r: 0, u: 0, w: 0 });
        const small = data.anchors.length - compared.length;
        const rows = [
            ['Branch', `<span class="cg-d">${data.d}D</span> <strong>${escHtml(data.feature_names.join(' + '))}</strong>`],
            ['Centres compared', `<strong>${compared.length}</strong> of ${data.anchors.length}`
                + (small ? ` <span class="cg-feat">· ${small} under Min rows</span>` : ''),
                'Centres with fewer rows than Min rows are listed but not compared'],
            [`Also described elsewhere`, `<strong>${withAlt.length}</strong>`,
                `Compared centres with at least one other centre holding ≥ ${pct(data.threshold, 0)} of the same rows, both ways`],
            ['Described with fewer characteristics', `<strong>${simpler.length}</strong>`,
                'Compared centres whose rows are also held by a centre of a lower-dimensional schema'],
            (tot.r + tot.u) > 0
                ? ['Alternatives found', `<strong>${tot.r + tot.u}</strong> <span class="cg-feat">·</span> `
                    + `<span class="cg-kin">${tot.r} related</span> <span class="cg-feat">·</span> `
                    + `<span class="cg-kin">${tot.u} unrelated</span>`
                    + (tot.w ? ` <span class="cg-feat">·</span> <span class="cg-kin-warn">${tot.w} ⚠</span>` : ''),
                    'Every listed pair, over all compared centres. Related: one set of characteristics contains the other. '
                    + '⚠: they nest by characteristics but not by rows. This count ignores the Descriptions filter.']
                : null,
            filtered
                ? ['Showing', `<strong>${shown.length}</strong> centre${shown.length === 1 ? '' : 's'} `
                    + `<span class="cg-feat">· ${compared.length - shown.length} with no such description`
                    + `${small ? ` · ${small} under Min rows` : ''}</span>`,
                    data.kinship === 'related'
                        ? 'Only centres that have a description extending or shortening these characteristics'
                        : 'Only centres that have a description built on other characteristics']
                : null,
        ];
        summary.innerHTML = statsHtml(rows).replace('cg-stats', 'cg-stats cg-sum');
    }
    renderCenterGroupsHistogram(data);
    const note = document.getElementById('cgNote');
    if (note) {
        const sc = currentPayload && currentPayload.search_centers, cc = currentPayload && currentPayload.centers;
        const differs = sc && cc && (sc.n_centers !== cc.n_centers);
        note.innerHTML = differs
            ? `<span class="cg-warn">The search scored this schema on a capacity-coarsened partition (${sc.n_centers} centres); the lattice shows the full-resolution one (${cc.n_centers}). This tab compares the search partition's centres — the ones the landscape counts.</span>`
            : '';
    }
    // A card the filter has just hidden cannot stay expanded.
    if (centerGroupsState.openAnchor !== null && !shown.some(a => a.cell === centerGroupsState.openAnchor)) {
        centerGroupsState.openAnchor = null;
        centerGroupsState.detail = null;
        setCenterGroupHighlight(null);
    }
    list.innerHTML = '';
    if (!data.anchors.length) {
        list.innerHTML = '<div class="cg-busy">This schema has no certified centre at this purity floor.</div>';
    } else if (!shown.length) {
        list.innerHTML = `<div class="cg-busy">No centre of this branch has a description built on
            ${data.kinship === 'related' ? 'an extension or a shortening of these characteristics' : 'other characteristics'}
            at ≥ ${pct(data.threshold, 0)}. Switch Descriptions back to “all” to see the ${data.anchors.length} centres.</div>`;
    }
    shown.forEach(a => {
        const card = document.createElement('button');
        card.type = 'button';
        const open = centerGroupsState.openAnchor === a.cell;
        card.className = 'cg-card' + (open ? ' open' : '') + (a.compared ? '' : ' cg-card-muted');
        const bound = `<span class="cg-feat" title="One-sided Clopper–Pearson lower bound at the schema's Bonferroni level α/C = ${a.alpha_eff.toExponential(2)}">at least ${pct(a.purity_lower)}</span>`;
        const stats = [
            ['Rows', `<strong>${a.n}</strong>`],
            [value, `<strong>${pct(a.purity)}</strong> ${bound}`, 'Share of the value among this centre\u2019s rows, and its certified lower bound'],
            [`Of all ${value}`, `<strong>${pct(a.share_of_value)}</strong>`,
                'How much of the value this one centre holds. These shares add up to the branch\u2019s coverage.'],
        ];
        const hist = a.compared ? cardHistogramHtml(a.histogram, data.threshold) : '';
        let body;
        if (!a.compared) {
            body = statsHtml(stats) + `<div class="cg-single">Fewer than ${data.min_rows} rows — not compared (lower “Min rows” to include it).</div>`;
        } else if (a.total === 0) {
            // Only reachable with Descriptions = all: a filter drops these cards.
            body = statsHtml(stats.concat([
                ['Other descriptions', '<span class="cg-single">none</span>',
                    `No other centre holds at least ${pct(data.threshold, 0)} of the same rows, both ways`],
            ])) + hist;
        } else {
            const s = a.simplest;
            const simplerLine = s && s.d < data.d
                ? `<div class="cg-simpler"><div>${s.identical || s.similarity >= 1 ? 'The same rows' : `${pct(s.similarity, 0)} the same rows`} with ${s.d} characteristic${s.d === 1 ? '' : 's'} <span class="cg-feat">(${s.n} rows, ${value} ${pct(s.purity)})</span>:</div>${conditionsHtml(s.conditions)}</div>`
                : '';
            body = statsHtml(stats.concat([
                ['Other descriptions', `<span class="cg-dup"><strong>${a.total}</strong>${a.n_identical ? ` · ${a.n_identical} with exactly the same rows` : ''}</span>`,
                    `Centres of other schemas holding at least ${pct(data.threshold, 0)} of the same rows, both ways. Click the card to see them.`],
                kinshipSplitEntry(a.kinship_counts, data.kinship),
            ])) + hist + simplerLine;
        }
        card.innerHTML = `<div class="cg-desc">${centreHtml(data.d, a.conditions)}</div>${body}`;
        if (a.compared && a.total > 0) card.onclick = () => toggleBranchAnchor(a);
        list.appendChild(card);
        if (open) {
            const det = document.createElement('div');
            det.className = 'cg-detail';
            det.id = 'cgDetail';
            list.appendChild(det);
            renderOverlapTable(det, {
                title: `${a.total} other centre${a.total === 1 ? '' : 's'} with ≥ ${pct(data.threshold, 0)} of the same rows`,
                subtitle: 'compared with this branch centre, both ways',
                refLabel: 'this branch',
                ref: Object.assign({ d: data.d, schema_features: data.features }, a),
                union: a.union,
                members: a.alternatives,
                page: { total: a.total, offset: a.offset, limit: a.limit, count: a.alternatives.length },
                go: (off) => loadBranchAnchorPage(a.cell, off),
                close: () => toggleBranchAnchor(a),
            });
        }
    });
    if (more) more.innerHTML = '';
}

function toggleBranchAnchor(a) {
    if (centerGroupsState.openAnchor === a.cell) {
        centerGroupsState.openAnchor = null;
        centerGroupsState.detail = null;
        setCenterGroupHighlight(null);
    } else {
        centerGroupsState.openAnchor = a.cell;
        centerGroupsState.detail = { anchor: a };
        setCenterGroupHighlight(a.landscape_cells);
    }
    renderBranchRedundancy();
}

async function loadBranchAnchorPage(cell, offset) {
    const req = centerGroupsRequest();
    if (!req || req.scope !== 'branch') return;
    const body = Object.assign({}, req.body, { anchor: cell, limit: CG_MEMBER_PAGE, offset });
    try {
        const res = await fetch('/api/centers/branch', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok || centerGroupsState.openAnchor !== cell || !centerGroupsState.data) return;
        const i = centerGroupsState.data.anchors.findIndex(x => x.cell === cell);
        if (i >= 0) centerGroupsState.data.anchors[i] = data.anchors[0];
        renderBranchRedundancy();
    } catch (err) {
        const el = document.getElementById('cgDetail');
        if (el) el.innerHTML = `<div class="cg-warn">Request failed: ${escHtml(err.message)}</div>`;
    }
}

// ---- All-centres scope ----------------------------------------------------
function renderCenterGroups() {
    const data = centerGroupsState.data;
    const list = document.getElementById('cgList');
    const more = document.getElementById('cgMore');
    const summary = document.getElementById('cgSummary');
    const note = document.getElementById('cgNote');
    if (note) note.innerHTML = '';
    if (!data || !list) return;
    if (summary) {
        summary.innerHTML = statsHtml([
            ['Centres compared', `<strong>${data.n_centers}</strong> <span class="cg-feat">· in all scored schemas, ≥ ${data.min_rows} rows</span>`,
                'Every centre of every schema the search scored, minus those below Min rows'],
            ['Distinct row sets', `<strong>${data.n_distinct}</strong>`,
                'Centres holding exactly the same rows are one row set; the comparison runs on these'],
            [`Groups at ${pct(data.threshold, 0)}`, `<strong>${data.n_groups}</strong>`,
                'Leader clustering: every member holds at least the threshold of its representative’s rows, and the other way round'],
            ['Described more than once', `<strong>${data.n_groups_with_duplicates}</strong> groups `
                + `<span class="cg-feat">· ${data.n_centers_in_duplicate_groups} centres</span>`,
                'Groups whose rows carry more than one description'],
        ]).replace('cg-stats', 'cg-stats cg-sum');
    }
    renderCenterGroupsHistogram(data);
    const value = centerGroupsValueLabel();
    list.innerHTML = '';
    if (!data.groups.length) list.innerHTML = '<div class="cg-busy">No group matches this filter.</div>';
    data.groups.forEach(g => {
        const r = g.representative;
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'cg-card' + (centerGroupsState.openGroup === g.group ? ' open' : '');
        const others = g.n_members - 1;
        const bound = `<span class="cg-feat" title="One-sided Clopper–Pearson lower bound at the schema's Bonferroni level α/C = ${r.alpha_eff.toExponential(2)}">at least ${pct(r.purity_lower)}</span>`;
        const stats = [
            ['Rows', `<strong>${r.n}</strong>`],
            [value, `<strong>${pct(r.purity)}</strong> ${bound}`, 'Share of the value among this centre\u2019s rows, and its certified lower bound'],
            [`Of all ${value}`, `<strong>${pct(r.share_of_value)}</strong>`,
                'How much of the value this one centre holds'],
            others > 0
                ? ['Other centres', `<span class="cg-dup"><strong>${others}</strong></span>`,
                    `Centres holding at least ${pct(data.threshold, 0)} of the representative\u2019s rows, both ways. Click the card to see them.`]
                : ['Other centres', '<span class="cg-single">none</span>',
                    'No other centre describes this group at the current threshold'],
            // The grouping itself is never filtered — paging comes from the
            // server — so the split is shown and the active side highlighted,
            // and the filter takes effect in the expanded member list.
            kinshipSplitEntry(g.kinship_counts, centerGroupsControls().kinship),
        ];
        card.innerHTML = `<div class="cg-desc">${centreHtml(r.d, r.conditions)}</div>` + statsHtml(stats);
        if (others > 0) card.onclick = () => toggleCenterGroup(g);
        list.appendChild(card);
        if (centerGroupsState.openGroup === g.group) {
            const det = document.createElement('div');
            det.className = 'cg-detail';
            det.id = 'cgDetail';
            list.appendChild(det);
            const d = centerGroupsState.detail;
            if (!d) { det.innerHTML = '<div class="cg-busy">Loading the centres of this group…</div>'; return; }
            renderOverlapTable(det, {
                title: `${d.total} other centre${d.total === 1 ? '' : 's'} in this group`,
                subtitle: `at least ${pct(d.threshold, 0)} of rows shared with the representative, both ways`,
                refLabel: 'representative',
                ref: d.representative,
                union: d.union,
                minSimilarity: d.min_similarity,
                kinshipCounts: d.kinship_counts,
                kinship: d.kinship,
                members: d.members,
                page: { total: d.total, offset: d.offset, limit: d.limit, count: d.members.length },
                go: (off) => loadCenterGroupDetail(off),
                close: () => toggleCenterGroup({ group: d.group }),
            });
        }
    });
    pagerInto(more, { total: data.total, offset: data.offset, limit: data.limit, count: data.groups.length, noun: 'groups' },
        (off) => refreshCenterGroups(off));
}

async function toggleCenterGroup(g) {
    if (centerGroupsState.openGroup === g.group) {
        centerGroupsState.openGroup = null;
        centerGroupsState.detail = null;
        setCenterGroupHighlight(null);
        renderCenterGroups();
        return;
    }
    centerGroupsState.openGroup = g.group;
    centerGroupsState.detail = null;
    setCenterGroupHighlight(g.landscape_cells);
    renderCenterGroups();
    await loadCenterGroupDetail(0);
}

async function loadCenterGroupDetail(offset) {
    const req = centerGroupsRequest();
    const gid = centerGroupsState.openGroup;
    if (!req || req.scope !== 'all' || gid === null) return;
    const body = Object.assign({}, req.body, { group: gid, limit: CG_MEMBER_PAGE, offset: offset || 0 });
    delete body.filter_d; delete body.sort;
    try {
        const res = await fetch('/api/centers/group', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const data = await res.json();
        if (centerGroupsState.openGroup !== gid) return;
        if (!res.ok) {
            const el = document.getElementById('cgDetail');
            if (el) el.innerHTML = `<div class="cg-warn">${escHtml(data.error || res.status)}</div>`;
            return;
        }
        centerGroupsState.detail = data;
        renderCenterGroups();
    } catch (err) {
        const el = document.getElementById('cgDetail');
        if (el) el.innerHTML = `<div class="cg-warn">Request failed: ${escHtml(err.message)}</div>`;
    }
}

// ---- The overlap table (both scopes) ----------------------------------------
// The rows one side holds and the other does not, with the share of the value
// among them - the column that decides between two similar descriptions. A
// share near the base rate means the extra rows are noise; a share near the
// centre's own purity means they are signal the other side is missing.
function differenceHtml(m, value, refShort) {
    const line = (sign, side, where) => {
        if (!side || side.n === 0) return '';
        const share = side.purity === null || side.purity === undefined
            ? '' : ` <span class="cg-feat">${value} ${pct(side.purity, 0)}</span>`;
        return `<div class="cg-diffline"><span class="cg-sign">${sign}${side.n}</span> ${where}${share}</div>`;
    };
    const here = line('+', m.only_here, 'here');
    const there = line('−', m.only_in_representative, `in ${refShort}`);
    return (here + there) || '<span class="cg-feat">same rows</span>';
}

// Similarity, with the chance level spelled out only when it is high enough
// to change the reading: two big centres overlap a great deal by accident,
// two small ones do not.
function similarityHtml(m) {
    const loud = m.chance_similarity >= 0.2;
    const tip = `Mutual containment |A ∩ B| / max(|A|, |B|), compared with the threshold. `
        + `Two unrelated centres of these sizes would reach ${pct(m.chance_similarity, 0)} by chance; `
        + `rescaled against that, this is ${pct(m.similarity_above_chance, 0)}.`;
    return `<span title="${escHtml(tip)}"><strong>${pct(m.similarity, 0)}</strong>`
        + (loud ? ` <span class="cg-warn">(${pct(m.similarity_above_chance, 0)} above chance)</span>` : '')
        + '</span>';
}

// How an alternative stands to the reference, as the two independent facts the
// server reports: `lineage` compares the two sets of CHARACTERISTICS, `relation`
// compares the two sets of ROWS. They normally agree — a description that adds
// characteristics selects a subset of the rows — and when they do not, the pair
// is an artefact of level coarsening (a "child" cell straddling two parent
// cells), which is marked rather than presented as a sub-cell.
const CG_RELATION_SIGN = { identical: '=', inside: '⊂', contains: '⊃', crossing: '✕' };
const CG_LINEAGE_WORD = { child: 'extends', parent: 'shortens', same_schema: 'same schema', unrelated: 'other' };

function kinshipHtml(m, refShort) {
    const k = m.kinship;
    if (!k || !k.lineage) return '';
    const rows = {
        identical: 'exactly the same rows',
        inside: `its rows are inside the ${refShort}'s`,
        contains: `its rows contain the ${refShort}'s`,
        crossing: 'each side holds rows the other does not',
    }[k.relation];
    const chars = {
        child: `adds characteristics to the ${refShort}'s`,
        parent: `uses fewer characteristics than the ${refShort}`,
        same_schema: 'uses the same characteristics',
        unrelated: `neither set of characteristics contains the other`,
    }[k.lineage];
    const warn = !k.consistent;
    const tip = `Characteristics: ${chars}. Rows: ${rows}.`
        + (warn ? ' These two disagree: a description that only adds characteristics cannot leave its parent\'s rows.'
            + ' This pair comes from two different level partitions (capacity coarsening), so it is not a sub-cell of the reference.' : '');
    return `<span class="cg-kin${warn ? ' cg-kin-warn' : ''}" title="${escHtml(tip)}">`
        + `${escHtml(CG_LINEAGE_WORD[k.lineage] || k.lineage)} <span class="cg-sign">${CG_RELATION_SIGN[k.relation] || '?'}</span>`
        + (warn ? ' ⚠' : '') + '</span>';
}

// The related / unrelated split of a card's alternatives, ALWAYS over the
// unfiltered set, so switching the filter never changes this line — it says
// what is being hidden. Returns a `statsHtml` entry, or null when there is
// nothing to split.
function kinshipSplitEntry(counts, active) {
    if (!counts || (counts.related + counts.unrelated) === 0) return null;
    const on = (kind) => (active === kind || active === 'all' || !active) ? '' : ' cg-off';
    const parts = [
        `<span class="cg-kin${on('related')}"><strong>${counts.related}</strong> related</span>`,
        `<span class="cg-kin${on('unrelated')}"><strong>${counts.unrelated}</strong> unrelated</span>`,
    ];
    if (counts.inconsistent) {
        parts.push(`<span class="cg-kin-warn" title="Related pairs whose rows do not nest the way their characteristics do. `
            + `The two schemas discretised a shared column differently (capacity coarsening), so neither cell is a sub-cell of the other.">`
            + `${counts.inconsistent} ⚠</span>`);
    }
    return ['Of those', parts.join(' · '),
        'Related: one set of characteristics contains the other. Unrelated: neither does — a description built on other characteristics. '
        + 'This split is over all alternatives, whatever the Descriptions filter shows.'];
}

function renderOverlapTable(el, spec) {
    const value = centerGroupsValueLabel();
    const r = spec.ref;
    const refShort = spec.refLabel === 'representative' ? 'rep.' : 'branch';
    const rows = [];
    rows.push(`<tr class="cg-rep">
        <td class="cg-desc-cell">${centreHtml(r.d, r.conditions, `<span class="cg-tag cg-tag-ref">${escHtml(spec.refLabel)}</span>`)}</td>
        <td>${r.n}</td><td>${pct(r.purity)} <span class="cg-feat">≥ ${pct(r.purity_lower)}</span></td>
        <td>—</td><td>—</td><td>—</td>
        <td><button type="button" class="landscape-open" data-feats="${r.schema_features.join(',')}">open</button></td></tr>`);
    spec.members.forEach(m => {
        rows.push(`<tr>
            <td class="cg-desc-cell">${centreHtml(m.d, m.conditions, m.identical ? '<span class="cg-tag">same rows</span>' : '', r.conditions)}</td>
            <td>${m.n}</td>
            <td>${pct(m.purity)} <span class="cg-feat">≥ ${pct(m.purity_lower)}</span></td>
            <td>${similarityHtml(m)}</td>
            <td class="cg-kin-cell">${kinshipHtml(m, refShort)}</td>
            <td class="cg-diff-cell">${differenceHtml(m, value, refShort)}</td>
            <td><button type="button" class="landscape-open" data-feats="${m.schema_features.join(',')}">open</button></td></tr>`);
    });
    const together = spec.union
        ? ` · all of them together hold <strong>${spec.union.n}</strong> rows at ${value} ${pct(spec.union.purity)}`
        : '';
    const weakest = (spec.minSimilarity !== undefined && spec.minSimilarity !== null)
        ? ` · weakest similarity in the group ${pct(spec.minSimilarity, 0)}` : '';
    const kc = spec.kinshipCounts;
    const split = kc && (kc.related + kc.unrelated) > 0
        ? ` · <span class="cg-kin">${kc.related} related</span>, <span class="cg-kin">${kc.unrelated} unrelated</span>`
        + (kc.inconsistent ? `, <span class="cg-kin-warn">${kc.inconsistent} ⚠</span>` : '')
        + (spec.kinship && spec.kinship !== 'all' ? ` (showing ${escHtml(spec.kinship)} only)` : '')
        : '';
    el.innerHTML = `
        <div class="cg-detail-head"><span>${escHtml(spec.title)}${together}${weakest}${split}
            <span class="cg-sub">· ${escHtml(spec.subtitle)} · <span class="cg-diff-key">highlighted</span> = not in the ${spec.refLabel === 'representative' ? 'representative' : 'branch centre'}</span></span>
            <button type="button" class="landscape-close" title="Close">✕</button></div>
        <div class="cg-table-wrap"><table class="cg-table">
            <thead><tr>
                <th>Centre</th><th>Rows</th><th>${escHtml(value)}</th>
                <th title="Mutual containment: each centre holds at least this share of the other's rows. The threshold is compared with this number. The chance level is in the cell's hover, and is spelled out in the cell when it is high enough to matter.">Similarity</th>
                <th title="Two facts. The word compares the CHARACTERISTICS with the ${escHtml(spec.refLabel)}'s: extends (adds some), shortens (uses fewer), other (neither set contains the other). The symbol compares the ROWS: = the same, ⊂ inside, ⊃ contains, ✕ each side has rows the other has not. ⚠ marks a pair where the two disagree, which only happens when the two schemas discretised a shared column differently.">Relation</th>
                <th title="Rows one side holds and the other does not: +N rows only in this centre, −N rows only in the ${escHtml(spec.refLabel)}, with the share of the value among them. A share near the base rate means those rows are noise; a share near the centre's own purity means the other side is missing signal.">Rows not shared</th>
                <th></th></tr></thead>
            <tbody>${rows.join('')}</tbody></table></div>
        <div class="landscape-cell-more" id="cgDetailMore"></div>`;
    el.querySelector('.landscape-close').onclick = (ev) => { ev.stopPropagation(); spec.close(); };
    el.querySelectorAll('button[data-feats]').forEach(b => {
        b.onclick = (ev) => { ev.stopPropagation(); openSchemaFromLandscape(b.dataset.feats.split(',').map(Number)); };
    });
    pagerInto(document.getElementById('cgDetailMore'), spec.page, spec.go);
}

// Rings on the landscape around the lattice cells holding the schemas of the
// expanded card (drawn by renderLandscape for the lattice shown there).
function setCenterGroupHighlight(cells) {
    landscapeState.highlight = cells && cells.length ? cells : null;
    if (viewMode === 'landscape' && landscapeState.bins) renderLandscape();
}

function landscapeHighlightPoints(bins) {
    const cells = landscapeState.highlight;
    if (!cells) return null;
    const all = bins.d === null || bins.d === undefined;
    const seen = new Map();
    cells.forEach(c => {
        if (!all && Number(c.d) !== Number(bins.d)) return;
        const iy = all ? c.iy_all : c.iy;
        const k = `${c.ix},${iy}`;
        seen.set(k, (seen.get(k) || 0) + c.schemas);
    });
    const xs = [], ys = [], hov = [];
    seen.forEach((n, k) => {
        const [ix, iy] = k.split(',').map(Number);
        xs.push(ix); ys.push(iy); hov.push(`${n} schema${n === 1 ? '' : 's'} of the card opened in Redundancy`);
    });
    return { xs, ys, hov };
}


// ---------------------------------------------------------------------------
// Tau-curves: the landscape's envelope over the purity floor (Section 4.9)
// ---------------------------------------------------------------------------
// One line per dimensionality d: at every whole-percent purity floor from
// the base rate of the searched indicator to 100 %, the best coverage
// (absence: mass certified free) any d-axis schema reaches, with that
// schema and its centre count in the hover. Computed once per target,
// criterion, direction, rule, alpha and min_samples - NOT per tau, the
// curve is the dependence on tau; the current certified boundary is drawn
// as a vertical cursor. Clicking a point lists the schemas of that d whose
// x-fraction at that floor lies in the point's ten-percent category (the
// landscape at that floor, `Landscape.by_x_category`); "open" moves the
// boundary to that floor and renders the schema.
let curvesState = {
    key: null,          // JSON of the parameters the curves belong to (no tau)
    data: null,         // /api/landscape/curves response
    point: null,        // {d, ti, ix} of the clicked point
    pointData: null,    // /api/landscape/at response
};
const TAU_CURVE_COLORS = { 1: '#94a3b8', 2: '#38bdf8', 3: '#a78bfa', 4: '#f472b6' };

function curvesKey() {
    const p = landscapeParams();
    if (!p) return null;
    const q = Object.assign({}, p);
    delete q.tau;
    return JSON.stringify(q);
}

async function refreshTauCurves() {
    const p = landscapeParams();
    if (!p) return;
    const key = curvesKey();
    if (curvesState.data && curvesState.key === key) {
        renderTauCurves();
        if (curvesState.pointData) renderTauPoint();
        return;
    }
    const note = document.getElementById('toCurveNote');
    if (note) note.textContent = 'Computing the curves…';
    try {
        const res = await fetch('/api/landscape/curves', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(p),
        });
        const data = await res.json();
        if (!res.ok) { showAnalysisError('Tau-curves: ' + (data.error || res.status)); return; }
        curvesState = { key, data, point: null, pointData: null };
        closeTauPoint();
        renderTauCurves();
    } catch (err) {
        showAnalysisError('Tau-curves request failed: ' + err.message);
    }
}

function tauCurveXLabel(data) {
    return data.direction === 'absence' ? 'purity floor: share of rows without the value, %' : 'purity floor: share of the value, %';
}

function tauCurveYLabel(data) {
    return data.x === 'mass' ? 'mass certified free of the value, %' : 'coverage of the value, %';
}

// ---- Trade-offs tab -------------------------------------------------------
// Three pictures of what coverage costs. The first two share an x axis (the
// purity floor) and are stacked, so the two quantities are read against one
// scale instead of two: the envelope's coverage above, the centre count of the
// schema that reaches it below. That schema CHANGES along the envelope, and a
// cost series of a moving schema would be a saw read as a trend, so every
// change of winner is marked on both panels; "Track: the selected branch"
// switches both to one fixed schema instead. The third answers the question
// the first two cannot: at the CURRENT floor, how much coverage is available
// at a given number of rules - the Pareto staircase, where the search's own
// winner is usually far to the right of the knee.
const TO_PLOT_CFG = { displayModeBar: false, responsive: true };

function toLayout(data, opts) {
    return {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(15,23,42,0.4)',
        margin: { l: 56, r: 12, t: 10, b: opts.xtitle ? 44 : 26 }, autosize: true,
        font: { color: '#94a3b8', size: 11 },
        xaxis: {
            title: opts.xtitle ? { text: opts.xtitle, font: { size: 11 } } : undefined,
            range: opts.xrange, gridcolor: 'rgba(255,255,255,0.06)', zeroline: false,
            type: opts.xtype || 'linear',
        },
        yaxis: {
            title: { text: opts.ytitle, font: { size: 11 } }, range: opts.yrange,
            gridcolor: 'rgba(255,255,255,0.06)', zeroline: false, rangemode: 'tozero',
        },
        legend: { orientation: 'h', x: 0, y: 1.0, yanchor: 'bottom', font: { size: 10 }, itemwidth: 16, traceorder: 'normal' },
        shapes: opts.shapes || [], hovermode: 'closest', showlegend: !!opts.legend,
    };
}

// Where the envelope changes schema: one tick per change, so a jump in the
// centre count is never mistaken for a property of the data.
function tauWinnerChanges(curve, taus) {
    const out = [];
    for (let i = 1; i < taus.length; i++) {
        const a = curve.feature_names[i - 1], b = curve.feature_names[i];
        if (!a || !b) continue;
        if (a.join('+') !== b.join('+')) out.push(i);
    }
    return out;
}

// Which dimensionality leads at each floor: the argmax of the upper panel,
// as intervals on the same axis, each in that d's own colour. Ties go to the
// SMALLER d — at equal coverage the shorter description is the better one, the
// rule the toolbar states — and then to the schema with fewer centres. Floors
// where nothing certifies have no leader and stay blank.
function leadSegments(data, taus, dims) {
    const segs = [];
    for (let i = 0; i < taus.length; i++) {
        let best = null;
        dims.forEach(d => {
            const c = data.curves[String(d)];
            const v = c.x[i];
            if (v <= 0) return;
            if (!best || v > best.v + 1e-12
                || (Math.abs(v - best.v) <= 1e-12 && c.n_centers[i] < best.k)) {
                best = { d, v, k: c.n_centers[i], name: c.feature_names[i].join(' + ') };
            }
        });
        const last = segs.length ? segs[segs.length - 1] : null;
        if (last && last.d === (best && best.d)) {
            last.i1 = i;
            last.v1 = best ? best.v : last.v1;
            last.name1 = best ? best.name : last.name1;
        } else if (best) {
            segs.push({ d: best.d, i0: i, i1: i, v0: best.v, v1: best.v, name0: best.name, name1: best.name });
        }
    }
    return segs;
}

function renderLeadStrip(data, taus, dims, shapes, xrange) {
    const el = document.getElementById('to-lead-plot');
    if (!el) return [];
    const segs = leadSegments(data, taus, dims);
    const half = taus.length > 1 ? (taus[1] - taus[0]) * 50 : 0.5;
    const traces = segs.map(s => {
        const x0 = taus[s.i0] * 100 - half, x1 = taus[s.i1] * 100 + half;
        return {
            type: 'bar', orientation: 'h', base: [x0], x: [Math.max(x1 - x0, 1e-6)], y: ['leads'],
            marker: { color: TAU_CURVE_COLORS[s.d] || '#e2e8f0', line: { width: 0 } },
            width: 0.72, showlegend: false, hoverinfo: 'text',
            hovertext: `${s.d}D leads from ${x0.toFixed(0)}% to ${x1.toFixed(0)}%`
                + ` · ${data.x} ${pct(s.v0)} → ${pct(s.v1)}`
                + (s.name0 === s.name1 ? ` · ${s.name0}` : ` · ${s.name0} … ${s.name1}`),
        };
    });
    const layout = toLayout(data, {
        ytitle: '', xtitle: tauCurveXLabel(data), xrange, shapes,
    });
    layout.barmode = 'overlay';
    layout.bargap = 0;
    layout.margin = { l: 56, r: 12, t: 4, b: 44 };
    layout.yaxis = { showticklabels: false, showgrid: false, zeroline: false, fixedrange: true, range: [-0.5, 0.5] };
    Plotly.react(el, traces, layout, TO_PLOT_CFG);
    return segs;
}

// The cost unit of the WHOLE tab: the centres panel and the frontier must
// never measure the description in two different units at once. "conditions"
// is d x centres — every centre of a d-schema fixes exactly d columns — which
// is what puts the four dimensionalities on one scale: a 4D rule spells out
// four (column = value) pairs where a 1D rule spells out one.
function costUnit() {
    const el = document.getElementById('toCost');
    return el && el.value === 'conditions' ? 'conditions' : 'centers';
}

function onCostChange() {
    renderTauCurves();
    refreshFrontier(true);
}

function trackMode() {
    const el = document.getElementById('toTrack');
    return el && el.value === 'branch' ? 'branch' : 'envelope';
}

// The selected branch as a series over the same tau grid: its own coverage and
// centre count, read off the envelope's family only where it IS the winner —
// elsewhere the server has not scored it at that floor, so the series is drawn
// only at the floors where the branch appears. Honest and cheap; a full curve
// for one fixed schema would need its own pass.
function branchSeries(data, taus) {
    const feats = centerGroupsBranchFeatures();
    if (!feats || !feats.length) return null;
    const want = feats.slice().sort((a, b) => a - b).join(',');
    const d = String(feats.length);
    const c = data.curves[d];
    if (!c) return null;
    const xs = [], cov = [], cen = [];
    taus.forEach((t, i) => {
        const f = c.features[i];
        if (!f) return;
        if (f.slice().sort((a, b) => a - b).join(',') !== want) return;
        xs.push(t * 100); cov.push(c.x[i] * 100); cen.push(c.n_centers[i]);
    });
    return xs.length ? { xs, cov, cen, d: feats.length } : null;
}

function renderTauCurves() {
    const data = curvesState.data;
    const covEl = document.getElementById('to-coverage-plot');
    const cenEl = document.getElementById('to-centres-plot');
    const note = document.getElementById('toCurveNote');
    if (!data || !covEl || !cenEl || typeof Plotly === 'undefined') return;
    const taus = data.taus;
    const dims = Object.keys(data.curves).map(Number).sort((a, b) => a - b);
    const track = trackMode();
    const cursor = readCertTau() * 100;
    const xmin = taus.length ? Math.floor(taus[0] * 100) - 1 : 0;
    const cursorShape = {
        type: 'line', xref: 'x', yref: 'paper', x0: cursor, x1: cursor, y0: 0, y1: 1,
        line: { color: '#f8fafc', width: 1, dash: 'dot' },
    };

    const unit = costUnit();
    const costNoun = unit === 'conditions' ? 'conditions' : 'centres';
    const costFactor = (d) => (unit === 'conditions' ? d : 1);
    const covTraces = [], cenTraces = [], marks = [];
    let totalChanges = 0;
    dims.forEach(d => {
        const c = data.curves[String(d)];
        const colour = TAU_CURVE_COLORS[d] || '#e2e8f0';
        const custom = taus.map((t, i) => [
            c.n_centers[i], c.feature_names[i].join(' + '), (c.mass[i] * 100).toFixed(2),
            c.n_certifying[i], c.n_family,
        ]);
        const dim = track === 'branch' ? 0.25 : 1;
        covTraces.push({
            type: 'scatter', mode: 'lines+markers', name: `${d}D`,
            x: taus.map(t => t * 100), y: c.x.map(v => v * 100), customdata: custom,
            line: { color: colour, width: 1.5 }, marker: { size: 4, color: colour },
            opacity: dim,
            hovertemplate: `<b>${d}D</b> at %{x:.0f}%: ${data.x} %{y:.2f}%<br>`
                + '%{customdata[1]}<br>%{customdata[0]} centres, mass %{customdata[2]}%<br>'
                + '%{customdata[3]} of %{customdata[4]} schemas certify a centre<extra></extra>',
        });
        cenTraces.push({
            type: 'scatter', mode: 'lines+markers', name: `${d}D`, showlegend: false,
            x: taus.map(t => t * 100), y: c.n_centers.map(v => v * costFactor(d)), customdata: custom,
            line: { color: colour, width: 1.5, shape: 'hv' }, marker: { size: 4, color: colour },
            opacity: dim,
            hovertemplate: `<b>${d}D</b> at %{x:.0f}%: %{y} ${costNoun}`
                + (unit === 'conditions' ? ` <i>(%{customdata[0]} centres × ${d})</i>` : '')
                + '<br>%{customdata[1]}<extra></extra>',
        });
        if (track !== 'branch') {
            const ch = tauWinnerChanges(c, taus);
            totalChanges += ch.length;
            ch.forEach(i => marks.push({
                type: 'line', xref: 'x', yref: 'paper', x0: taus[i] * 100, x1: taus[i] * 100,
                y0: 0, y1: 0.06, line: { color: colour, width: 1 },
            }));
        }
    });
    const bs = track === 'branch' ? branchSeries(data, taus) : null;
    if (bs) {
        const colour = TAU_CURVE_COLORS[bs.d] || '#f8fafc';
        covTraces.push({
            type: 'scatter', mode: 'lines+markers', name: 'selected branch',
            x: bs.xs, y: bs.cov, line: { color: '#f8fafc', width: 2 }, marker: { size: 5, color: colour },
            hovertemplate: `selected branch at %{x:.0f}%: ${data.x} %{y:.2f}%<extra></extra>`,
        });
        cenTraces.push({
            type: 'scatter', mode: 'lines+markers', name: 'selected branch', showlegend: false,
            x: bs.xs, y: bs.cen.map(v => v * costFactor(bs.d)), line: { color: '#f8fafc', width: 2, shape: 'hv' },
            marker: { size: 5, color: colour },
            hovertemplate: `selected branch at %{x:.0f}%: %{y} ${costNoun}<extra></extra>`,
        });
    }
    if (curvesState.point) {
        const c = data.curves[String(curvesState.point.d)];
        const ti = curvesState.point.ti;
        if (c) {
            covTraces.push({
                type: 'scatter', mode: 'markers', x: [taus[ti] * 100], y: [c.x[ti] * 100],
                marker: { size: 11, color: 'rgba(0,0,0,0)', line: { color: '#f8fafc', width: 2 } },
                hoverinfo: 'skip', showlegend: false,
            });
        }
    }
    // The winner-change ticks go under the CENTRES panel only: on the coverage
    // panel a change of schema leaves no artefact (the envelope is continuous
    // by construction), while on the centres panel it is a step that would
    // otherwise read as a cost that moved.
    Plotly.react(covEl, covTraces, toLayout(data, {
        ytitle: tauCurveYLabel(data), xrange: [xmin, 101], yrange: [0, 102],
        shapes: [cursorShape], legend: true,
    }), TO_PLOT_CFG);
    Plotly.react(cenEl, cenTraces, toLayout(data, {
        ytitle: unit === 'conditions'
            ? 'conditions in its description (d × centres)' : 'certified centres of that schema',
        xrange: [xmin, 101], shapes: [cursorShape].concat(marks),
    }), TO_PLOT_CFG);
    const segs = renderLeadStrip(data, taus, dims, [cursorShape], [xmin, 101]);

    const cenTitle = document.getElementById('toCentresTitle');
    if (cenTitle) {
        cenTitle.textContent = (unit === 'conditions' ? 'Conditions' : 'Centres') + ' vs. purity floor';
    }
    covEl.removeAllListeners && covEl.removeAllListeners('plotly_click');
    covEl.on('plotly_click', ev => {
        const pt = ev.points && ev.points[0];
        if (!pt || pt.curveNumber >= dims.length) return;
        openTauPoint(dims[pt.curveNumber], pt.pointNumber, 0);
    });
    if (note) {
        const p0 = (data.anchor * 100).toFixed(1);
        const nfam = dims.map(d => `${d}D: ${data.curves[String(d)].n_family}`).join(', ');
        note.innerHTML = `${data.x === 'mass' ? 'Mass certified free' : 'Coverage'} of the best schema per dimensionality at
            every whole-percent purity floor above the base rate (${p0}%)${data.rule === 'certified' ? ', certified rule (100% is not certifiable)' : ''},
            and what that schema costs to write down — ${costNoun}.${unit === 'conditions'
                ? ' A d-schema spells out d (column = value) pairs per centre, so this is the unit that puts the four'
                  + ' dimensionalities on one scale: in centres the 4D envelope is 5× the 2D one on these data, in conditions 10×.'
                : ' Centres count rules, not their length: a 4D rule fixes four columns where a 1D rule fixes one, so switch'
                  + ' Cost in to conditions to compare the four lines on one scale.'} Schemas scored — ${nfam}. Dotted line: the current boundary;
            click a point on the upper panel for the schemas behind it.
            ${track === 'branch'
                ? 'Both panels follow the selected branch, drawn only at the floors where it is its dimensionality’s best schema.'
                : `<strong>The schema changes along each line</strong> — ${totalChanges} times in all, ticked under the centres
                   panel: a step there at a tick is a different schema, not a cost that moved.`}
            The band under them is the argmax of the upper panel — which dimensionality leads at each floor, in that d's colour,
            ${segs.length} stretch${segs.length === 1 ? '' : 'es'} in all${segs.length && segs.length <= 6 ? `: ${escHtml(segs.map(g => `${g.d}D`).join(' → '))}` : ''};
            at equal coverage the smaller d wins the band, since the shorter description is the better one.${segs.length > 6 ? ' Many short alternating stretches mean the lines run together there, not that the answer changes that often.' : ''}
            All of it is maxima over a family scored and then chosen by looking, so the values are optimistic and are
            not out-of-sample estimates.`;
    }
}

// ---- The cost-coverage frontier at the current floor ----------------------
let frontierState = { key: null, data: null };

function frontierKey() {
    const p = landscapeParams();
    if (!p) return null;
    const el = document.getElementById('toCost');
    return JSON.stringify([p, el ? el.value : 'centers']);
}

async function refreshFrontier(force) {
    const p = landscapeParams();
    if (!p) return;
    const key = frontierKey();
    if (!force && frontierState.data && frontierState.key === key) { renderFrontier(); return; }
    const el = document.getElementById('toCost');
    try {
        const res = await fetch('/api/landscape/frontier', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.assign({}, p, { cost: el ? el.value : 'centers' })),
        });
        const data = await res.json();
        if (!res.ok) { showAnalysisError('Frontier: ' + (data.error || res.status)); return; }
        frontierState = { key, data };
        renderFrontier();
    } catch (err) {
        showAnalysisError('Frontier request failed: ' + err.message);
    }
}

function renderFrontier() {
    const data = frontierState.data;
    const el = document.getElementById('to-frontier-plot');
    const note = document.getElementById('toFrontierNote');
    if (!data || !el || typeof Plotly === 'undefined') return;
    const dims = Object.keys(data.dims).map(Number).sort((a, b) => a - b);
    const costWord = data.cost === 'conditions' ? 'conditions' : 'centres';
    const traces = [];
    dims.forEach(d => {
        const steps = data.dims[String(d)].steps;
        if (!steps.length) return;
        const colour = TAU_CURVE_COLORS[d] || '#e2e8f0';
        traces.push({
            type: 'scatter', mode: 'lines+markers', name: `${d}D`,
            x: steps.map(s => s.cost), y: steps.map(s => s.coverage * 100),
            customdata: steps.map(s => [s.feature_names.join(' + '), s.n_centers, s.conditions]),
            line: { color: colour, width: 1.5, shape: 'hv' }, marker: { size: 6, color: colour },
            hovertemplate: `<b>${d}D</b> at %{x} ${costWord}: ${data.x} %{y:.1f}%<br>`
                + '%{customdata[0]}<br>%{customdata[1]} centres, %{customdata[2]} conditions<extra></extra>',
        });
    });
    const feats = centerGroupsBranchFeatures();
    if (feats && feats.length) {
        const want = feats.slice().sort((a, b) => a - b).join(',');
        let here = null;
        dims.forEach(d => data.dims[String(d)].steps.forEach(s => {
            if (s.features.slice().sort((a, b) => a - b).join(',') === want) here = s;
        }));
        if (here) {
            traces.push({
                type: 'scatter', mode: 'markers', name: 'selected branch',
                x: [here.cost], y: [here.coverage * 100],
                marker: { size: 13, color: 'rgba(0,0,0,0)', line: { color: '#f8fafc', width: 2 } },
                hovertemplate: `selected branch: %{x} ${costWord}, ${data.x} %{y:.1f}%<extra></extra>`,
            });
        }
    }
    Plotly.react(el, traces, toLayout(data, {
        ytitle: data.x === 'mass' ? 'mass certified free, %' : 'coverage of the value, %',
        xtitle: `cost of the description: ${costWord} (a step means "with at most this many")`,
        yrange: [0, 102], legend: true,
    }), TO_PLOT_CFG);
    if (note) {
        const best = [];
        dims.forEach(d => {
            const steps = data.dims[String(d)].steps;
            if (steps.length) {
                const top = steps[steps.length - 1];
                const knee = steps.find(s => s.coverage >= top.coverage * 0.9);
                if (knee && knee.cost < top.cost) {
                    best.push(`${d}D reaches ${pct(top.coverage)} at ${top.cost} ${costWord}, but ${pct(knee.coverage)} — `
                        + `${((knee.coverage / top.coverage) * 100).toFixed(0)}% of it — already at ${knee.cost}`);
                }
            }
        });
        note.innerHTML = `At the current purity floor, the most ${data.x === 'mass' ? 'mass' : 'coverage'} any schema of each
            dimensionality certifies with at most that many ${costWord}. A schema is on the staircase only if no schema of its
            dimensionality beats it on both axes at once. The search ranks by ${data.x} alone, so everything left of each line's
            right end is invisible in the branch list.
            ${best.length ? '<br>' + escHtml(best.join('; ')) + '.' : ''}
            <br>These are maxima over ${data.n_candidates} scored schemas, read after looking at them: the values are optimistic
            and carry no uncorrected p-value. Opening a schema still reports out-of-sample coverage.`;
    }
}

async function openTauPoint(d, ti, offset) {
    const p = landscapeParams();
    const data = curvesState.data;
    if (!p || !data) return;
    const c = data.curves[String(d)];
    if (!c || ti < 0 || ti >= data.taus.length) return;
    const tau = data.taus[ti];
    const ix = landscapeBinIndex(c.x[ti], 10);
    if (c.x[ti] <= 0) {
        curvesState.point = { d, ti, ix, tau };
        curvesState.pointData = { total: 0, offset: 0, limit: 50, schemas: [], empty_reason: 'no schema certifies a centre at this floor' };
        renderTauPoint();
        renderTauCurves();
        return;
    }
    try {
        const res = await fetch('/api/landscape/at', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.assign({}, p, { tau, d, ix, limit: 50, offset })),
        });
        const out = await res.json();
        if (!res.ok) { showAnalysisError('Tau-curve point: ' + (out.error || res.status)); return; }
        curvesState.point = { d, ti, ix, tau };
        curvesState.pointData = out;
        renderTauPoint();
        renderTauCurves();
    } catch (err) {
        showAnalysisError('Tau-curve point request failed: ' + err.message);
    }
}

function closeTauPoint() {
    curvesState.point = null;
    curvesState.pointData = null;
    const panel = document.getElementById('tauPointPanel');
    if (panel) panel.style.display = 'none';
    if (viewMode === 'tradeoffs' && curvesState.data) renderTauCurves();
}

function renderTauPoint() {
    const panel = document.getElementById('tauPointPanel');
    const list = document.getElementById('tauPointList');
    const more = document.getElementById('tauPointMore');
    const title = document.getElementById('tauPointTitle');
    const pt = curvesState.point, out = curvesState.pointData, data = curvesState.data;
    if (!panel || !list || !pt || !out || !data) return;
    panel.style.display = '';
    const xName = data.x;
    const tauPct = Math.round(pt.tau * 100);
    if (title) {
        title.textContent = out.empty_reason
            ? `${pt.d}D at ${tauPct}%: ${out.empty_reason}`
            : `${out.total} ${pt.d}D schema${out.total === 1 ? '' : 's'} with ${xName} in (${pt.ix * 10},${(pt.ix + 1) * 10}]% at a ${tauPct}% floor — highest first`;
    }
    list.innerHTML = '';
    (out.schemas || []).forEach(sc => {
        const row = document.createElement('div');
        row.className = 'landscape-row';
        const xv = xName === 'mass' ? sc.mass : sc.coverage;
        row.innerHTML = `
            <span class="axes">${sc.feature_names.join(' + ')}</span>
            <span class="num">${xName} ${(xv * 100).toFixed(2)}%</span>
            <span class="num">${sc.n_centers} centre${sc.n_centers === 1 ? '' : 's'}</span>
            <span class="num">mass ${(sc.mass * 100).toFixed(2)}%</span>
            <button type="button" class="landscape-open" title="Move the certified boundary to ${tauPct}% and render this schema">open at ${tauPct}%</button>`;
        row.querySelector('.landscape-open').onclick = () => openSchemaAtTau(sc.features, pt.tau);
        list.appendChild(row);
    });
    if (more) {
        more.innerHTML = '';
        if (out.total > 0) {
            const shown = out.offset + out.schemas.length;
            more.textContent = `${out.offset + 1}–${shown} of ${out.total}`;
            if (out.offset > 0) {
                const b = document.createElement('button'); b.className = 'landscape-open'; b.textContent = '← previous';
                b.onclick = () => openTauPoint(pt.d, pt.ti, Math.max(0, out.offset - out.limit)); more.appendChild(b);
            }
            if (shown < out.total) {
                const b = document.createElement('button'); b.className = 'landscape-open'; b.textContent = 'next →';
                b.onclick = () => openTauPoint(pt.d, pt.ti, out.offset + out.limit); more.appendChild(b);
            }
        }
    }
}

// Opens a schema at a purity floor chosen on the curve: the certified
// boundary is moved to that floor (the same input applyCertificate reads,
// so the lattice, HUD and catalog dividers all follow), then the schema is
// rendered exactly as an "open" from the lattice would render it.
async function openSchemaAtTau(features, tau) {
    if (lastTargetCol === null) return;
    const tauEl = document.getElementById('certTau');
    if (tauEl) {
        tauEl.value = String(Math.round(tau * 100));
        syncColorScaleFromInputs();
    }
    await runAnalysis(lastTargetCol, lastCriterion, features.slice());
}
