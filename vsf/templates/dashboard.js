// vsf.dashboard client-side driver — v2.0 "Clean Core" (see
// Project_Master_Document.md). There is no server to fetch from:
// everything is precomputed in Python and embedded as DASHBOARD_DATA /
// BRANCHES_DATA (see vsf/dashboard.py). DASHBOARD_DATA.branch_dims lists
// every dimensionality (1..4) for which Independent Branch Discovery
// (Project_Master_Document.md Section 4) found a branch; BRANCHES_DATA is
// keyed by that dimensionality (as a string) and holds one fully
// independent `vsf.vis.prepare_visualization_payload` payload per branch —
// branch d=2's features are NOT guaranteed to be a subset/superset of
// branch d=1's or d=3's (feature synergy, Section 4.4).
//
// Two interactions this file must keep visually distinct
// (Project_Master_Document.md Section 5.6):
//   1. Branch selection (selectBranch): swaps `currentPayload` to a whole
//      new independently-discovered branch. INSTANT — no animation, since
//      the axes/features may be completely different and Object Constancy
//      does not apply across branches.
//   2. Within-branch dimensionality collapse/split (setDimensionality):
//      after a branch is selected, marginalizing it down to fewer spatial
//      axes (or back up, up to that branch's own d) is an ANIMATED
//      transition (transitionDimensionality) that preserves relative
//      circle sizes (Object Constancy) — it never changes which branch/
//      features are in play.
//
// Removed relative to v1.0 (see git history — do not
// re-add): the Scenario A/B/C/D picker and per-column "live scenario"
// switching (SCENARIOS_DATA/selectScenario), the Top Insights auto-
// discovery sidebar (TOP_INSIGHTS_DATA/populateTopCatalog), dirty-center
// conjunctive-filter mining and its floating XAI/Mitosis panel
// (MINING_DATA/renderXaiPanel/MitosisEngine), the composite AND-filter
// builder, and Graph Inference / Knowledge-Base chain mining. The plain
// per-column/criterion catalog below is kept, but purely for display: this
// static export was baked for exactly one target/criterion and has no
// server to refit against, so browsing the catalog never changes the
// visualizer.

let currentPayload = null;
let activeBranchD = null;          // string key into BRANCHES_DATA, e.g. "3"
let activeDimensionality = null;   // view-dimensionality WITHIN the active branch (1..branch.metrics.d)
let activeSliceIndex = null;       // current 4D slice index (null = "All" composite frame)
let currentRenderedDim = null;     // dimensionality Plotly is currently showing (null = nothing rendered yet)
let isAnimating = false;

// ---------------------------------------------------------------------
// Purity colour scale — 3 zones (Project_Master_Document.md Section 5.3).
// v2.2 removed the yellow band and the hard-coded 0.25 / 0.75 / 0.85
// boundaries. The GREEN boundary is tau, baked into this export at build time
// because Coverage was computed from it and a static page cannot recompute;
// the LOWER boundary is cosmetic and the reader can move it here.
//   [0, brownFrom)      -> red
//   [brownFrom, tau)    -> brown
//   [tau, 1]            -> green: a discrete centre, counted in Coverage
const PROB_COLORS = [
    '#ef4444', // Red   — below the reader's lower boundary
    '#8a5a34', // Brown — between the boundaries
    '#22c55e'  // Green — at or above tau: a discrete centre
];
// Absence export (DASHBOARD_DATA.direction === 'absence', see
// vsf.avr.Direction): the certified zone is drawn red -- "certified free of
// the value" -- and the low zone slate grey, never green.
const PROB_COLORS_ABSENCE = ['#64748b', '#8a5a34', '#ef4444'];
const ACTIVE_COLORS = (typeof DASHBOARD_DATA !== 'undefined' && DASHBOARD_DATA.direction === 'absence')
    ? PROB_COLORS_ABSENCE : PROB_COLORS;
const PALETTE_COUNT = PROB_COLORS.length;

function buildDiscreteColorscale() {
    const scale = [];
    for (let i = 0; i < PALETTE_COUNT; i++) {
        const lo = i / PALETTE_COUNT;
        const hi = (i + 1) / PALETTE_COUNT;
        scale.push([lo, ACTIVE_COLORS[i]]);
        scale.push([hi, ACTIVE_COLORS[i]]);
    }
    return scale;
}
const DISCRETE_COLORSCALE = buildDiscreteColorscale();

// v2.2 Certificate colour scale. See vsf/webapp/static/js/app.js for the
// full rationale; in brief, the v2.1 bands (0.25 / 0.75 / 0.85 on the POINT
// purity) had no derivation and drew a cell holding one sample of the target
// value as a fully saturated green centre. Every boundary below is either
// the certificate or the dataset-wide base rate, so none is a free
// parameter, and a singleton cell can never be green.
//
//   Green  -> certified: simultaneous lower bound on purity >= tau
//   Yellow -> candidate: upper bound >= tau, not certified
//   Brown  -> enriched over the base rate, provably below tau
//   Red    -> at or below the base rate
let dashboardTau = 0.90;
let dashboardPrevalence = 0.0;

// Rewrites the colour legend from the certificate baked into this export, so
// the swatch captions can never drift from the thresholds the renderer uses.
function renderCertificateLegend(payload) {
    const el = document.getElementById('purityLegend');
    if (!el) return;
    const cert = payload.certificate || {};
    const tau = ((cert.tau !== undefined && cert.tau !== null) ? cert.tau : 0.90) * 100;
    const alpha = ((cert.alpha !== undefined && cert.alpha !== null) ? cert.alpha : 0.05) * 100;
    const prev = (payload.centers && payload.centers.prevalence !== undefined)
        ? payload.centers.prevalence * 100 : 0;
    const label = cert.positive_label || payload.target_name || 'target';
    const brown = readBrownFrom() * 100;
    const strict = (cert.rule === 'certified');
    el.innerHTML = `
        <div class="purity-legend-item"><span class="purity-swatch" style="background: var(--purity-green);"></span><b>Discrete centre</b> — ${strict ? `lower bound ≥ ${tau.toFixed(0)}%` : `purity ≥ ${tau.toFixed(0)}%`}</div>
        <div class="purity-legend-item"><span class="purity-swatch" style="background: var(--purity-brown);"></span>Mixed — ${brown.toFixed(0)}% – ${tau.toFixed(0)}%</div>
        <div class="purity-legend-item"><span class="purity-swatch" style="background: var(--purity-red);"></span>Low — below ${brown.toFixed(0)}%</div>
        <div style="font-size:0.72rem;opacity:0.75;margin-top:6px;line-height:1.45;">
            Positive value: <b>${label}</b>. Base rate ${prev.toFixed(2)}%.
            Green cells are exactly the cells Coverage is computed from; the green
            boundary was fixed when this file was exported. Every cell carries a
            ${(100 - alpha).toFixed(0)}% Clopper–Pearson interval in its hover text.
            All figures on ${(payload.total_samples || 0).toLocaleString()} rows.
        </div>`;
}

function readBrownFrom() {
    const el = document.getElementById('colorBrownFrom');
    const v = el ? Number(el.value) / 100 : 0.40;
    return (Number.isFinite(v) && v >= 0 && v <= 1) ? v : 0.40;
}

function getColorIndexForCell(purity, isCenter, tau, brownFrom) {
    // `isCenter` was decided server-side at export time and is authoritative;
    // under the strict rule a cell can sit above tau by point purity and
    // still not be a centre.
    if (isCenter === true) return 2;
    if (isCenter === false && purity >= tau) return 1;
    if (purity >= tau) return 2;
    if (purity >= brownFrom) return 1;
    return 0;
}

// Re-colours in place when the reader moves the cosmetic lower boundary.
function applyColorBoundary() {
    if (!currentPayload) return;
    renderCertificateLegend(currentPayload);
    renderPlot(currentPayload);
}

// ---------------------------------------------------------------------
// Boot
function init() {
    populateCatalog(DASHBOARD_DATA.catalog || []);

    const allDims = DASHBOARD_DATA.branch_dims || [];
    if (allDims.length === 0 || !BRANCHES_DATA) {
        console.warn('vsf.dashboard: no branches were discovered for this target — nothing to render.');
        const container = document.getElementById('branchesAccordion');
        if (container) {
            container.innerHTML = '<div class="info-note">No branches were found for this target (the dataset may have no usable feature columns).</div>';
        }
        return;
    }

    refreshBranchesView(/* preserveActiveIfPossible */ false);
}

// Re-renders the branch list from BRANCHES_DATA (already fully baked into
// this export — no recomputation) and picks which branch becomes active.
// Called once on boot (preserveActiveIfPossible=false — always pick the
// server's own default). There is no client-side branch filter — every
// discovered branch is always shown.
function refreshBranchesView(preserveActiveIfPossible) {
    populateBranches();

    const allDims = (DASHBOARD_DATA.branch_dims || []).slice().sort((a, b) => a - b).map(String);
    if (allDims.length === 0) return; // handled by init()'s own early return

    if (preserveActiveIfPossible && activeBranchD && allDims.includes(activeBranchD)) {
        // Still valid — leave the visualizer exactly as-is, but
        // populateBranches() above just rebuilt the branch-list DOM from
        // scratch (it never sets `.active` on creation, unlike the live
        // app's card renderer), so the active marker has to be reapplied
        // here or the previously-selected card would render unmarked.
        _markActiveBranch(activeBranchD);
        return;
    }

    const preferredDefault = DASHBOARD_DATA.default_branch != null ? String(DASHBOARD_DATA.default_branch) : null;
    const nextKey = (preferredDefault && allDims.includes(preferredDefault))
        ? preferredDefault
        : allDims[allDims.length - 1];

    activeBranchD = nextKey;
    currentPayload = BRANCHES_DATA[nextKey];
    activeDimensionality = currentPayload.metrics.d; // show the branch at its own full dimensionality
    activeSliceIndex = currentPayload.slice_axis ? 0 : null;
    currentRenderedDim = null;

    const targetBadge = document.getElementById('targetBadge');
    if (targetBadge) targetBadge.innerText = 'Target: ' + (currentPayload.target_name || DASHBOARD_DATA.target);

    updateDashboard(currentPayload);
    _markActiveBranch(nextKey);
}

// ---------------------------------------------------------------------
// Branch selector (Master panel #1) — up to 4 independently-discovered
// branches (Project_Master_Document.md Section 4). Clicking one is an
// INSTANT swap: new axes, new features, no interpolation (Section 5.6,
// case 1) — never confuse this with setDimensionality()/
// transitionDimensionality() below, which animate WITHIN one branch.
function selectBranch(d) {
    const key = String(d);
    if (!BRANCHES_DATA || !BRANCHES_DATA[key]) {
        console.warn('vsf.dashboard: no branch data for d=', d);
        return;
    }
    if (key === activeBranchD) return;
    if (isAnimating) return; // don't interrupt an in-flight collapse/split animation

    // Stop any running 4D autoplay — it's animating a frame axis that
    // belongs to the branch being replaced.
    if (sliceAutoplayTimer !== null) {
        clearInterval(sliceAutoplayTimer);
        sliceAutoplayTimer = null;
        _setSliceAutoplayButtonState(false);
    }

    activeBranchD = key;
    currentPayload = BRANCHES_DATA[key];
    activeDimensionality = currentPayload.metrics.d; // show the branch at its own full dimensionality
    activeSliceIndex = currentPayload.slice_axis ? 0 : null;
    currentRenderedDim = null; // force a fresh, non-animated render (see setDimensionality)

    updateDashboard(currentPayload);
    _markActiveBranch(key);
}

function populateBranches() {
    const container = document.getElementById('branchesAccordion');
    if (!container) return;
    container.innerHTML = '';

    const allDims = (DASHBOARD_DATA.branch_dims || []).slice().sort((a, b) => a - b);
    if (allDims.length === 0) {
        container.innerHTML = '<div class="info-note">No branches were found for this target.</div>';
        return;
    }

    allDims.forEach(d => {
        const key = String(d);
        const branch = BRANCHES_DATA[key];
        if (!branch) return;
        const featuresText = (branch.selected_features || []).join(' + ');

        const item = document.createElement('div');
        item.className = 'branch-item';
        item.dataset.branchD = key;
        item.innerHTML = `
            <div class="branch-item-header">
                <span class="branch-item-dim">${d}D</span>
                <span class="branch-item-metrics">
                    <span class="branch-uadj" title="Share of all target-value samples inside certified discrete centres. This is what the branch is ranked and delivered by.">${
                        (branch.centers && branch.centers.n_centers)
                            ? `coverage ${(branch.centers.coverage * 100).toFixed(1)}% · ${branch.centers.n_centers} centre${branch.centers.n_centers === 1 ? '' : 's'}`
                            : 'no certified centres'
                    }</span>
                </span>
            </div>
            <div class="branch-item-features">${featuresText || '(no features selected)'}</div>
        `;
        item.onclick = () => selectBranch(d);
        container.appendChild(item);
    });
}

function _markActiveBranch(key) {
    document.querySelectorAll('#branchesAccordion .branch-item').forEach(item => {
        item.classList.toggle('active', item.dataset.branchD === key);
    });
}

// ---------------------------------------------------------------------
// Dataset Columns catalog (Master panel #2) — informational only. This
// export was baked for exactly one target/criterion; there is no server
// to refit against, so clicking a column/criterion here never changes the
// visualizer (unlike v1.0's live per-column scenario switching, which is
// removed — see git history). Only expand/collapse
// and a static note on the baked-in target/criterion.
function populateCatalog(cols) {
    const accordion = document.getElementById('catalogAccordion');
    if (!accordion) return;
    accordion.innerHTML = '';

    cols.forEach(col => {
        const charItem = document.createElement('div');
        charItem.className = 'char-item';
        charItem.dataset.colId = col.id;

        const charHeader = document.createElement('div');
        charHeader.className = 'char-header';
        charHeader.innerHTML = `
            <span class="char-title">${col.label}</span>
            <span class="char-icon">▶</span>
        `;
        charHeader.onclick = () => {
            charItem.classList.toggle('open');
        };

        const charContent = document.createElement('div');
        charContent.className = 'char-content';

        (col.criteria || []).forEach(crit => {
            const isActive = crit.active === true;

            const critItem = document.createElement('div');
            critItem.className = 'crit-item' + (isActive ? ' active' : ' inactive');

            const critHeader = document.createElement('div');
            critHeader.className = 'crit-header';
            critHeader.innerHTML = `<span>${crit.label}</span>`;

            const critContent = document.createElement('div');
            critContent.className = 'crit-content';
            critContent.innerHTML = isActive
                ? `<div class="info-note">This is the target/criterion this export was built for.</div>`
                : `<div class="info-note">This static export is baked for one target/criterion and cannot refit. To explore "${col.label} = ${crit.label}" instead, re-run <code>vsf.export_full_dashboard(df, target="${col.id}", criterion="${crit.id}")</code>.</div>`;

            if (isActive) {
                critItem.classList.add('open');
                charItem.classList.add('open');
            }

            critHeader.onclick = (e) => {
                e.stopPropagation();
                critItem.classList.toggle('open');
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

function toggleMainAcc(suffix) {
    const header = document.getElementById('header' + suffix);
    const content = document.getElementById('content' + suffix);
    if (!header || !content) return;
    const nowOpen = header.classList.toggle('open');
    content.classList.toggle('open');
    header.setAttribute('aria-expanded', nowOpen ? 'true' : 'false');
}

function showLoader(show) {
    const loader = document.getElementById('loader');
    if (!loader) return;
    if (show) loader.classList.add('active');
    else loader.classList.remove('active');
}

// ---------------------------------------------------------------------
// Detail panel: renders `currentPayload` (one branch) and its toolbar/HUD.
function updateDashboard(payload) {
    currentPayload = payload;
    currentRenderedDim = null; // force a full, non-animated re-render (new axes/features)

    if (payload.slice_axis) {
        renderSliceTabs(payload);
    } else {
        const sliceCtrl = document.getElementById('slice-controller');
        if (sliceCtrl) sliceCtrl.style.display = 'none';
    }

    // v2.2: `total_samples` is the N every statistic is computed on;
    // `rendered_samples` is how many rows are drawn as spheres. The two were
    // silently conflated before (see Project_Master_Document.md 0-ter).
    const totalEl = document.getElementById('totalSamplesVal');
    if (totalEl) {
        const nStat = payload.total_samples || (payload.x ? payload.x.length : 0);
        const nDrawn = (payload.rendered_samples !== undefined && payload.rendered_samples !== null)
            ? payload.rendered_samples : nStat;
        totalEl.innerText = (nDrawn < nStat)
            ? `${nStat.toLocaleString()} (${nDrawn.toLocaleString()} drawn)`
            : nStat.toLocaleString();
    }
    renderCertificateLegend(payload);

    // Populate Exact Values for the stroke ("Purity Contour") settings
    const exactSelect = document.getElementById('strokeExactVal');
    if (exactSelect) {
        exactSelect.innerHTML = '<option value="">-- Select --</option>';
        const dimStr = (activeDimensionality || payload.metrics.d).toString();
        const g = payload.grids ? payload.grids[dimStr] : null;
        const purities = g ? g.purity : payload.grid_purity;
        if (purities) {
            const uniqueP = [...new Set(purities)].sort((a, b) => a - b);
            uniqueP.forEach(p => {
                const pct = (p * 100).toFixed(1);
                exactSelect.innerHTML += `<option value="${p}">${pct}%</option>`;
            });
        }
    }

    renderDimensionButtons(payload);
    setDimensionality(activeDimensionality);
}

// Within-branch dimensionality buttons: 1..branch.metrics.d only — a
// branch never offers MORE spatial dims than it was itself found with
// (Project_Master_Document.md Section 5.6, case 2: collapse/split moves
// within one branch's own axes, it never invents a new one).
function renderDimensionButtons(payload) {
    const container = document.getElementById('dimButtonsGroup');
    if (!container) return;
    container.innerHTML = '';

    const maxD = Math.min((payload.metrics && payload.metrics.d) ? payload.metrics.d : 1, 4);
    for (let d = 1; d <= maxD; d++) {
        const btn = document.createElement('button');
        btn.className = 'toggle-btn dim-btn' + (d === activeDimensionality ? ' active' : '');
        btn.dataset.dim = d;
        btn.innerText = `${d}D`;
        btn.title = (d === maxD)
            ? `This branch's full ${d}D system`
            : `Collapse this branch to ${d}D (marginalize the remaining axis${maxD - d > 1 ? 'es' : ''})`;
        btn.onclick = () => setDimensionality(d);
        container.appendChild(btn);
    }
}

// v2.3: `formatSignificance` (raw-MI p-value formatting) was removed along
// with `BranchResult.mi`/`p_value`/`p_value_familywise` (see vsf.avr's
// module docstring) -- the coverage-search p-value this export still ships
// is `search_centers.coverage_p_value(_familywise)`, read directly below.
//
// HUD metrics — every readout must track the CURRENTLY VIEWED collapsed
// dimensionality `viewD`, not the branch's fixed full-d aggregate: a
// viewD < m.d readout is a projection of this branch's own axes onto its
// first `viewD` of them, and generally carries LESS information than the
// full branch (see vsf.vis's `view_metrics` and
// vsf.avr.BranchResult.coverage_by_prefix_d docstrings) — showing the full-branch
// values while collapsed silently overstates what the visible axes alone
// explain.
//
// The p-value is NOT recomputed per collapsed view: it belongs to the branch
// as searched and selected. It is therefore shown unchanged, and the
// dimensionality label makes clear when the view is a projection.
function updateHUDForDimension(viewD) {
    if (!currentPayload || !currentPayload.metrics) return;
    const m = currentPayload.metrics;
    const vm = currentPayload.view_metrics;
    const dKey = String(viewD);

    const dEl = document.getElementById('val-branch-d');
    if (dEl) dEl.innerText = (viewD === m.d) ? `${m.d}D` : `${viewD}D (of ${m.d}D branch)`;

    // v2.2 headline: coverage / K / pooled purity for the CURRENTLY VIEWED
    // dimensionality, read from the displayed partition itself.
    const pick = (obj, fallback) =>
        (obj && obj[dKey] !== undefined && obj[dKey] !== null) ? obj[dKey] : fallback;
    const cBlock = currentPayload.centers || {};
    const sc = currentPayload.search_centers;
    const viewCoverage = pick(vm && vm.coverage_by_d, cBlock.coverage);
    const viewCenters = pick(vm && vm.n_centers_by_d, cBlock.n_centers);
    const viewPurity = pick(vm && vm.purity_by_d, cBlock.purity_pooled);

    const covEl = document.getElementById('val-coverage');
    if (covEl) {
        if (viewCoverage === undefined || viewCoverage === null) {
            covEl.innerText = 'n/a';
            covEl.style.color = 'var(--text-dim)';
        } else if (sc && sc.undetermined_reason) {
            covEl.innerText = 'undetermined';
            covEl.style.color = 'var(--text-dim)';
            covEl.title = sc.undetermined_reason;
        } else {
            covEl.innerText = `${(viewCoverage * 100).toFixed(1)}%`;
            covEl.style.color = viewCoverage > 0 ? 'var(--green)' : 'var(--text-dim)';
            const cv = sc && sc.coverage_cv;
            if (cv) {
                covEl.title = `In-sample ${(viewCoverage * 100).toFixed(1)}%. Out-of-sample (${cv.n_repeats}x${cv.n_splits}-fold, Nadeau-Bengio SE): ${(cv.mean * 100).toFixed(1)}% +/- ${(cv.se * 100).toFixed(1)} points.`;
            }
        }
    }
    const kEl = document.getElementById('val-centers');
    if (kEl) kEl.innerText = (viewCenters === undefined || viewCenters === null) ? 'n/a' : String(viewCenters);
    const purEl = document.getElementById('val-purity');
    if (purEl) {
        purEl.innerText = viewCenters
            ? `purity ${(viewPurity * 100).toFixed(1)}%`
            : (cBlock.max_purity_lower !== undefined
                ? `best bound ${(cBlock.max_purity_lower * 100).toFixed(1)}%`
                : 'purity —');
    }

    const pEl = document.getElementById('val-pvalue');
    if (pEl) {
        const p = (sc && sc.coverage_p_value_familywise !== null && sc.coverage_p_value_familywise !== undefined)
            ? sc.coverage_p_value_familywise
            : (cBlock && cBlock.coverage_p_value);
        pEl.innerText = (p === null || p === undefined) ? 'not tested' : p.toFixed(3);
        pEl.style.color = (p !== null && p !== undefined && p <= 0.01)
            ? 'var(--green)' : 'var(--text-dim)';
    }
}

function setDimensionality(d) {
    if (d === currentRenderedDim && d === activeDimensionality) return;
    if (isAnimating) return;

    const fromDim = currentRenderedDim;
    activeDimensionality = d;

    document.querySelectorAll('#dimButtonsGroup .dim-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.dim, 10) === d);
    });

    updateHUDForDimension(d);

    const sliceCtrl = document.getElementById('slice-controller');
    if (sliceCtrl) {
        sliceCtrl.style.display = (d >= 4 && currentPayload && currentPayload.slice_axis) ? 'flex' : 'none';
    }

    if (currentPayload) {
        if (fromDim === null) {
            // Either the very first render, or we just switched branches
            // (updateDashboard always resets currentRenderedDim to null) —
            // either way: instant render, no Object Constancy animation.
            renderPlot(currentPayload);
        } else {
            // Same branch, different view-dimensionality: animated
            // collapse/split (Project_Master_Document.md Section 5.6, case 2).
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
    // "All" tab — marginalizes over the 4th axis into one composite 3D view.
    const allActiveClass = (activeSliceIndex === null) ? ' active' : '';
    tabsHtml += `<button class="slice-tab${allActiveClass}" onclick="selectSlice(null)" data-slice="all">All<span class="slice-count">(${payload.total_samples})</span></button>`;

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
            btn.classList.toggle('active', parseInt(btn.dataset.dim, 10) === 4);
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

// ---------------------------------------------------------------------
// 4D Time / Frame Controller (step / play / pause / speed) — unchanged in
// shape from v1.0: the 4th selected feature's K categories are K
// chronological frames of the 3D scene (payload.slice_axis.ticks, grids
// "4_0".."4_{K-1}"), plus a composite "All" frame (grid "4_all",
// activeSliceIndex === null). Every function below is a thin driver on top
// of the *existing* selectSlice(idx) — the same function a frame-tab click
// calls — so there remains exactly one code path that actually re-renders
// the plot for a frame change.
let sliceAutoplayTimer = null;
let sliceAutoplaySpeedMs = 800;

function _sliceFrameCount() {
    return (currentPayload && currentPayload.slice_axis && Array.isArray(currentPayload.slice_axis.ticks))
        ? currentPayload.slice_axis.ticks.length
        : 0;
}

function stepSlice(direction) {
    const K = _sliceFrameCount();
    if (K === 0 || isAnimating) return;
    const cur = (activeSliceIndex === null) ? -1 : activeSliceIndex;
    const next = ((cur + direction) % K + K) % K;
    selectSlice(next);
}

function _setSliceAutoplayButtonState(playing) {
    const btn = document.getElementById('btnSlicePlayPause');
    if (!btn) return;
    btn.classList.toggle('playing', playing);
    btn.innerText = playing ? '⏸️' : '▶️';
    btn.title = playing ? 'Pause' : 'Play';
}

function toggleSliceAutoplay() {
    if (_sliceFrameCount() === 0) return;

    if (sliceAutoplayTimer !== null) {
        clearInterval(sliceAutoplayTimer);
        sliceAutoplayTimer = null;
        _setSliceAutoplayButtonState(false);
        return;
    }

    _setSliceAutoplayButtonState(true);
    // Advance one frame immediately so pressing play is visibly responsive
    // instead of waiting a full interval on an unchanged frame.
    stepSlice(1);
    sliceAutoplayTimer = setInterval(() => stepSlice(1), sliceAutoplaySpeedMs);
}

function setSliceAutoplaySpeed(ms) {
    const parsed = parseInt(ms, 10);
    sliceAutoplaySpeedMs = Number.isFinite(parsed) && parsed > 0 ? parsed : 800;
    if (sliceAutoplayTimer !== null) {
        clearInterval(sliceAutoplayTimer);
        sliceAutoplayTimer = setInterval(() => stepSlice(1), sliceAutoplaySpeedMs);
    }
}

// ---------------------------------------------------------------------
// Plot construction & rendering (unchanged in shape from v1.0 — this
// machinery already operates purely on whatever `payload` it's handed, so
// it works identically whether `payload` is one branch of many or, as in
// v1.0, the single fitted scenario).
function buildPlotData(payload, dim, sliceIndex) {
    const dimStr = dim.toString();

    let gridKey = dimStr;
    if (dim >= 4 && payload.slice_axis) {
        gridKey = (sliceIndex !== null) ? `4_${sliceIndex}` : `4_all`;
    }

    let g = null;
    if (payload.grids) {
        g = payload.grids[gridKey] || payload.grids[dimStr] || payload.grids["3"];
    }

    const xCoords = g ? g.x : payload.grid_x;
    const yCoords = g ? g.y : payload.grid_y;
    const zCoords = g ? g.z : payload.grid_z;

    const currentPurity = g ? g.purity : payload.grid_purity;
    const currentCertified = g ? g.certified : payload.grid_certified;
    const cert = payload.certificate || {};
    dashboardTau = (cert.tau !== undefined && cert.tau !== null) ? cert.tau : 0.90;
    dashboardPrevalence = (payload.centers && payload.centers.prevalence !== undefined)
        ? payload.centers.prevalence : 0.0;
    const currentSizes = g ? g.sizes : payload.grid_sizes;
    const currentHover = g ? g.hover_text : payload.grid_hover_text;
    const currentCustomdata = g ? g.customdata : payload.grid_customdata;

    const xLen = payload.axis_ticks ? payload.axis_ticks.x.vals.length : 4;
    const yLen = payload.axis_ticks ? payload.axis_ticks.y.vals.length : 4;
    const zLen = payload.axis_ticks ? payload.axis_ticks.z.vals.length : 4;

    const xMin = -0.5, xMax = xLen - 0.5;
    const yMin = -0.5, yMax = yLen - 0.5;
    const zMin = -0.5, zMax = zLen - 0.5;

    const glX = [], glY = [], glZ = [];

    // ALWAYS draw the full 3D room for a consistent spatial metaphor.
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

        const colorIndex = getColorIndexForCell(
            p,
            (currentCertified && currentCertified[i] !== undefined) ? currentCertified[i] : undefined,
            dashboardTau,
            readBrownFrom()
        );

        fx.push(xCoords[i]);
        fy.push(yCoords[i]);
        fz.push(zCoords[i]);

        const hex = ACTIVE_COLORS[colorIndex];
        const rr = parseInt(hex.slice(1, 3), 16);
        const gg = parseInt(hex.slice(3, 5), 16);
        const bb = parseInt(hex.slice(5, 7), 16);
        fColors.push(`rgb(${rr}, ${gg}, ${bb})`);

        fPurity.push(p);

        // Area Scaling (Tufte, 1983): diameter ~ sqrt(N). Maximum diameter
        // set so the 1D cluster perfectly touches cell bounds; 55 is an
        // empirical constant for Plotly's 3D scatter marker sizing.
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
    const dim = activeDimensionality || payload.metrics.d || 1;
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

// Within-branch collapse/split transition (Project_Master_Document.md
// Section 5.6, case 2). `currentPayload` never changes here — only which
// view-dimensionality of that SAME branch is displayed.
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

window.addEventListener('DOMContentLoaded', init);
