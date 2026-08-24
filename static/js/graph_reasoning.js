/**
 * Graph Inference Reasoning Engine & Visualizer
 */

let allColumns = [];
let activeInputs = {};
let graphData = null;
let currentPhase = 0;
let phaseTimer = null;
let nodeElements = {};
let nodePositions = {};

// DOM Elements
const canvasContainer = document.getElementById('canvasContainer');
const nodesContainer = document.getElementById('nodesContainer');
const edgesSvg = document.getElementById('edgesSvg');
const popover = document.getElementById('nodePopover');
const layersOverlay = document.getElementById('layersOverlay');

// Phase descriptions
const PHASES = {
    0: { name: 'Ожидание', delay: 0 },
    1: { name: 'Фаза 1: База знаний (Центр)', delay: 1000 },
    2: { name: 'Фаза 2: Поляризация (Входы и Цель)', delay: 1200 },
    3: { name: 'Фаза 3: Первая волна вывода (Слой 1)', delay: 1200 },
    4: { name: 'Фаза 4: Глубокие зависимости (Слой 2)', delay: 1200 },
    5: { name: 'Фаза 5: Трассировка лучших путей', delay: 1200 },
    6: { name: 'Фаза 6: Итоговый вердикт', delay: 0 }
};

async function init() {
    await fetchColumns();
    setupEventListeners();
}

async function fetchColumns() {
    try {
        const res = await fetch('/api/columns');
        const data = await res.json();
        allColumns = data.columns;
        
        const inputColSelect = document.getElementById('newInputCol');
        const targetSelect = document.getElementById('targetSelect');
        
        inputColSelect.innerHTML = '<option value="">Выберите признак...</option>';
        targetSelect.innerHTML = '';
        
        allColumns.forEach(col => {
            // Populate inputs dropdown
            if (col.id !== 'class') {
                const opt = document.createElement('option');
                opt.value = col.id;
                opt.textContent = col.label;
                inputColSelect.appendChild(opt);
            }
            // Populate target dropdown
            const optT = document.createElement('option');
            optT.value = col.id;
            optT.textContent = col.label;
            if (col.id === 'class') optT.selected = true;
            targetSelect.appendChild(optT);
        });
        
        updateTargetCriteria();
    } catch (e) {
        console.error("Failed to fetch columns", e);
    }
}

function updateTargetCriteria() {
    const targetCol = document.getElementById('targetSelect').value;
    const critSelect = document.getElementById('targetCriterionSelect');
    critSelect.innerHTML = '';
    
    const colDef = allColumns.find(c => c.id === targetCol);
    if (colDef && colDef.criteria) {
        colDef.criteria.forEach(crit => {
            const opt = document.createElement('option');
            opt.value = crit.id;
            opt.textContent = crit.label;
            critSelect.appendChild(opt);
        });
    }
}

function setupEventListeners() {
    const colSelect = document.getElementById('newInputCol');
    const valSelect = document.getElementById('newInputVal');
    const btnAdd = document.getElementById('btnAddInput');
    const targetSelect = document.getElementById('targetSelect');
    const slider = document.getElementById('nmiThreshold');
    
    colSelect.addEventListener('change', () => {
        valSelect.innerHTML = '<option value="">Значение...</option>';
        if (colSelect.value) {
            valSelect.disabled = false;
            const colDef = allColumns.find(c => c.id === colSelect.value);
            if (colDef) {
                colDef.criteria.forEach(crit => {
                    const opt = document.createElement('option');
                    opt.value = crit.id;
                    opt.textContent = crit.label;
                    valSelect.appendChild(opt);
                });
            }
        } else {
            valSelect.disabled = true;
        }
    });
    
    targetSelect.addEventListener('change', updateTargetCriteria);
    
    btnAdd.addEventListener('click', () => {
        const col = colSelect.value;
        const val = valSelect.value;
        if (col && val) {
            activeInputs[col] = val;
            renderActiveInputs();
            colSelect.value = "";
            valSelect.value = "";
            valSelect.disabled = true;
        }
    });
    
    slider.addEventListener('input', (e) => {
        document.getElementById('nmiThresholdLabel').textContent = parseFloat(e.target.value).toFixed(2);
    });
    
    document.getElementById('btnRunInference').addEventListener('click', runInference);
    
    // Playback Controls
    document.getElementById('btnPlayPause').addEventListener('click', togglePlayPause);
    document.getElementById('btnNextPhase').addEventListener('click', () => { pauseAnimation(); nextPhase(); });
    document.getElementById('btnPrevPhase').addEventListener('click', () => { pauseAnimation(); prevPhase(); });
    
    document.getElementById('popoverCloseBtn').addEventListener('click', closePopover);
    
    // Knowledge Base Controls
    document.getElementById('btnOpenKB').addEventListener('click', openKBModal);
    document.getElementById('kbCloseBtn').addEventListener('click', closeKBModal);
    document.getElementById('btnMineLinks').addEventListener('click', mineGlobalLinks);
}

function renderActiveInputs() {
    const container = document.getElementById('activeInputs');
    container.innerHTML = '';
    
    for (const [col, val] of Object.entries(activeInputs)) {
        const colDef = allColumns.find(c => c.id === col);
        const crit = colDef?.criteria.find(c => c.id === val);
        
        const badge = document.createElement('div');
        badge.className = 'input-badge';
        badge.innerHTML = `
            <span class="badge-text"><b>${colDef?.label || col}</b>: ${crit?.label || val}</span>
            <span class="remove-btn" data-col="${col}">×</span>
        `;
        container.appendChild(badge);
    }
    
    container.querySelectorAll('.remove-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            delete activeInputs[e.target.dataset.col];
            renderActiveInputs();
        });
    });
}

async function runInference() {
    if (Object.keys(activeInputs).length === 0) {
        alert("Пожалуйста, добавьте хотя бы одно входное наблюдение (например: Запах = гнилостный)!");
        return;
    }
    
    const target = document.getElementById('targetSelect').value;
    const targetCriterion = document.getElementById('targetCriterionSelect').value;
    const nmi = document.getElementById('nmiThreshold').value;
    
    try {
        const res = await fetch('/api/graph_inference', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                inputs: activeInputs,
                target: target,
                target_criterion: targetCriterion,
                nmi_threshold: parseFloat(nmi)
            })
        });
        
        graphData = await res.json();
        
        document.getElementById('playbackControls').style.display = 'flex';
        document.getElementById('summarySection').style.display = 'none';
        layersOverlay.style.display = 'flex';
        
        startAnimationSequence();
    } catch (e) {
        console.error("Inference execution failed", e);
    }
}

// ==========================================
// ANIMATION & LAYOUT ENGINE (NO OVERLAPS)
// ==========================================

function startAnimationSequence() {
    pauseAnimation();
    currentPhase = 1;
    createNodeCards();
    applyPhase(currentPhase);
    scheduleNextPhase();
}

function pauseAnimation() {
    if (phaseTimer) {
        clearTimeout(phaseTimer);
        phaseTimer = null;
    }
    document.getElementById('btnPlayPause').textContent = '▶️';
}

function togglePlayPause() {
    if (phaseTimer) {
        pauseAnimation();
    } else {
        document.getElementById('btnPlayPause').textContent = '⏸️';
        if (currentPhase >= 6) currentPhase = 0;
        nextPhase();
    }
}

function scheduleNextPhase() {
    if (currentPhase < 6) {
        document.getElementById('btnPlayPause').textContent = '⏸️';
        phaseTimer = setTimeout(() => {
            nextPhase();
        }, PHASES[currentPhase].delay);
    } else {
        document.getElementById('btnPlayPause').textContent = '🔁';
    }
}

function nextPhase() {
    if (currentPhase < 6) {
        currentPhase++;
        applyPhase(currentPhase);
        scheduleNextPhase();
    }
}

function prevPhase() {
    if (currentPhase > 1) {
        currentPhase--;
        applyPhase(currentPhase);
    }
}

function createNodeCards() {
    nodesContainer.innerHTML = '';
    edgesSvg.innerHTML = '';
    nodeElements = {};
    nodePositions = {};
    
    graphData.nodes.forEach(n => {
        const el = document.createElement('div');
        el.className = 'graph-node';
        el.id = `node-${n.id}`;
        
        // Display texts
        const titleText = n.label.split('(')[0].trim();
        let roleText = 'Фича';
        if (n.is_input) roleText = 'Вход';
        else if (n.is_target) roleText = 'Цель';
        else if (n.layer > 0) roleText = `Слой ${n.layer}`;
        
        let predValText = n.top_prediction ? n.top_prediction.label : '-';
        let predProbText = n.top_prediction ? `${(n.top_prediction.posterior * 100).toFixed(0)}%` : '';
        
        if (n.is_input) {
            const inpVal = activeInputs[n.id];
            const colDef = allColumns.find(c => c.id === n.id);
            const crit = colDef?.criteria.find(c => c.id === inpVal);
            predValText = crit ? crit.label : inpVal;
            predProbText = '100%';
        } else if (n.is_target) {
            predValText = graphData.target_criterion_label || graphData.target_criterion;
            predProbText = `${(graphData.target_probability * 100).toFixed(1)}%`;
        }
        
        el.innerHTML = `
            <div class="node-header-row">
                <div class="node-title" title="${n.label}">${titleText}</div>
                <div class="node-role-badge">${roleText}</div>
            </div>
            <div class="node-pred-row">
                <div class="node-pred-val" title="${predValText}">${predValText}</div>
                <div class="node-pred-prob">${predProbText}</div>
            </div>
        `;
        
        el.addEventListener('click', (e) => showPopover(n, e));
        
        nodesContainer.appendChild(el);
        nodeElements[n.id] = el;
    });
    
    calculatePositions();
}

function calculatePositions() {
    const width = canvasContainer.clientWidth;
    const height = canvasContainer.clientHeight;
    const paddingX = 110;
    const usableW = width - paddingX * 2;
    const centerY = height / 2;
    
    // Group active nodes by layer
    // Find max layer index
    let maxLayer = 1;
    graphData.nodes.forEach(n => {
        if (n.layer > maxLayer) maxLayer = n.layer;
    });
    
    const layerGroups = {};
    for (let l = 0; l <= maxLayer; l++) layerGroups[l] = [];
    
    graphData.nodes.forEach(n => {
        if (n.layer >= 0) {
            layerGroups[n.layer].push(n.id);
        }
    });
    
    // Center Cluster Coordinates (Phase 1)
    const totalNodes = graphData.nodes.length;
    graphData.nodes.forEach((n, i) => {
        const angle = (i / totalNodes) * Math.PI * 2;
        const radius = Math.min(width, height) * 0.28;
        nodePositions[n.id] = {
            clusterX: width / 2 + Math.cos(angle) * radius,
            clusterY: centerY + Math.sin(angle) * (radius * 0.7),
            polarizedX: 0,
            polarizedY: 0
        };
    });
    
    // Layered Coordinates (Phase 2+) - STRICT NON-OVERLAP FORMULA
    const nodeH = 76;
    const gapY = 16;
    
    for (let l = 0; l <= maxLayer; l++) {
        const group = layerGroups[l];
        const count = group.length;
        
        // Calculate X for column
        const colX = paddingX + (l / maxLayer) * usableW;
        
        // Calculate Y for each node in this column
        const totalHeight = count * nodeH + (count - 1) * gapY;
        const startY = Math.max(70, centerY - totalHeight / 2 + nodeH / 2);
        
        group.forEach((nodeId, idx) => {
            const nodeY = startY + idx * (nodeH + gapY);
            nodePositions[nodeId].polarizedX = colX;
            nodePositions[nodeId].polarizedY = nodeY;
        });
    }
    
    // Dimmed nodes (layer == -1) placed off-canvas
    graphData.nodes.filter(n => n.layer < 0).forEach((n, idx) => {
        nodePositions[n.id].polarizedX = width / 2 + (idx - 5) * 40;
        nodePositions[n.id].polarizedY = height + 150;
    });
}

function applyPhase(phase) {
    document.getElementById('phaseIndicator').textContent = PHASES[phase].name;
    
    if (phase === 1) {
        // Phase 1: Knowledge Base Cluster in Center
        graphData.nodes.forEach(n => {
            const el = nodeElements[n.id];
            const pos = nodePositions[n.id];
            el.style.left = `${pos.clusterX}px`;
            el.style.top = `${pos.clusterY}px`;
            el.className = 'graph-node';
        });
        edgesSvg.innerHTML = '';
    } else {
        // Phase 2+: Polarized Layers
        graphData.nodes.forEach(n => {
            const el = nodeElements[n.id];
            const pos = nodePositions[n.id];
            el.style.left = `${pos.polarizedX}px`;
            el.style.top = `${pos.polarizedY}px`;
            
            if (n.layer < 0) {
                el.classList.add('node-dimmed');
            } else {
                el.classList.remove('node-dimmed');
            }
            
            if (n.is_input) el.classList.add('node-input');
            else if (n.is_target) el.classList.add('node-target');
        });
    }
    
    if (phase >= 3) {
        // Draw layer 1 connections
        drawEdgesForMaxLayer(1);
        graphData.nodes.filter(n => n.layer === 1).forEach(n => {
            nodeElements[n.id].classList.add('node-active');
        });
    }
    
    if (phase >= 4) {
        // Draw deep layers
        drawEdgesForMaxLayer(999);
        graphData.nodes.filter(n => n.layer > 1 && !n.is_target).forEach(n => {
            nodeElements[n.id].classList.add('node-active');
        });
    }
    
    if (phase >= 5) {
        // Highlight golden target paths
        highlightTargetPaths();
        const targetNode = graphData.nodes.find(n => n.is_target);
        if (targetNode) nodeElements[targetNode.id].classList.add('node-active');
    }
    
    if (phase === 6) {
        renderFinalSummary();
    }
}

function drawEdgesForMaxLayer(maxLayer) {
    edgesSvg.innerHTML = '';
    
    graphData.edges.forEach(e => {
        const srcNode = graphData.nodes.find(n => n.id === e.source);
        const tgtNode = graphData.nodes.find(n => n.id === e.target);
        
        if (srcNode && tgtNode && srcNode.layer >= 0 && tgtNode.layer >= 0) {
            if (srcNode.layer <= maxLayer - 1) {
                const p1 = nodePositions[e.source];
                const p2 = nodePositions[e.target];
                
                const x1 = p1.polarizedX;
                const y1 = p1.polarizedY;
                const x2 = p2.polarizedX;
                const y2 = p2.polarizedY;
                
                const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                const cx1 = x1 + (x2 - x1) * 0.5;
                const cy1 = y1;
                const cx2 = x1 + (x2 - x1) * 0.5;
                const cy2 = y2;
                
                const d = `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`;
                
                path.setAttribute("d", d);
                path.setAttribute("class", "edge-line edge-active");
                path.setAttribute("data-source", e.source);
                path.setAttribute("data-target", e.target);
                path.style.opacity = Math.max(0.2, Math.min(0.9, e.nmi * 1.5));
                
                edgesSvg.appendChild(path);
            }
        }
    });
}

function highlightTargetPaths() {
    const paths = edgesSvg.querySelectorAll('path');
    const targetId = graphData.nodes.find(n => n.is_target)?.id;
    
    paths.forEach(p => {
        if (p.dataset.target === targetId) {
            p.classList.add('edge-target-path');
            p.style.opacity = '1';
        }
    });
}

function renderFinalSummary() {
    const summarySection = document.getElementById('summarySection');
    summarySection.style.display = 'block';
    
    // Direct vs Chain MI
    document.getElementById('statDirectNmi').textContent = (graphData.direct_nmi * 100).toFixed(1) + '%';
    document.getElementById('statChainNmi').textContent = (graphData.chain_score * 100).toFixed(1) + '%';
    
    // Verdict
    const verdictBanner = document.getElementById('verdictBanner');
    const isEdible = graphData.target_criterion === 'e';
    verdictBanner.className = isEdible ? 'verdict-banner edible' : 'verdict-banner';
    
    document.getElementById('verdictValue').textContent = `${graphData.target_criterion_label} (${graphData.target_criterion})`;
    document.getElementById('verdictProb').textContent = `${(graphData.target_probability * 100).toFixed(1)}% Уверенность`;
    
    // Narrative List
    const narrativeList = document.getElementById('chainNarrativeList');
    narrativeList.innerHTML = '';
    
    graphData.reasoning_steps.forEach(s => {
        const item = document.createElement('div');
        item.className = `narrative-item ${s.type}`;
        
        let connInfo = s.nmi_next ? `<div style="font-size: 0.68rem; color: var(--accent-cyan);">↓ Связь NMI: ${(s.nmi_next * 100).toFixed(1)}%</div>` : '';
        
        item.innerHTML = `
            <div class="narrative-item-title">${s.title}</div>
            <div class="narrative-item-detail">${s.detail}</div>
            ${connInfo}
        `;
        narrativeList.appendChild(item);
    });
}

// ==========================================
// POPOVER & BELIEF DISTRIBUTION
// ==========================================

function showPopover(nodeData, event) {
    const title = document.getElementById('popoverTitle');
    const subtitle = document.getElementById('popoverSubtitle');
    const badge = document.getElementById('popoverBadge');
    
    title.textContent = nodeData.label.split('(')[0].trim();
    subtitle.textContent = nodeData.raw_name;
    
    if (nodeData.is_input) badge.textContent = "ВХОД";
    else if (nodeData.is_target) badge.textContent = "ЦЕЛЬ";
    else badge.textContent = `СЛОЙ ${nodeData.layer}`;
    
    const list = document.getElementById('popoverProbList');
    list.innerHTML = '';
    
    nodeData.distribution.slice(0, 6).forEach(d => {
        if (d.prior === 0 && d.posterior === 0) return;
        
        const row = document.createElement('div');
        row.className = 'prob-row';
        const probPct = (d.posterior * 100).toFixed(1);
        
        row.innerHTML = `
            <div class="prob-label">
                <span>${d.label}</span>
                <strong>${probPct}%</strong>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: 0%"></div>
            </div>
        `;
        list.appendChild(row);
        
        setTimeout(() => {
            row.querySelector('.prob-bar-fill').style.width = `${probPct}%`;
        }, 50);
    });
    
    // Position popover near mouse
    const canvasRect = canvasContainer.getBoundingClientRect();
    let left = event.clientX - canvasRect.left + 20;
    let top = event.clientY - canvasRect.top - 60;
    
    if (left + 300 > canvasContainer.clientWidth) left = canvasContainer.clientWidth - 310;
    if (top < 10) top = 10;
    
    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
    popover.classList.remove('hidden');
    
    event.stopPropagation();
}

function closePopover() {
    popover.classList.add('hidden');
}

document.addEventListener('click', closePopover);
popover.addEventListener('click', (e) => e.stopPropagation());

window.addEventListener('resize', () => {
    if (graphData) {
        calculatePositions();
        applyPhase(currentPhase);
    }
});

// Start initialization
init();

// ==========================================
// KNOWLEDGE BASE (GLOBAL NMI MINER)
// ==========================================

function openKBModal() {
    const targetSelect = document.getElementById('kbTargetSelect');
    const mainTargetVal = document.getElementById('targetSelect').value;
    targetSelect.innerHTML = '<option value="all">🌐 Весь датасет (Искать любые побеждающие цепочки)</option>';
    
    allColumns.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col.id;
        opt.textContent = col.label;
        if (col.id === mainTargetVal) opt.selected = true;
        targetSelect.appendChild(opt);
    });
    
    document.getElementById('kbModal').classList.remove('hidden');
}

function closeKBModal() {
    document.getElementById('kbModal').classList.add('hidden');
}

async function mineGlobalLinks() {
    const target = document.getElementById('kbTargetSelect').value;
    const minNmi = parseFloat(document.getElementById('kbMinNmiSelect').value);
    
    const resultsContainer = document.getElementById('kbResults');
    resultsContainer.innerHTML = '<div class="kb-empty-state">Поиск цепочек, где вывод через медиаторов превосходит прямой замер... ⏳</div>';
    
    try {
        const res = await fetch('/api/mine_graph_links', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ target: target, min_nmi: minNmi, max_depth: 3, only_winning: true })
        });
        
        const data = await res.json();
        renderKBResults(data);
    } catch (e) {
        console.error("Mining failed", e);
        resultsContainer.innerHTML = '<div class="kb-empty-state" style="color: #ef4444;">Ошибка майнинга</div>';
    }
}

function renderKBResults(data) {
    const container = document.getElementById('kbResults');
    container.innerHTML = '';
    
    if (!data.top_chains || data.top_chains.length === 0) {
        container.innerHTML = '<div class="kb-empty-state">Побеждающие цепочки не найдены для выбранных условий. Попробуйте снизить порог NMI.</div>';
        return;
    }
    
    const countBanner = document.createElement('div');
    countBanner.style.fontSize = '0.85rem';
    countBanner.style.color = 'var(--text-muted)';
    countBanner.style.marginBottom = '12px';
    countBanner.innerHTML = `Найдено <b>${data.total_found}</b> цепочек, где <b>Цепочка ≥ Прямой NMI</b> (Цель: <b>${data.target_label}</b>):`;
    container.appendChild(countBanner);
    
    data.top_chains.forEach((chain, idx) => {
        const card = document.createElement('div');
        card.className = 'kb-chain-card';
        
        let pathHtml = '';
        chain.path.forEach((node, i) => {
            const label = chain.path_labels[i].split('(')[0].trim();
            const isInput = i === 0;
            const isTarget = i === chain.path.length - 1;
            const cls = isInput ? 'kb-node input' : (isTarget ? 'kb-node target' : 'kb-node');
            
            pathHtml += `<span class="${cls}">${label}</span>`;
            if (i < chain.path.length - 1) {
                pathHtml += ` <span class="kb-arrow">➔</span> `;
            }
        });
        
        const chainPct = (chain.chain_score * 100).toFixed(1) + '%';
        const directPct = (chain.direct_nmi * 100).toFixed(1) + '%';
        const gainPct = (chain.gain * 100).toFixed(1) + '%';
        const ratioText = chain.ratio ? ` (x${chain.ratio.toFixed(1)})` : '';
        
        let gainHtml = '';
        if (chain.gain > 0.001) {
            gainHtml = `<span style="color: #10b981; font-weight: bold; font-size: 0.75rem;">+${gainPct} выигрыш${ratioText}</span>`;
        } else {
            gainHtml = `<span style="color: #a5b4fc; font-size: 0.75rem;">Эквивалент прямого пути</span>`;
        }
        
        card.innerHTML = `
            <div style="flex: 1; padding-right: 15px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 5px;">
                    <span style="font-size: 0.68rem; color: var(--accent-cyan); font-weight: 700; text-transform: uppercase;">${chain.type_label}</span>
                    ${gainHtml}
                </div>
                <div class="kb-chain-path">${pathHtml}</div>
                <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 4px;">
                    Прямой путь: <b>${directPct}</b> ➔ Косвенная цепочка: <b style="color: var(--accent-gold);">${chainPct}</b>
                </div>
            </div>
            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 6px;">
                <div class="kb-score" style="color: var(--accent-gold);">${chainPct}</div>
                <button class="kb-apply-btn" onclick="applyChainToGraph('${chain.input}', '${chain.target}')">Применить</button>
            </div>
        `;
        
        container.appendChild(card);
    });
}

function applyChainToGraph(inputId, targetId) {
    closeKBModal();
    
    // Clear active inputs
    activeInputs = {};
    
    // Set Target
    const targetSelect = document.getElementById('targetSelect');
    targetSelect.value = targetId;
    updateTargetCriteria();
    
    // Populate active input
    const colDef = allColumns.find(c => c.id === inputId);
    if (colDef && colDef.criteria.length > 0) {
        activeInputs[inputId] = colDef.criteria[0].id;
    }
    
    renderActiveInputs();
    
    // Auto Run
    runInference();
}
