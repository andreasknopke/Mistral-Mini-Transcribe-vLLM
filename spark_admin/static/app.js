const bootstrapNode = document.getElementById('bootstrap-data');
const bootstrap = bootstrapNode ? JSON.parse(bootstrapNode.textContent) : { services: [], configs: [] };

const serviceCards = document.getElementById('service-cards');
const systemSummary = document.getElementById('system-summary');
const gpuCards = document.getElementById('gpu-cards');
const logServiceSelect = document.getElementById('log-service-select');
const logsOutput = document.getElementById('logs-output');
const refreshLogsButton = document.getElementById('refresh-logs');
const configSelect = document.getElementById('config-select');
const configMeta = document.getElementById('config-meta');
const configContent = document.getElementById('config-content');
const reloadConfigButton = document.getElementById('reload-config');
const saveConfigButton = document.getElementById('save-config');
const saveRestartConfigButton = document.getElementById('save-restart-config');
const backupLabelInput = document.getElementById('backup-label');
const createBackupButton = document.getElementById('create-backup');
const refreshBackupsButton = document.getElementById('refresh-backups');
const backupStatus = document.getElementById('backup-status');
const backupList = document.getElementById('backup-list');
const backupJobPanel = document.getElementById('backup-job-panel');
const backupJobTitle = document.getElementById('backup-job-title');
const backupJobMessage = document.getElementById('backup-job-message');
const backupJobState = document.getElementById('backup-job-state');
const backupJobProgressBar = document.getElementById('backup-job-progress-bar');
const backupJobDetail = document.getElementById('backup-job-detail');

let backupState = {
    backups: [],
    backup_root: '',
    included_paths: [],
    excluded_directories: [],
    excluded_patterns: [],
    auto_restore_point: true,
};
let activeBackupJob = null;
let activeBackupJobPoll = null;

/* ───── Memory Chart State ───── */
const CHART_COLORS = {
    voxtral:    { line: '#38bdf8', fill: 'rgba(56,189,248,0.15)' },  // blue
    correction: { line: '#22c55e', fill: 'rgba(34,197,94,0.15)' },   // green
    whisperx:   { line: '#f59e0b', fill: 'rgba(245,158,11,0.15)' },  // amber
    vibevoice:  { line: '#a78bfa', fill: 'rgba(167,139,250,0.15)' }, // purple
};
const CHART_MAX_POINTS = 60;  // 60 ticks × 10s = 10 min window
const memoryHistory = {
    voxtral: [],
    correction: [],
    whisperx: [],
    vibevoice: [],
};
const unifiedMemHistory = [];  // total system RAM (includes GPU on UMA)
let totalMemoryGiB = 110;  // practical vLLM limit on 128GB UMA
let chartInitialized = false;

/* ───── Network Chart State ───── */
const NET_CHART_COLORS = {
    send_rate: { line: '#f59e0b', fill: 'rgba(245,158,11,0.25)' },
    recv_rate: { line: '#06b6d4', fill: 'rgba(6,182,212,0.25)' },
};
const CONN_COLORS = {
    voxtral: '#3b82f6',
    correction: '#22c55e',
    whisperx: '#f59e0b',
    vibevoice: '#a78bfa',
};
const netHistory = { send_rate: [], recv_rate: [] };
const connHistory = { voxtral: [], correction: [], whisperx: [], vibevoice: [] };
let networkChartInitialized = false;

/* ───── CPU/GPU Chart State ───── */
const CPU_CORE_COLORS = [
    '#38bdf8', '#22c55e', '#f59e0b', '#ef4444', '#a78bfa', '#ec4899',
    '#14b8a6', '#f97316', '#6366f1', '#84cc16', '#e879f9', '#06b6d4',
    '#facc15', '#fb923c', '#4ade80', '#f43f5e', '#818cf8', '#2dd4bf',
    '#fbbf24', '#a3e635',
];
const GPU_COLOR = { line: '#ef4444', fill: 'rgba(239,68,68,0.20)' };
const cpuHistory = { cores: [], gpu: [] };  // cores: array of arrays per core
let cpuChartInitialized = false;
let cpuCoreCount = 0;

/* ───── Disk Chart State ───── */
const DISK_COLORS = {
    read_rate:  { line: '#06b6d4', fill: 'rgba(6,182,212,0.25)' },
    write_rate: { line: '#f97316', fill: 'rgba(249,115,22,0.25)' },
};
const diskHistory = { read_rate: [], write_rate: [] };
let diskChartInitialized = false;

function showToast(message, isError = false) {
    const node = document.createElement('div');
    node.className = 'toast';
    node.textContent = message;
    node.style.borderColor = isError ? 'rgba(239,68,68,0.5)' : 'rgba(34,197,94,0.4)';
    document.body.appendChild(node);
    setTimeout(() => node.remove(), 2800);
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char] || char));
}

function setBackupActionButtonsDisabled(disabled) {
    if (createBackupButton) {
        createBackupButton.disabled = disabled;
    }
    if (refreshBackupsButton) {
        refreshBackupsButton.disabled = disabled;
    }
    backupList?.querySelectorAll('button').forEach((button) => {
        if (button.dataset.downloadBackup) {
            return;
        }
        button.disabled = disabled;
    });
}

function renderBackupJob(job) {
    activeBackupJob = job || null;
    if (!backupJobPanel || !backupJobTitle || !backupJobMessage || !backupJobState || !backupJobProgressBar || !backupJobDetail) {
        return;
    }
    if (!job) {
        backupJobPanel.classList.add('hidden');
        return;
    }

    backupJobPanel.classList.remove('hidden');
    const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
    backupJobTitle.textContent = job.type === 'restore' ? 'Restore-Job' : 'Backup-Job';
    backupJobMessage.textContent = job.message || 'Job läuft…';
    backupJobState.textContent = job.status || 'running';
    backupJobProgressBar.style.width = `${progress}%`;
    backupJobDetail.innerHTML = `
        <span>Fortschritt: ${progress}%</span>
        <span>Stand: ${escapeHtml(formatDateTime(job.updated_at))}</span>
        ${job.detail ? `<span>${escapeHtml(job.detail)}</span>` : ''}
        ${job.error ? `<span>${escapeHtml(job.error)}</span>` : ''}
    `;

    backupJobState.className = 'badge';
    if (job.status === 'failed') {
        backupJobState.classList.add('badge-danger');
    } else if (job.status === 'completed') {
        backupJobState.classList.add('badge-good');
    }

    const isRunning = job.status === 'queued' || job.status === 'running';
    setBackupActionButtonsDisabled(isRunning);
}

async function pollBackupJob(jobId) {
    if (activeBackupJobPoll) {
        clearInterval(activeBackupJobPoll);
        activeBackupJobPoll = null;
    }

    const refresh = async () => {
        const job = await fetchJson(`/api/backups/jobs/${jobId}`);
        renderBackupJob(job);
        if (job.status === 'completed') {
            if (job.result?.backups) {
                renderBackups({ ...backupState, backups: job.result.backups });
            }
            const extra = job.result?.restore_point ? ` Vorher wurde ${job.result.restore_point.label} angelegt.` : '';
            showToast((job.result?.message || job.message || 'Vorgang abgeschlossen.') + extra);
            if (activeBackupJobPoll) {
                clearInterval(activeBackupJobPoll);
                activeBackupJobPoll = null;
            }
            await Promise.all([loadOverview(), loadConfig(), loadLogs(), loadBackups()]);
            return;
        }
        if (job.status === 'failed') {
            if (activeBackupJobPoll) {
                clearInterval(activeBackupJobPoll);
                activeBackupJobPoll = null;
            }
            showToast(job.error || job.detail || 'Backup-Job fehlgeschlagen.', true);
            await loadBackups();
        }
    };

    await refresh();
    activeBackupJobPoll = window.setInterval(async () => {
        try {
            await refresh();
        } catch (error) {
            if (activeBackupJobPoll) {
                clearInterval(activeBackupJobPoll);
                activeBackupJobPoll = null;
            }
            renderBackupJob({
                id: jobId,
                type: activeBackupJob?.type || 'create',
                status: 'failed',
                progress: activeBackupJob?.progress || 0,
                message: 'Statusabfrage fehlgeschlagen',
                detail: error.message,
                error: error.message,
                updated_at: new Date().toISOString(),
            });
            showToast(error.message, true);
        }
    }, 1200);
}

function formatDateTime(value) {
    if (!value) {
        return 'n/a';
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
        return value;
    }
    return parsed.toLocaleString('de-DE', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

function fitCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.round(rect.width * dpr);
    const h = Math.round(rect.height * dpr);
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, W: rect.width, H: rect.height };
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        ...options,
    });

    if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
            const payload = await response.json();
            detail = payload.detail || detail;
        } catch {
        }
        throw new Error(detail);
    }

    return response.json();
}

function formatBytes(value) {
    if (value === null || value === undefined) {
        return 'n/a';
    }
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let index = 0;
    let current = value;
    while (current >= 1024 && index < units.length - 1) {
        current /= 1024;
        index += 1;
    }
    return `${current.toFixed(current >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return 'n/a';
    }
    return `${Number(value).toFixed(1)}%`;
}

function statusClass(activeState, httpOk) {
    if (httpOk) {
        return 'status-good';
    }
    if (activeState === 'active') {
        return 'status-warn';  // systemd active but HTTP not responding yet
    }
    if (activeState === 'activating' || activeState === 'reloading') {
        return 'status-warn';
    }
    return 'status-bad';
}

function renderSummary(metrics) {
    const memory = metrics.memory || {};
    const swap = metrics.swap || {};
    const uptimeHours = metrics.uptime_seconds ? (metrics.uptime_seconds / 3600).toFixed(1) : 'n/a';

    systemSummary.innerHTML = `
        <div class="kv summary-grid">
            <div><span>Hostname</span><br><strong>${metrics.hostname || 'n/a'}</strong></div>
            <div><span>Load</span><br><strong>${metrics.load_average ? metrics.load_average.join(' / ') : 'n/a'}</strong></div>
            <div><span>CPU</span><br><strong>${formatPercent(metrics.cpu_percent)}</strong></div>
            <div><span>RAM</span><br><strong>${formatBytes(memory.used)} / ${formatBytes(memory.total)}</strong></div>
            <div><span>Swap</span><br><strong>${formatBytes(swap.used)} / ${formatBytes(swap.total)}</strong></div>
            <div><span>Uptime</span><br><strong>${uptimeHours} h</strong></div>
        </div>
    `;
}

function renderGpus(metrics) {
    const gpus = metrics.gpus || [];
    gpuCards.innerHTML = gpus.map((gpu) => {
        const memLine = gpu.unified_memory
            ? `<span>Speicher</span><br><strong>Unified (siehe RAM & Chart)</strong>`
            : `<span>Speicher</span><br><strong>${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB</strong>`;
        return `
        <article class="gpu-card">
            <h3>${gpu.name || 'GPU'}</h3>
            <div class="kv">
                <div><span>Temperatur</span><br><strong>${gpu.temperature_c ?? 'n/a'} °C</strong></div>
                <div><span>Auslastung</span><br><strong>${gpu.utilization_gpu_percent ?? 'n/a'}%</strong></div>
                <div>${memLine}</div>
                <div><span>Power</span><br><strong>${gpu.power_draw_watts ?? 'n/a'} W</strong></div>
            </div>
        </article>`;
    }).join('');
}

/* ───── Memory Rolling Chart ───── */
function initChartLegend() {
    const legend = document.getElementById('memory-chart-legend');
    if (!legend || chartInitialized) return;
    chartInitialized = true;
    const labels = { voxtral: 'Voxtral', correction: 'Correction LLM', whisperx: 'WhisperX', vibevoice: 'ForcedAligner' };
    legend.innerHTML = `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#e5e7eb"></span>Unified (Gesamt-RAM)</span>` +
        Object.entries(CHART_COLORS).map(([key, c]) =>
            `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${c.line}"></span>${labels[key]}</span>`
        ).join('');
}

function bytesToGiB(b) { return b / (1024 ** 3); }

function pushMemorySample(containerMemory, systemMemory) {
    for (const key of Object.keys(memoryHistory)) {
        const bytes = containerMemory?.[key]?.used_bytes ?? 0;
        const arr = memoryHistory[key];
        arr.push(bytesToGiB(bytes));
        if (arr.length > CHART_MAX_POINTS) arr.shift();
    }
    // Track total system RAM usage (includes GPU allocations on UMA)
    const usedBytes = systemMemory?.used ?? 0;
    unifiedMemHistory.push(bytesToGiB(usedBytes));
    if (unifiedMemHistory.length > CHART_MAX_POINTS) unifiedMemHistory.shift();
}

function renderMemoryChart() {
    const canvas = document.getElementById('memory-chart');
    if (!canvas) return;
    const { ctx, W, H } = fitCanvas(canvas);
    ctx.clearRect(0, 0, W, H);

    // Fixed Y scale = total system memory
    const maxVal = totalMemoryGiB;

    const padLeft = 52;
    const padRight = 12;
    const padTop = 8;
    const padBottom = 22;
    const plotW = W - padLeft - padRight;
    const plotH = H - padTop - padBottom;

    // Grid lines
    ctx.strokeStyle = 'rgba(51,65,85,0.5)';
    ctx.lineWidth = 1;
    ctx.font = '11px Inter, sans-serif';
    ctx.fillStyle = '#94a3b8';
    const gridLines = 4;
    for (let i = 0; i <= gridLines; i++) {
        const y = padTop + (plotH / gridLines) * i;
        const val = maxVal - (maxVal / gridLines) * i;
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(padLeft + plotW, y);
        ctx.stroke();
        ctx.fillText(`${val.toFixed(0)} G`, 4, y + 4);
    }

    // Total memory limit line at top
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft + plotW, padTop);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ef4444';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(`Limit: ${maxVal} GiB`, padLeft + plotW - 80, padTop + 12);

    // Time axis labels
    ctx.fillStyle = '#64748b';
    ctx.fillText(`${CHART_MAX_POINTS * 10}s`, padLeft, H - 2);
    ctx.fillText('jetzt', padLeft + plotW - 24, H - 2);

    // Stacked area chart — draw bottom-to-top
    // Stack order: whisperx (bottom), vibevoice, correction, voxtral (top)
    const stackOrder = ['whisperx', 'vibevoice', 'correction', 'voxtral'];
    const numPoints = Math.max(...stackOrder.map(k => memoryHistory[k].length), 2);
    const step = plotW / (CHART_MAX_POINTS - 1);

    // Build cumulative stacks per time point
    const stacks = {};
    let prevCumulative = new Array(numPoints).fill(0);
    for (const key of stackOrder) {
        const arr = memoryHistory[key];
        const padded = new Array(numPoints - arr.length).fill(0).concat(arr);
        const cumulative = padded.map((v, i) => prevCumulative[i] + v);
        stacks[key] = { cumulative, base: [...prevCumulative] };
        prevCumulative = cumulative;
    }

    // Draw stacked areas top-to-bottom (so top layer is drawn first for proper overlap)
    const offset = (CHART_MAX_POINTS - numPoints) * step;
    for (let s = stackOrder.length - 1; s >= 0; s--) {
        const key = stackOrder[s];
        const { cumulative, base } = stacks[key];
        const colors = CHART_COLORS[key];

        // Fill area between cumulative and base
        ctx.beginPath();
        for (let i = 0; i < numPoints; i++) {
            const x = padLeft + offset + i * step;
            const y = padTop + plotH - (cumulative[i] / maxVal) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        // Close along the base line (reversed)
        for (let i = numPoints - 1; i >= 0; i--) {
            const x = padLeft + offset + i * step;
            const y = padTop + plotH - (base[i] / maxVal) * plotH;
            ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fillStyle = colors.fill.replace('0.15', '0.35');
        ctx.fill();

        // Draw top line of this layer
        ctx.beginPath();
        for (let i = 0; i < numPoints; i++) {
            const x = padLeft + offset + i * step;
            const y = padTop + plotH - (cumulative[i] / maxVal) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Unified memory line (total system RAM usage)
    if (unifiedMemHistory.length > 1) {
        const uStep = plotW / (CHART_MAX_POINTS - 1);
        const uOffset = (CHART_MAX_POINTS - unifiedMemHistory.length) * uStep;
        ctx.beginPath();
        for (let i = 0; i < unifiedMemHistory.length; i++) {
            const x = padLeft + uOffset + i * uStep;
            const y = padTop + plotH - (unifiedMemHistory[i] / maxVal) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Current values + total overlay
    const labels = { voxtral: 'Voxtral', correction: 'Correction', whisperx: 'WhisperX', vibevoice: 'ForcedAligner' };
    let labelY = padTop + 16;
    const unifiedCurrent = unifiedMemHistory.length > 0 ? unifiedMemHistory[unifiedMemHistory.length - 1] : 0;
    ctx.fillStyle = '#e5e7eb';
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText(`Unified: ${unifiedCurrent.toFixed(1)} / ${maxVal} GiB (${(unifiedCurrent/maxVal*100).toFixed(0)}%)`, padLeft + 8, labelY);
    labelY += 18;
    let totalCurrent = 0;
    for (const key of stackOrder.slice().reverse()) {
        const arr = memoryHistory[key];
        const val = arr.length > 0 ? arr[arr.length - 1] : 0;
        totalCurrent += val;
        ctx.fillStyle = CHART_COLORS[key].line;
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.fillText(`${labels[key]}: ${val.toFixed(1)} GiB`, padLeft + 8, labelY);
        labelY += 18;
    }
}

/* ───── Network & Connections Rolling Chart ───── */
function initNetworkLegend() {
    const legend = document.getElementById('network-chart-legend');
    if (!legend || networkChartInitialized) return;
    networkChartInitialized = true;
    legend.innerHTML = [
        `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${NET_CHART_COLORS.recv_rate.line}"></span>Empfang</span>`,
        `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${NET_CHART_COLORS.send_rate.line}"></span>Senden</span>`,
        `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${CONN_COLORS.voxtral}"></span>Voxtral Conn</span>`,
        `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${CONN_COLORS.correction}"></span>Correction Conn</span>`,
        `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${CONN_COLORS.whisperx}"></span>WhisperX Conn</span>`,
    ].join('');
}

function formatRate(bytesPerSec) {
    const bits = bytesPerSec * 8;
    if (bits >= 1e9) return `${(bits / 1e9).toFixed(1)} Gbit/s`;
    if (bits >= 1e6) return `${(bits / 1e6).toFixed(1)} Mbit/s`;
    if (bits >= 1e3) return `${(bits / 1e3).toFixed(0)} Kbit/s`;
    return `${bits.toFixed(0)} bit/s`;
}

function pushNetworkSample(network, connections) {
    const sr = network?.send_rate ?? 0;
    const rr = network?.recv_rate ?? 0;
    netHistory.send_rate.push(sr);
    netHistory.recv_rate.push(rr);
    if (netHistory.send_rate.length > CHART_MAX_POINTS) netHistory.send_rate.shift();
    if (netHistory.recv_rate.length > CHART_MAX_POINTS) netHistory.recv_rate.shift();
    for (const key of Object.keys(connHistory)) {
        const val = connections?.[key] ?? 0;
        connHistory[key].push(val);
        if (connHistory[key].length > CHART_MAX_POINTS) connHistory[key].shift();
    }
}

function renderNetworkChart() {
    const canvas = document.getElementById('network-chart');
    if (!canvas) return;
    const { ctx, W, H } = fitCanvas(canvas);
    ctx.clearRect(0, 0, W, H);

    const padLeft = 52, padRight = 12, padTop = 8, padBottom = 22;
    const plotW = W - padLeft - padRight;
    const plotH = H - padTop - padBottom;
    const halfH = plotH * 0.55; // top half for throughput
    const connH = plotH * 0.40; // bottom part for connections
    const connTop = padTop + halfH + plotH * 0.05;

    // ── Throughput (top half) ──
    const allRates = [...netHistory.send_rate, ...netHistory.recv_rate, 1024];
    const maxRate = Math.max(...allRates) * 1.2;
    const step = plotW / (CHART_MAX_POINTS - 1);

    // Grid
    ctx.strokeStyle = 'rgba(51,65,85,0.4)';
    ctx.lineWidth = 0.5;
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = '#94a3b8';
    for (let i = 0; i <= 3; i++) {
        const y = padTop + (halfH / 3) * i;
        const val = maxRate - (maxRate / 3) * i;
        ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(padLeft + plotW, y); ctx.stroke();
        ctx.fillText(formatRate(val), 2, y + 4);
    }

    // Draw rate lines
    for (const [key, colors] of Object.entries(NET_CHART_COLORS)) {
        const arr = netHistory[key];
        if (arr.length < 2) continue;
        const off = (CHART_MAX_POINTS - arr.length) * step;
        // Fill
        ctx.beginPath();
        for (let i = 0; i < arr.length; i++) {
            const x = padLeft + off + i * step;
            const y = padTop + halfH - (arr[i] / maxRate) * halfH;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.lineTo(padLeft + off + (arr.length - 1) * step, padTop + halfH);
        ctx.lineTo(padLeft + off, padTop + halfH);
        ctx.closePath();
        ctx.fillStyle = colors.fill;
        ctx.fill();
        // Line
        ctx.beginPath();
        for (let i = 0; i < arr.length; i++) {
            const x = padLeft + off + i * step;
            const y = padTop + halfH - (arr[i] / maxRate) * halfH;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Labels
    const curSend = netHistory.send_rate.at(-1) ?? 0;
    const curRecv = netHistory.recv_rate.at(-1) ?? 0;
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillStyle = NET_CHART_COLORS.recv_rate.line;
    ctx.fillText(`↓ ${formatRate(curRecv)}`, padLeft + 8, padTop + 14);
    ctx.fillStyle = NET_CHART_COLORS.send_rate.line;
    ctx.fillText(`↑ ${formatRate(curSend)}`, padLeft + 110, padTop + 14);

    // ── Connections (bottom half) — bar-style line chart ──
    const connKeys = ['voxtral', 'correction', 'whisperx', 'vibevoice'];
    const allConns = connKeys.flatMap(k => connHistory[k]);
    const maxConn = Math.max(...allConns, 1) + 1;

    // Separator line
    ctx.strokeStyle = 'rgba(51,65,85,0.6)';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(padLeft, connTop - 2); ctx.lineTo(padLeft + plotW, connTop - 2); ctx.stroke();

    // Grid for connections
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter, sans-serif';
    for (let i = 0; i <= 2; i++) {
        const y = connTop + (connH / 2) * i;
        const val = maxConn - (maxConn / 2) * i;
        ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(padLeft + plotW, y); ctx.stroke();
        ctx.fillText(`${Math.round(val)}`, padLeft - 16, y + 4);
    }
    ctx.fillText('Verb.', 2, connTop + 4);

    // Draw connection lines
    const connLabels = { voxtral: 'Voxtral', correction: 'Correction', whisperx: 'WhisperX', vibevoice: 'ForcedAligner' };
    let connLabelX = padLeft + 8;
    for (const key of connKeys) {
        const arr = connHistory[key];
        if (arr.length < 2) continue;
        const off = (CHART_MAX_POINTS - arr.length) * step;
        ctx.beginPath();
        for (let i = 0; i < arr.length; i++) {
            const x = padLeft + off + i * step;
            const y = connTop + connH - (arr[i] / maxConn) * connH;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = CONN_COLORS[key];
        ctx.lineWidth = 2;
        ctx.stroke();

        const cur = arr.at(-1) ?? 0;
        ctx.fillStyle = CONN_COLORS[key];
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.fillText(`${connLabels[key]}: ${cur}`, connLabelX, connTop + connH + 14);
        connLabelX += 110;
    }

    // Time axis
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(`${CHART_MAX_POINTS * 10}s`, padLeft, H - 2);
    ctx.fillText('jetzt', padLeft + plotW - 24, H - 2);
}

function renderServices(services) {
    for (const service of services) {
        const card = serviceCards.querySelector(`[data-service-key="${service.key}"]`);
        if (!card) {
            continue;
        }
        const stateNode = card.querySelector('[data-field="state"]');
        const httpNode = card.querySelector('[data-field="http"]');
        const stateClass = statusClass(service.status.active_state, service.http.ok);
        const sysState = service.status.active_state;
        const noSystemd = (sysState !== 'active' && sysState !== 'activating');
        const isUp = sysState === 'active' || sysState === 'activating';
        const stateLabel = isUp && !service.http.ok ? 'startet…' : (noSystemd && service.http.ok) ? 'docker/running' : `${sysState}/${service.status.sub_state}`;
        stateNode.textContent = stateLabel;
        stateNode.className = stateClass;
        const isStarting = (sysState === 'active' || sysState === 'activating') && !service.http.ok;
        httpNode.textContent = service.http.ok ? `OK (${service.http.status_code})` : isStarting ? 'Startet…' : (service.http.error || 'Fehler');
        httpNode.className = stateClass;
    }
}

/* ───── CPU/GPU Rolling Chart ───── */
function initCpuChartLegend() {
    if (!cpuChartInitialized) cpuChartInitialized = true;
    const el = document.getElementById('cpu-chart-legend');
    if (!el || cpuCoreCount === 0) return;
    const items = [];
    items.push(`<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${GPU_COLOR.line}"></span>GPU</span>`);
    for (let i = 0; i < cpuCoreCount; i++) {
        const arr = cpuHistory.cores[i];
        const cur = arr ? arr[arr.length - 1] || 0 : 0;
        const active = cur > 0.5;
        const c = active ? CPU_CORE_COLORS[i % CPU_CORE_COLORS.length] : '#555';
        const textStyle = active ? '' : ' style="color:#666"';
        items.push(`<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${c}"></span><span${textStyle}>Core ${i}</span></span>`);
    }
    el.innerHTML = items.join('');
}

function pushCpuSample(metrics) {
    const perCore = metrics.cpu_percent_per_core || [];
    const gpuPct = metrics.gpu_utilization_percent;
    if (cpuCoreCount === 0 && perCore.length > 0) cpuCoreCount = perCore.length;
    // Ensure arrays exist for each core
    while (cpuHistory.cores.length < cpuCoreCount) cpuHistory.cores.push([]);
    for (let i = 0; i < cpuCoreCount; i++) {
        cpuHistory.cores[i].push(perCore[i] ?? 0);
        if (cpuHistory.cores[i].length > CHART_MAX_POINTS) cpuHistory.cores[i].shift();
    }
    cpuHistory.gpu.push(gpuPct ?? 0);
    if (cpuHistory.gpu.length > CHART_MAX_POINTS) cpuHistory.gpu.shift();
}

function renderCpuChart() {
    const canvas = document.getElementById('cpu-chart');
    if (!canvas) return;
    const { ctx, W, H } = fitCanvas(canvas);
    const padLeft = 44, padRight = 10, padTop = 28, padBottom = 18;
    const plotW = W - padLeft - padRight;
    const plotH = H - padTop - padBottom;
    ctx.clearRect(0, 0, W, H);

    // Y-axis: 0-100%
    ctx.font = '11px monospace';
    ctx.fillStyle = '#64748b';
    ctx.strokeStyle = 'rgba(100,116,139,0.15)';
    for (let pct = 0; pct <= 100; pct += 25) {
        const y = padTop + plotH - (pct / 100) * plotH;
        ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(padLeft + plotW, y); ctx.stroke();
        ctx.fillText(`${pct}%`, 4, y + 4);
    }

    const len = cpuHistory.gpu.length;
    if (len < 2) return;
    const dx = plotW / (CHART_MAX_POINTS - 1);
    const xOff = (CHART_MAX_POINTS - len) * dx;

    // Draw GPU as filled area
    ctx.beginPath();
    for (let i = 0; i < len; i++) {
        const x = padLeft + xOff + i * dx;
        const y = padTop + plotH - (cpuHistory.gpu[i] / 100) * plotH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = GPU_COLOR.line; ctx.lineWidth = 2; ctx.stroke();
    // fill
    ctx.lineTo(padLeft + xOff + (len - 1) * dx, padTop + plotH);
    ctx.lineTo(padLeft + xOff, padTop + plotH); ctx.closePath();
    ctx.fillStyle = GPU_COLOR.fill; ctx.fill();

    // Draw per-core CPU lines
    for (let c = 0; c < cpuCoreCount; c++) {
        const arr = cpuHistory.cores[c];
        if (!arr || arr.length < 2) continue;
        ctx.beginPath();
        const color = CPU_CORE_COLORS[c % CPU_CORE_COLORS.length];
        for (let i = 0; i < arr.length; i++) {
            const x = padLeft + xOff + i * dx;
            const y = padTop + plotH - (arr[i] / 100) * plotH;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.7; ctx.stroke();
        ctx.globalAlpha = 1.0;
    }

    // Current values label
    const curGpu = cpuHistory.gpu[len - 1];
    const curCpuAvg = cpuCoreCount > 0
        ? cpuHistory.cores.reduce((s, arr) => s + (arr[arr.length - 1] || 0), 0) / cpuCoreCount
        : 0;
    ctx.font = '12px monospace'; ctx.fillStyle = GPU_COLOR.line;
    ctx.fillText(`GPU ${curGpu.toFixed(0)}%`, padLeft + 8, padTop + 14);
    ctx.fillStyle = '#38bdf8';
    ctx.fillText(`CPU ⌀ ${curCpuAvg.toFixed(0)}%`, padLeft + 100, padTop + 14);

    // X-axis labels
    ctx.fillStyle = '#64748b'; ctx.font = '10px monospace';
    ctx.fillText(`${CHART_MAX_POINTS * 10}s`, padLeft, H - 2);
    ctx.fillText('jetzt', padLeft + plotW - 24, H - 2);
}

/* ───── Disk Rolling Chart ───── */
function initDiskChartLegend() {
    if (diskChartInitialized) return;
    diskChartInitialized = true;
    const el = document.getElementById('disk-chart-legend');
    if (!el) return;
    el.innerHTML = [
        `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${DISK_COLORS.read_rate.line}"></span>Lesen</span>`,
        `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${DISK_COLORS.write_rate.line}"></span>Schreiben</span>`,
    ].join('');
}

function pushDiskSample(diskIo) {
    diskHistory.read_rate.push(diskIo?.read_rate ?? 0);
    diskHistory.write_rate.push(diskIo?.write_rate ?? 0);
    if (diskHistory.read_rate.length > CHART_MAX_POINTS) diskHistory.read_rate.shift();
    if (diskHistory.write_rate.length > CHART_MAX_POINTS) diskHistory.write_rate.shift();
}

function formatDiskRate(bytesPerSec) {
    if (bytesPerSec >= 1024 * 1024 * 1024) return `${(bytesPerSec / (1024**3)).toFixed(1)} GB/s`;
    if (bytesPerSec >= 1024 * 1024) return `${(bytesPerSec / (1024**2)).toFixed(1)} MB/s`;
    if (bytesPerSec >= 1024) return `${(bytesPerSec / 1024).toFixed(0)} KB/s`;
    return `${bytesPerSec.toFixed(0)} B/s`;
}

function renderDiskChart(disks) {
    const canvas = document.getElementById('disk-chart');
    if (!canvas) return;
    const { ctx, W, H } = fitCanvas(canvas);
    const barH = 28;
    const barTop = 6;
    const chartTop = barTop + barH + 20;
    const padLeft = 55, padRight = 10, padBottom = 18;
    const plotW = W - padLeft - padRight;
    const plotH = H - chartTop - padBottom;
    ctx.clearRect(0, 0, W, H);

    // ── Capacity bar (top) ──
    if (disks && disks.length > 0) {
        // Sum all disks or show root
        const root = disks.find(d => d.mountpoint === '/') || disks[0];
        const usedGiB = root.used / (1024**3);
        const totalGiB = root.total / (1024**3);
        const pct = root.percent / 100;
        // Background
        ctx.fillStyle = 'rgba(100,116,139,0.2)';
        ctx.beginPath(); ctx.roundRect(padLeft, barTop, plotW, barH, 4); ctx.fill();
        // Used
        const barColor = pct > 0.9 ? '#ef4444' : pct > 0.75 ? '#f59e0b' : '#22c55e';
        ctx.fillStyle = barColor;
        ctx.beginPath(); ctx.roundRect(padLeft, barTop, plotW * pct, barH, 4); ctx.fill();
        // Label
        ctx.font = '12px monospace'; ctx.fillStyle = '#e2e8f0';
        ctx.fillText(`${root.mountpoint}  ${usedGiB.toFixed(0)} / ${totalGiB.toFixed(0)} GiB (${root.percent.toFixed(0)}%)`, padLeft + 8, barTop + 18);
    }

    // ── I/O rate chart (bottom) ──
    const len = diskHistory.read_rate.length;
    const allVals = [...diskHistory.read_rate, ...diskHistory.write_rate];
    let maxVal = Math.max(...allVals, 1024 * 1024); // min 1 MB/s scale
    // Nice round max
    const steps = [1024*1024, 5*1024*1024, 10*1024*1024, 25*1024*1024, 50*1024*1024, 100*1024*1024, 250*1024*1024, 500*1024*1024, 1024*1024*1024];
    for (const s of steps) { if (s >= maxVal) { maxVal = s; break; } }

    // Y grid
    ctx.font = '11px monospace'; ctx.fillStyle = '#64748b'; ctx.strokeStyle = 'rgba(100,116,139,0.15)';
    for (let i = 0; i <= 4; i++) {
        const val = maxVal * (i / 4);
        const y = chartTop + plotH - (i / 4) * plotH;
        ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(padLeft + plotW, y); ctx.stroke();
        ctx.fillText(formatDiskRate(val), 2, y + 4);
    }

    if (len < 2) return;
    const dx = plotW / (CHART_MAX_POINTS - 1);
    const xOff = (CHART_MAX_POINTS - len) * dx;

    // Draw read & write as filled areas
    for (const key of ['write_rate', 'read_rate']) {
        const arr = diskHistory[key];
        const col = DISK_COLORS[key];
        ctx.beginPath();
        for (let i = 0; i < len; i++) {
            const x = padLeft + xOff + i * dx;
            const y = chartTop + plotH - (arr[i] / maxVal) * plotH;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = col.line; ctx.lineWidth = 2; ctx.stroke();
        ctx.lineTo(padLeft + xOff + (len - 1) * dx, chartTop + plotH);
        ctx.lineTo(padLeft + xOff, chartTop + plotH); ctx.closePath();
        ctx.fillStyle = col.fill; ctx.fill();
    }

    // Current values
    const curRead = diskHistory.read_rate[len - 1];
    const curWrite = diskHistory.write_rate[len - 1];
    ctx.font = '12px monospace';
    ctx.fillStyle = DISK_COLORS.read_rate.line;
    ctx.fillText(`R ${formatDiskRate(curRead)}`, padLeft + 8, chartTop + 14);
    ctx.fillStyle = DISK_COLORS.write_rate.line;
    ctx.fillText(`W ${formatDiskRate(curWrite)}`, padLeft + 120, chartTop + 14);

    // X-axis
    ctx.fillStyle = '#64748b'; ctx.font = '10px monospace';
    ctx.fillText(`${CHART_MAX_POINTS * 10}s`, padLeft, H - 2);
    ctx.fillText('jetzt', padLeft + plotW - 24, H - 2);
}

async function loadOverview() {
    const data = await fetchJson('/api/overview');
    renderSummary(data.metrics);
    renderGpus(data.metrics);
    renderServices(data.services);
    if (data.metrics.memory?.total) {
        totalMemoryGiB = Math.round(data.metrics.memory.total / (1024 ** 3));
    }
    initChartLegend();
    pushMemorySample(data.metrics.container_memory, data.metrics.memory);
    renderMemoryChart();
    initNetworkLegend();
    pushNetworkSample(data.metrics.network, data.metrics.service_connections);
    renderNetworkChart();
    pushCpuSample(data.metrics);
    initCpuChartLegend();
    renderCpuChart();
    initDiskChartLegend();
    pushDiskSample(data.metrics.disk_io);
    renderDiskChart(data.metrics.disks);
}

async function loadLogs() {
    const serviceKey = logServiceSelect.value;
    logsOutput.textContent = 'Lade Logs…';
    const data = await fetchJson(`/api/services/${serviceKey}/logs?lines=200`);
    logsOutput.textContent = data.logs || 'Keine Logs verfügbar.';
}

async function loadConfig() {
    const configId = configSelect.value;
    configContent.value = 'Lade Konfiguration…';
    const data = await fetchJson(`/api/configs/${configId}`);
    configMeta.textContent = `${data.service} • ${data.path}`;
    configContent.value = data.content;
}

function renderBackups(data) {
    backupState = {
        ...backupState,
        ...data,
        backups: data.backups || backupState.backups,
    };

    if (!backupStatus || !backupList) {
        return;
    }

    const backups = backupState.backups || [];
    if (!backups.length) {
        backupStatus.textContent = backupState.backup_root
            ? `Noch keine Backups in ${backupState.backup_root}.`
            : 'Noch keine Backups vorhanden.';
        backupList.innerHTML = '';
        return;
    }

    backupStatus.textContent = `${backups.length} Backup(s) in ${backupState.backup_root || 'System-Archiv'}. Restore legt${backupState.auto_restore_point ? ' automatisch ' : ' optional '}einen Sicherungspunkt an.`;
    backupList.innerHTML = backups.map((backup) => {
        const sourceItems = (backup.sources || []).slice(0, 4).map((source) => `
            <li>${escapeHtml(source)}</li>
        `).join('');
        const extraCount = Math.max((backup.sources || []).length - 4, 0);
        return `
            <article class="backup-card">
                <div class="service-header">
                    <div>
                        <h3>${escapeHtml(backup.label || backup.name)}</h3>
                        <div class="backup-meta">${escapeHtml(formatDateTime(backup.created_at))} • ${escapeHtml(formatBytes(backup.size_bytes))} • ${escapeHtml(backup.reason || 'manual')}</div>
                    </div>
                    <div class="button-row backup-card-actions">
                        <button type="button" data-download-backup="${escapeHtml(backup.name)}">Download</button>
                        <button type="button" class="danger" data-delete-backup="${escapeHtml(backup.name)}">Löschen</button>
                        <button type="button" class="danger" data-restore-backup="${escapeHtml(backup.name)}">Wiederherstellen</button>
                    </div>
                </div>
                <div class="backup-details">
                    <span>${escapeHtml(String(backup.source_count || (backup.sources || []).length))} Quelle(n)</span>
                    <span>Datei: ${escapeHtml(backup.name)}</span>
                </div>
                ${(backup.sources || []).length ? `
                    <div class="backup-sources">
                        <strong>Enthaltene Pfade</strong>
                        <ul>
                            ${sourceItems}
                            ${extraCount ? `<li>… plus ${extraCount} weitere</li>` : ''}
                        </ul>
                    </div>
                ` : ''}
            </article>
        `;
    }).join('');

    if (activeBackupJob) {
        renderBackupJob(activeBackupJob);
    }
}

async function loadBackups() {
    const data = await fetchJson('/api/backups');
    renderBackups(data);
}

async function createBackup() {
    if (backupStatus) {
        backupStatus.textContent = 'Backup wird erstellt…';
    }
    const data = await fetchJson('/api/backups', {
        method: 'POST',
        body: JSON.stringify({ label: backupLabelInput?.value?.trim() || '' }),
    });
    if (backupLabelInput) {
        backupLabelInput.value = '';
    }
    renderBackupJob(data.job);
    await pollBackupJob(data.job.id);
}

async function restoreBackup(name) {
    const backup = (backupState.backups || []).find((item) => item.name === name);
    const label = backup?.label || name;
    const confirmed = window.confirm(`Backup "${label}" wirklich wiederherstellen?\n\nDabei werden die gesicherten Installationspfade zurückkopiert und laufende Dienste kurz gestoppt.`);
    if (!confirmed) {
        return;
    }

    if (backupStatus) {
        backupStatus.textContent = `Restore von ${label} läuft…`;
    }
    const data = await fetchJson('/api/backups/restore', {
        method: 'POST',
        body: JSON.stringify({
            name,
            restart_services: true,
            create_restore_point: backupState.auto_restore_point,
        }),
    });
    renderBackupJob(data.job);
    await pollBackupJob(data.job.id);
}

async function deleteBackup(name) {
    const backup = (backupState.backups || []).find((item) => item.name === name);
    const label = backup?.label || name;
    if (!window.confirm(`Backup "${label}" wirklich löschen?`)) {
        return;
    }
    const data = await fetchJson(`/api/backups/${encodeURIComponent(name)}`, {
        method: 'DELETE',
    });
    renderBackups({ ...backupState, backups: data.backups || [] });
    showToast(data.message || 'Backup gelöscht.');
}

async function saveConfig(restartService) {
    const configId = configSelect.value;
    const body = {
        content: configContent.value,
        restart_service: restartService,
    };
    const data = await fetchJson(`/api/configs/${configId}`, {
        method: 'PUT',
        body: JSON.stringify(body),
    });
    showToast(data.message || 'Konfiguration gespeichert.');
    await loadOverview();
    if (restartService) {
        await loadLogs();
    }
}

async function sendServiceAction(serviceKey, action) {
    const data = await fetchJson(`/api/services/${serviceKey}/action`, {
        method: 'POST',
        body: JSON.stringify({ action }),
    });
    showToast(data.message || `${action} ausgeführt.`);
    await loadOverview();
    if (logServiceSelect.value === serviceKey) {
        await loadLogs();
    }
}

serviceCards?.addEventListener('click', async (event) => {
    const button = event.target.closest('button');
    if (!button) {
        return;
    }
    const card = button.closest('[data-service-key]');
    const serviceKey = card?.dataset.serviceKey;
    if (!serviceKey) {
        return;
    }
    try {
        if (button.dataset.action) {
            await sendServiceAction(serviceKey, button.dataset.action);
            return;
        }
        if (button.hasAttribute('data-open-logs')) {
            logServiceSelect.value = serviceKey;
            await loadLogs();
            logsOutput.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    } catch (error) {
        showToast(error.message, true);
    }
});

refreshLogsButton?.addEventListener('click', async () => {
    try {
        await loadLogs();
    } catch (error) {
        showToast(error.message, true);
    }
});

logServiceSelect?.addEventListener('change', async () => {
    try {
        await loadLogs();
    } catch (error) {
        showToast(error.message, true);
    }
});

reloadConfigButton?.addEventListener('click', async () => {
    try {
        await loadConfig();
    } catch (error) {
        showToast(error.message, true);
    }
});

configSelect?.addEventListener('change', async () => {
    try {
        await loadConfig();
    } catch (error) {
        showToast(error.message, true);
    }
});

saveConfigButton?.addEventListener('click', async () => {
    try {
        await saveConfig(false);
    } catch (error) {
        showToast(error.message, true);
    }
});

saveRestartConfigButton?.addEventListener('click', async () => {
    try {
        await saveConfig(true);
    } catch (error) {
        showToast(error.message, true);
    }
});

createBackupButton?.addEventListener('click', async () => {
    try {
        await createBackup();
    } catch (error) {
        showToast(error.message, true);
    }
});

refreshBackupsButton?.addEventListener('click', async () => {
    try {
        if (backupStatus) {
            backupStatus.textContent = 'Backup-Liste wird geladen…';
        }
        await loadBackups();
    } catch (error) {
        showToast(error.message, true);
    }
});

backupList?.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-restore-backup], [data-delete-backup], [data-download-backup]');
    if (!button) {
        return;
    }
    try {
        if (button.dataset.restoreBackup) {
            await restoreBackup(button.dataset.restoreBackup);
            return;
        }
        if (button.dataset.deleteBackup) {
            await deleteBackup(button.dataset.deleteBackup);
            return;
        }
        if (button.dataset.downloadBackup) {
            window.open(`/api/backups/${encodeURIComponent(button.dataset.downloadBackup)}/download`, '_blank', 'noopener');
        }
    } catch (error) {
        showToast(error.message, true);
    }
});

(async function init() {
    try {
        await Promise.all([loadOverview(), loadConfig(), loadLogs(), loadBackups()]);
        setInterval(async () => {
            try {
                await loadOverview();
            } catch {
            }
        }, 10000);
        initTerminal();
    } catch (error) {
        showToast(error.message, true);
    }
})();

/* ───── WebSocket Terminal (xterm.js) ───── */
function initTerminal() {
    const container = document.getElementById('xterm-container');
    if (!container || typeof Terminal === 'undefined') return;

    // Sync left panel height to right panel
    function syncPanelHeights() {
        const left = document.querySelector('.metrics-panel');
        const right = document.querySelector('.services-panel');
        if (left && right && window.innerWidth >= 1200) {
            left.style.maxHeight = right.offsetHeight + 'px';
        } else if (left) {
            left.style.maxHeight = '';
        }
    }
    syncPanelHeights();
    window.addEventListener('resize', syncPanelHeights);
    // Re-sync periodically as charts change height
    setInterval(syncPanelHeights, 2000);

    const term = new Terminal({
        cursorBlink: true,
        fontSize: 13,
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
        theme: {
            background: '#0f172a',
            foreground: '#e2e8f0',
            cursor: '#38bdf8',
            selectionBackground: 'rgba(56,189,248,0.3)',
            black: '#1e293b', brightBlack: '#475569',
            red: '#ef4444', brightRed: '#f87171',
            green: '#22c55e', brightGreen: '#4ade80',
            yellow: '#f59e0b', brightYellow: '#fbbf24',
            blue: '#3b82f6', brightBlue: '#60a5fa',
            magenta: '#a78bfa', brightMagenta: '#c4b5fd',
            cyan: '#06b6d4', brightCyan: '#22d3ee',
            white: '#e2e8f0', brightWhite: '#f8fafc',
        },
        scrollback: 5000,
        convertEol: true,
    });

    const fitAddon = new FitAddon.FitAddon();
    term.loadAddon(fitAddon);
    if (typeof WebLinksAddon !== 'undefined') {
        term.loadAddon(new WebLinksAddon.WebLinksAddon());
    }

    term.open(container);
    fitAddon.fit();

    let ws = null;

    function connect() {
        if (ws && ws.readyState <= 1) ws.close();
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${location.host}/ws/terminal`);
        ws.onopen = () => {
            term.clear();
            term.writeln('\x1b[32m\u25cf Verbunden\x1b[0m');
            // Send initial resize
            const dims = fitAddon.proposeDimensions();
            if (dims) ws.send('\x01' + JSON.stringify({ rows: dims.rows, cols: dims.cols }));
        };
        ws.onmessage = (ev) => term.write(ev.data);
        ws.onclose = () => term.writeln('\r\n\x1b[31m\u25cf Verbindung getrennt\x1b[0m');
        ws.onerror = () => term.writeln('\r\n\x1b[31m\u25cf WebSocket Fehler\x1b[0m');
    }

    term.onData((data) => {
        if (ws && ws.readyState === 1) ws.send(data);
    });

    // Handle window resize
    const ro = new ResizeObserver(() => {
        fitAddon.fit();
        if (ws && ws.readyState === 1) {
            const dims = fitAddon.proposeDimensions();
            if (dims) ws.send('\x01' + JSON.stringify({ rows: dims.rows, cols: dims.cols }));
        }
    });
    ro.observe(container);

    // Reconnect button
    const btn = document.getElementById('terminal-reconnect');
    if (btn) btn.addEventListener('click', connect);

    connect();
}

/* ───── Spark System Controls ───── */
const sparkRestartBtn = document.getElementById('spark-restart-btn');
const sparkShutdownBtn = document.getElementById('spark-shutdown-btn');

if (sparkRestartBtn) {
    sparkRestartBtn.addEventListener('click', async () => {
        if (!confirm('Möchten Sie das Spark System wirklich neustarten?')) return;
        try {
            const res = await fetch('/api/spark/restart', { method: 'POST' });
            if (res.ok) {
                showToast('Spark System wird neugestartet...');
            } else {
                const data = await res.json().catch(() => ({}));
                showToast(data.detail || 'Fehler beim Neustart', true);
            }
        } catch (e) {
            showToast('Netzwerkfehler beim Neustart', true);
        }
    });
}

if (sparkShutdownBtn) {
    sparkShutdownBtn.addEventListener('click', async () => {
        if (!confirm('Möchten Sie das Spark System wirklich herunterfahren?')) return;
        try {
            const res = await fetch('/api/spark/shutdown', { method: 'POST' });
            if (res.ok) {
                showToast('Spark System wird heruntergefahren...');
            } else {
                const data = await res.json().catch(() => ({}));
                showToast(data.detail || 'Fehler beim Herunterfahren', true);
            }
        } catch (e) {
            showToast('Netzwerkfehler beim Herunterfahren', true);
        }
    });
}
