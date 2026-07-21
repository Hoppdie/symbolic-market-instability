/* INSTABILITY dashboard — runs list, live triggering, strip timelines. */

'use strict';

const LEVEL_NAMES = { L: 'Low', M: 'Medium', H: 'High' };
const LEVEL_COLORS = { L: 'var(--low)', M: 'var(--medium)', H: 'var(--high)' };
const POLL_MS = 4000;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const runList = document.getElementById('run-list');
const emptyState = document.getElementById('empty-state');
const template = document.getElementById('run-card-template');
const tooltip = document.getElementById('chart-tooltip');

/* ---------------- Motion: scroll reveals + hero parallax ---------------- */

const revealObserver = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      revealObserver.unobserve(e.target);
    }
  }
}, { threshold: 0.12 });

function observeReveals(root = document) {
  root.querySelectorAll('.reveal:not(.visible)').forEach(el => revealObserver.observe(el));
}

function initParallax() {
  if (reducedMotion) return;
  const layers = document.querySelectorAll('[data-parallax]');
  let ticking = false;
  const apply = () => {
    const y = window.scrollY;
    layers.forEach(el => {
      el.style.transform = `translateY(${y * parseFloat(el.dataset.parallax)}px)`;
    });
    ticking = false;
  };
  window.addEventListener('scroll', () => {
    if (!ticking) { requestAnimationFrame(apply); ticking = true; }
  }, { passive: true });
}

/* ---------------- Data fetching ---------------- */

/* When served from GitHub Pages there is no API — a static export embeds
   all run data as window.STATIC_DATA and the launch form is hidden. */
const STATIC = window.STATIC_DATA || null;

async function fetchRuns() {
  if (STATIC) return STATIC.runs || [];
  const res = await fetch('/api/runs');
  const data = await res.json();
  return data.runs || [];
}

async function fetchRunDetail(runId) {
  if (STATIC) {
    const detail = (STATIC.details || {})[runId];
    if (!detail) throw new Error(`run ${runId}: not in static export`);
    return detail;
  }
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!res.ok) throw new Error(`run ${runId}: HTTP ${res.status}`);
  return res.json();
}

/* ---------------- Stat tiles ---------------- */

function renderStats(runs) {
  const completed = runs.filter(r => r.status === 'completed' && r.summary);
  const latest = completed[0];

  setStat('stat-total-runs', String(runs.length));
  setStat('stat-days', latest ? fmtInt(latest.summary.days_analyzed) : '–');
  setStat('stat-high-days', latest ? fmtInt(latest.summary.high_days) : '–');

  let bestLead = null;
  for (const r of completed) {
    for (const key of ['best_lead_time_days']) {
      if (r.summary[key] != null) bestLead = Math.max(bestLead ?? 0, r.summary[key]);
    }
  }
  setStat('stat-lead', bestLead != null ? `${bestLead}d` : '–');
}

function setStat(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function fmtInt(n) { return n == null ? '–' : Number(n).toLocaleString('en-US'); }

/* ---------------- Run cards ---------------- */

function renderRuns(runs) {
  runList.querySelectorAll('.run-card').forEach(el => el.remove());
  emptyState.hidden = runs.length > 0;

  for (const run of runs) {
    runList.appendChild(buildRunCard(run));
  }
  observeReveals(runList);
}

function buildRunCard(run) {
  const card = template.content.firstElementChild.cloneNode(true);
  card.dataset.runId = run.run_id;

  card.querySelector('.run-id').textContent = run.run_id;
  const p = run.params || {};
  card.querySelector('.run-params').textContent =
    `${p.ticker || '?'} · ${p.start_date || '?'} → ${p.end_date || '?'}`;

  updateCardStatus(card, run);

  const head = card.querySelector('.run-card-head');
  head.addEventListener('click', () => toggleDetail(card, head));
  return card;
}

function updateCardStatus(card, run) {
  const badge = card.querySelector('.status-badge');
  badge.textContent = (run.status || 'unknown').toUpperCase();
  badge.className = `status-badge mono ${run.status || ''}`;

  const counts = card.querySelector('.run-counts');
  counts.textContent = '';
  if (run.status === 'completed' && run.summary) {
    for (const [code, key] of [['L', 'low_days'], ['M', 'medium_days'], ['H', 'high_days']]) {
      const chip = document.createElement('span');
      chip.className = 'count-chip';
      const dot = document.createElement('span');
      dot.className = `dot dot--${LEVEL_NAMES[code].toLowerCase()}`;
      chip.appendChild(dot);
      chip.appendChild(document.createTextNode(
        `${LEVEL_NAMES[code]} ${fmtInt(run.summary[key])}`));
      counts.appendChild(chip);
    }
  } else if (run.status === 'failed' && run.error) {
    const err = document.createElement('span');
    err.className = 'count-chip';
    err.style.color = 'var(--high)';
    err.textContent = run.error.slice(0, 90);
    counts.appendChild(err);
  } else if (run.status === 'running') {
    const note = document.createElement('span');
    note.className = 'count-chip';
    note.textContent = 'pipeline in flight…';
    counts.appendChild(note);
  }
}

/* ---------------- Detail view ---------------- */

async function toggleDetail(card, head) {
  const detail = card.querySelector('.run-detail');
  const open = head.getAttribute('aria-expanded') === 'true';
  if (open) {
    head.setAttribute('aria-expanded', 'false');
    detail.hidden = true;
    return;
  }
  head.setAttribute('aria-expanded', 'true');
  detail.hidden = false;

  if (!card.dataset.loaded) {
    try {
      const data = await fetchRunDetail(card.dataset.runId);
      card.dataset.loaded = '1';
      renderDetail(card, data);
      renderStripPreview(card, data.timeline);
    } catch (err) {
      detail.querySelector('.detail-timeline').textContent = String(err);
    }
  }
}

function renderStripPreview(card, timeline) {
  if (!timeline || !timeline.levels) return;
  const strip = card.querySelector('.run-strip-preview');
  strip.textContent = '';
  for (const seg of compressLevels(timeline.levels)) {
    const i = document.createElement('i');
    i.className = `seg--${seg.level}`;
    i.style.flex = String(seg.len);
    strip.appendChild(i);
  }
}

function renderDetail(card, data) {
  const chartHost = card.querySelector('.timeline-chart');
  const axis = card.querySelector('.timeline-axis');
  const timeline = data.timeline;

  if (timeline && timeline.levels) {
    chartHost.appendChild(buildTimelineSVG(timeline));
    for (const label of [timeline.start_date, midDate(timeline), timeline.end_date]) {
      const span = document.createElement('span');
      span.textContent = label;
      axis.appendChild(span);
    }
  } else {
    chartHost.textContent = 'No timeline data for this run.';
  }

  renderMetrics(card.querySelector('.detail-metrics'), data.metrics);
  initTableToggle(card, data);
}

function compressLevels(levels) {
  const segments = [];
  let cur = null;
  for (const ch of levels) {
    if (cur && cur.level === ch) cur.len += 1;
    else segments.push(cur = { level: ch, len: 1 });
  }
  return segments;
}

function buildTimelineSVG(timeline) {
  const W = 1000, H = 84;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label',
    `Instability timeline from ${timeline.start_date} to ${timeline.end_date}`);

  const total = timeline.levels.length;
  let x = 0;
  for (const seg of compressLevels(timeline.levels)) {
    const w = (seg.len / total) * W;
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x.toFixed(2));
    rect.setAttribute('y', '0');
    // hairline gap between segments wide enough to afford one
    rect.setAttribute('width', Math.max(w - (w > 6 ? 1 : 0), 0.5).toFixed(2));
    rect.setAttribute('height', String(H));
    rect.setAttribute('fill', LEVEL_COLORS[seg.level] || 'var(--hairline)');
    svg.appendChild(rect);
    x += w;
  }

  attachTimelineTooltip(svg, timeline);
  return svg;
}

function midDate(timeline) {
  const a = Date.parse(timeline.start_date), b = Date.parse(timeline.end_date);
  return isoDay(new Date((a + b) / 2));
}

function isoDay(d) { return d.toISOString().slice(0, 10); }

function attachTimelineTooltip(svg, timeline) {
  const total = timeline.levels.length;
  const a = Date.parse(timeline.start_date);
  const b = Date.parse(timeline.end_date);

  svg.addEventListener('mousemove', (ev) => {
    const rect = svg.getBoundingClientRect();
    const frac = Math.min(Math.max((ev.clientX - rect.left) / rect.width, 0), 1);
    const idx = Math.min(Math.floor(frac * total), total - 1);
    const level = LEVEL_NAMES[timeline.levels[idx]] || '?';
    const approx = isoDay(new Date(a + frac * (b - a))).slice(0, 7);

    tooltip.textContent = `≈ ${approx} · ${level.toUpperCase()}`;
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(ev.clientX + 14, window.innerWidth - 160)}px`;
    tooltip.style.top = `${ev.clientY - 40}px`;
  });
  svg.addEventListener('mouseleave', () => { tooltip.hidden = true; });
}

function renderMetrics(host, metrics) {
  host.textContent = '';
  if (!metrics) {
    const note = document.createElement('p');
    note.className = 'mono';
    note.style.color = 'var(--ink-muted)';
    note.textContent = 'No evaluation metrics for this run.';
    host.appendChild(note);
    return;
  }
  const panels = Object.entries(metrics)
    .map(([key, m]) => [m?.name || key.replace(/_/g, ' ').toUpperCase(), m]);
  if (!panels.length) {
    const note = document.createElement('p');
    note.className = 'mono';
    note.style.color = 'var(--ink-muted)';
    note.textContent = 'No crash events fall inside this run’s date range.';
    host.appendChild(note);
    return;
  }
  for (const [title, m] of panels) {
    if (!m) continue;
    const panel = document.createElement('div');
    panel.className = 'metric-panel';
    const h = document.createElement('h4');
    h.textContent = title;
    panel.appendChild(h);

    if (m.note) {
      const p = document.createElement('p');
      p.textContent = m.note;
      p.style.color = 'var(--ink-muted)';
      panel.appendChild(p);
    } else {
      addMetricRow(panel, 'Lead time', m.lead_time_days != null ? `${m.lead_time_days} days` : '–');
      addMetricRow(panel, 'Precision', fmtPct(m.precision));
      addMetricRow(panel, 'Recall', fmtPct(m.recall));
      addMetricRow(panel, 'Warnings before crash', fmtInt(m.warnings_before_crash));
    }
    host.appendChild(panel);
  }
}

function addMetricRow(panel, label, value) {
  const row = document.createElement('div');
  row.className = 'metric-row';
  const span = document.createElement('span');
  span.textContent = label;
  const b = document.createElement('b');
  b.textContent = value;
  row.append(span, b);
  panel.appendChild(row);
}

function fmtPct(v) { return v == null ? '–' : `${(v * 100).toFixed(1)}%`; }

/* Accessible alternative to the color strip: plain numbers. */
function initTableToggle(card, data) {
  const btn = card.querySelector('.toggle-table');
  const host = card.querySelector('.detail-table');
  btn.addEventListener('click', () => {
    if (!host.dataset.built) {
      host.appendChild(buildSummaryTable(data));
      host.dataset.built = '1';
    }
    host.hidden = !host.hidden;
    btn.textContent = host.hidden ? 'Table view' : 'Hide table';
  });
}

function buildSummaryTable(data) {
  const rows = [];
  const s = data.summary || {};
  rows.push(['Days analyzed', fmtInt(s.days_analyzed)]);
  rows.push(['Low days', fmtInt(s.low_days)]);
  rows.push(['Medium days', fmtInt(s.medium_days)]);
  rows.push(['High days', fmtInt(s.high_days)]);
  rows.push(['Duration', s.duration_seconds != null ? `${s.duration_seconds}s` : '–']);
  for (const [key, m] of Object.entries(data.metrics || {})) {
    if (m && m.lead_time_days != null) {
      rows.push([`${m.name || key} — lead time`, `${m.lead_time_days} days`]);
    }
  }

  const table = document.createElement('table');
  const thead = table.createTHead().insertRow();
  for (const t of ['Measure', 'Value']) {
    const th = document.createElement('th');
    th.textContent = t;
    thead.appendChild(th);
  }
  const tbody = table.createTBody();
  for (const [k, v] of rows) {
    const tr = tbody.insertRow();
    tr.insertCell().textContent = k;
    tr.insertCell().textContent = v;
  }
  return table;
}

/* ---------------- Launch form + polling ---------------- */

function initForm() {
  const form = document.getElementById('run-form');
  const status = document.getElementById('form-status');
  const btn = document.getElementById('launch-btn');

  if (STATIC) {
    // Read-only snapshot: no server to launch runs against.
    const launch = document.getElementById('launch');
    if (launch) launch.hidden = true;
    return;
  }

  const tickerSelect = document.getElementById('f-ticker');
  const customField = document.getElementById('custom-ticker-field');
  const customInput = document.getElementById('f-ticker-custom');
  tickerSelect.addEventListener('change', () => {
    const custom = tickerSelect.value === '__custom__';
    customField.hidden = !custom;
    customInput.required = custom;
    if (custom) customInput.focus();
  });
  const selectedTicker = () =>
    tickerSelect.value === '__custom__'
      ? customInput.value.trim().toUpperCase()
      : tickerSelect.value;

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    status.className = 'form-status mono';
    status.textContent = 'LAUNCHING…';
    btn.disabled = true;

    try {
      const payload = {
        ticker: selectedTicker(),
        start_date: form.start_date.value,
        end_date: form.end_date.value,
      };
      const res = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (res.status === 202) {
        status.classList.add('ok');
        status.textContent = `RUN ${data.run_id} IN FLIGHT`;
        await refresh();
        pollRun(data.run_id);
      } else if (res.status === 409) {
        status.classList.add('error');
        status.textContent = `BUSY — RUN ${data.run_id || ''} STILL IN FLIGHT`;
      } else {
        status.classList.add('error');
        status.textContent = (data.errors || [data.error || 'launch failed']).join(' · ').toUpperCase();
      }
    } catch (err) {
      status.classList.add('error');
      status.textContent = String(err).toUpperCase();
    } finally {
      btn.disabled = false;
    }
  });
}

const activePolls = new Set();

function pollRun(runId) {
  if (activePolls.has(runId)) return;
  activePolls.add(runId);
  const timer = setInterval(async () => {
    try {
      const run = await fetchRunDetail(runId);
      const card = runList.querySelector(`[data-run-id="${CSS.escape(runId)}"]`);
      if (card) updateCardStatus(card, run);
      if (run.status !== 'running') {
        clearInterval(timer);
        activePolls.delete(runId);
        const status = document.getElementById('form-status');
        status.className = `form-status mono ${run.status === 'completed' ? 'ok' : 'error'}`;
        status.textContent = `RUN ${runId} ${run.status.toUpperCase()}`;
        await refresh();
      }
    } catch { /* transient poll failure — keep trying */ }
  }, POLL_MS);
}

/* ---------------- Boot ---------------- */

async function refresh() {
  const runs = await fetchRuns();
  renderStats(runs);
  renderRuns(runs);
  runs.filter(r => r.status === 'running').forEach(r => pollRun(r.run_id));
  return runs;
}

initParallax();
observeReveals();
initForm();
refresh().catch(err => {
  emptyState.hidden = false;
  emptyState.querySelector('.empty-hint').textContent = `Could not load runs: ${err}`;
});
