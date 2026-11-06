/* Sandboxed renderer: only the fixed preload API is exposed. */
const $ = id => document.getElementById(id);
const api = window.spyder;
const state = { history: [], current: null, live: [], summaries: {}, busy: false, connected: false, normalized: false, info: null, prompt: null };
const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const date = value => new Date(value).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
let noticeTimer;

function notice(message, error = false) {
  clearTimeout(noticeTimer);
  $('notice-text').textContent = message;
  $('notice').classList.toggle('error', error);
  $('notice').hidden = false;
  if (!error) noticeTimer = setTimeout(() => { $('notice').hidden = true; }, 7000);
}

function loading(title, detail, cancel = false) {
  $('loading-title').textContent = title;
  $('loading-detail').textContent = detail;
  $('loading-cancel').hidden = !cancel;
  $('loading-cancel').disabled = false;
  $('loading-overlay').hidden = false;
}
function hideLoading() { $('loading-overlay').hidden = true; }
function activity(title, detail = '') {
  $('progress-title').textContent = title;
  $('progress-detail').textContent = detail;
}
function idle() {
  activity(state.current ? 'Measurement complete' : 'Ready to measure', state.current ? 'Copy values from the cards or save files.' : 'Instructions appear in the PsychoPy window.');
}
function settings() {
  return {
    monitor_name: $('monitor-name').value.trim(), screen: Number($('screen').value),
    num_levels: Number($('levels').value), repetitions: Number($('repetitions').value),
    pause: Number($('pause').value), verify: $('verify').checked, libusb_path: $('libusb-path').value.trim(),
  };
}
function setBusy(value) {
  state.busy = value;
  for (const id of ['start', 'check-device', 'test-screen', 'open-session', 'recent-sessions', 'new-session', 'python-settings', 'choose-python', 'monitor-name', 'screen', 'levels', 'repetitions', 'pause', 'verify', 'libusb-path']) $(id).disabled = value;
  $('open-menu').inert = value;
  if (value) $('open-menu').open = false;
  $('activity-spinner').hidden = !value;
  $('export').disabled = value || !state.current;
  for (const metric of ['gamma', 'black', 'white', 'error']) {
    $(`copy-${metric}`).disabled = value || !state.current || (metric === 'error' && !state.current.corrected);
  }
  for (const item of $('history').querySelectorAll('button')) item.disabled = value;
}
function setDevice(connected, text, failed = false) {
  state.connected = connected;
  $('device-text').textContent = text;
  $('device-status').title = text;
  $('device-status').classList.toggle('connected', connected);
  $('device-status').classList.toggle('failed', failed);
  $('check-device').textContent = connected ? 'Reconnect' : 'Connect';
}
function renderHistory() {
  $('history-count').textContent = state.history.length;
  const items = state.history.map(session => {
    const button = document.createElement('button');
    button.className = `history-item${state.current?.id === session.id ? ' active' : ''}`;
    button.disabled = state.busy;
    const name = document.createElement('strong');
    name.textContent = session.settings.monitor_name;
    const meta = document.createElement('small');
    const timestamp = document.createElement('span'); timestamp.textContent = date(session.created_utc);
    const gamma = document.createElement('span'); gamma.textContent = `γ ${fmt(session.psychopy.gamma)}`;
    meta.append(timestamp, gamma); button.append(name, meta);
    button.addEventListener('click', () => { showSession(session); $('history-dialog').close(); });
    return button;
  });
  if (!items.length) {
    const empty = document.createElement('p'); empty.className = 'history-empty';
    empty.textContent = 'Completed measurements will appear here.'; items.push(empty);
  }
  $('history').replaceChildren(...items);
}
function addSession(session) {
  state.history = [session, ...state.history.filter(item => item.id !== session.id)].slice(0, 100);
  showSession(session);
}
function showSession(session) {
  state.current = session; state.live = [];
  state.summaries = { uncorrected: session.uncorrected, corrected: session.corrected };
  for (const [key, id] of [['monitor_name', 'monitor-name'], ['num_levels', 'levels'], ['repetitions', 'repetitions'], ['pause', 'pause'], ['libusb_path', 'libusb-path']]) $(id).value = session.settings[key] ?? '';
  if (![...$('screen').options].some(option => Number(option.value) === session.settings.screen)) $('screen').add(new Option(`Screen ${session.settings.screen}`, session.settings.screen));
  $('screen').value = session.settings.screen; $('verify').checked = session.settings.verify;
  $('session-date').textContent = date(session.created_utc);
  $('chart-subtitle').textContent = `${session.settings.monitor_name} · ${session.settings.num_levels} levels · ${session.settings.repetitions} repetitions`;
  $('chart-caption').textContent = session.corrected ? 'Correction verified · Shading ±1 SD' : 'Correction not measured · Shading ±1 SD';
  updateMetrics(); renderHistory(); drawChart(); updateEstimate(); setBusy(state.busy);
  if (!state.busy) idle();
}
function updateMetrics() {
  const baseline = state.summaries.uncorrected, corrected = state.summaries.corrected;
  $('gamma-value').textContent = fmt(baseline?.gamma);
  $('gamma-detail').textContent = baseline ? `± ${fmt(baseline.gamma_std)} SD · ${baseline.runs.length} repetitions` : 'Original measured exponent';
  $('black-value').textContent = fmt(baseline?.black_luminance);
  $('white-value').textContent = fmt(baseline?.white_luminance, 2);
  $('error-value').textContent = fmt(corrected?.linearity_rmse_percent, 2);
  $('error-detail').textContent = corrected ? `Corrected gamma ${fmt(corrected.gamma)} · target 1` : 'RMSE of luminance range';
}
function updateEstimate() {
  const s = settings(), readings = s.num_levels * s.repetitions * (s.verify ? 2 : 1);
  const seconds = readings * (s.pause + 0.15);
  $('estimate').textContent = `${readings} readings · ≈ ${seconds < 60 ? `${Math.ceil(seconds)} s` : `${Math.ceil(seconds / 60)} min`} + setup`;
}

const NS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs = {}, content) {
  const element = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (content !== undefined) element.textContent = content;
  return element;
}

function liveRuns(phase) {
  const runs = new Map();
  for (const sample of state.live.filter(point => point.phase === phase)) {
    if (!runs.has(sample.repetition)) runs.set(sample.repetition, { levels: [], luminance: [] });
    const run = runs.get(sample.repetition);
    run.levels.push(sample.level);
    run.luminance.push(sample.luminance);
  }
  return [...runs.values()];
}

function drawChart() {
  const original = state.summaries.uncorrected;
  const corrected = state.summaries.corrected;
  const phases = [
    { summary: original, runs: original?.runs || liveRuns('uncorrected'), color: '#9383ba', id: 'original' },
    { summary: corrected, runs: corrected?.runs || liveRuns('corrected'), color: '#639d89', id: 'corrected' },
  ];
  const readings = phases.flatMap(p => p.runs.flatMap(run => run.luminance));
  const black = original?.black_luminance ?? 0;
  const white = original?.white_luminance ?? (Math.max(...readings, 160));
  const range = white - black || 1;
  const convert = value => state.normalized ? (value - black) / range : value;
  const values = readings.map(convert);
  const min = Math.min(0, ...values);
  const max = Math.max(1, ...values, ...(!readings.length ? [state.normalized ? 1 : 160] : [])) * 1.07;
  const textAttrs = { fill: '#6f7b72', 'font-size': 9, 'font-family': 'Segoe UI, sans-serif' };
  for (const phase of phases) {
    const chart = $(`chart-${phase.id}`);
    const width = Math.max(200, chart.clientWidth), height = Math.max(130, chart.clientHeight);
    chart.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const left = 48, right = width - 14, top = 9, bottom = height - 34;
    const x = value => left + value * (right - left);
    const y = value => bottom - (convert(value) - min) / (max - min) * (bottom - top);
    chart.replaceChildren();
    $(`empty-${phase.id}`).hidden = phase.runs.some(run => run.luminance.length > 0);
    for (let i = 0; i <= 4; i++) {
      const v = min + (max - min) / 1.07 * i / 4;
      const py = bottom - (v - min) / (max - min) * (bottom - top);
      chart.append(svg('line', { x1: left, y1: py, x2: right, y2: py, stroke: '#dce0d9', 'stroke-width': 1, 'stroke-dasharray': '3 5' }));
      chart.append(svg('text', { ...textAttrs, x: left - 12, y: py + 4, 'text-anchor': 'end' }, state.normalized ? `${Math.round(v * 100)}%` : fmt(v, max < 10 ? 1 : 0)));
    }
    for (let i = 0; i <= 5; i++) {
      const px = x(i / 5);
      chart.append(svg('text', { ...textAttrs, x: px, y: bottom + 17, 'text-anchor': 'middle' }, (i / 5).toFixed(1)));
    }
    chart.append(svg('text', { ...textAttrs, x: (left + right) / 2, y: height - 3, 'text-anchor': 'middle' }, 'Requested intensity'));
    const middle = (top + bottom) / 2;
    chart.append(svg('text', { ...textAttrs, x: 11, y: middle, transform: `rotate(-90 11 ${middle})`, 'text-anchor': 'middle' }, state.normalized ? 'Normalized luminance' : 'Luminance (cd/m²)'));
    const pointsPath = points => points.map((p, i) => `${i ? 'L' : 'M'}${x(p.level).toFixed(2)},${y(p.luminance).toFixed(2)}`).join(' ');
    if (original) {
      chart.append(svg('path', { d: pointsPath([{ level: 0, luminance: black }, { level: 1, luminance: white }]), fill: 'none', stroke: '#78869b', 'stroke-width': 1.4, 'stroke-dasharray': '6 6' }));
      if (phase.id === 'original') {
      const model = Array.from({ length: 101 }, (_, i) => ({ level: i / 100, luminance: black + range * (i / 100) ** original.gamma }));
      chart.append(svg('path', { d: pointsPath(model), fill: 'none', stroke: '#7871aa', 'stroke-width': 1.5, 'stroke-dasharray': '2 5' }));
    }
    }
    if ($('show-runs').checked) {
      for (const run of phase.runs) {
        chart.append(svg('path', { d: pointsPath(run.levels.map((level, i) => ({ level, luminance: run.luminance[i] }))), fill: 'none', stroke: phase.color, 'stroke-width': 1, opacity: .23 }));
      }
    }
    const mean = phase.summary?.mean || (phase.runs.at(-1)?.levels || []).map((level, i) => ({ level, luminance: phase.runs.at(-1).luminance[i], sd: 0 }));
    if (!mean.length) continue;
    const upper = mean.map(p => ({ level: p.level, luminance: p.luminance + p.sd }));
    const lower = [...mean].reverse().map(p => ({ level: p.level, luminance: p.luminance - p.sd }));
    chart.append(svg('path', { d: `${pointsPath(upper)} ${pointsPath(lower).replace(/^M/, 'L')} Z`, fill: phase.color, opacity: .09 }));
    chart.append(svg('path', { d: pointsPath(mean), fill: 'none', stroke: phase.color, 'stroke-width': 2.2, 'stroke-linejoin': 'round' }));
    for (const point of mean) {
      const marker = svg('circle', { cx: x(point.level), cy: y(point.luminance), r: 3.1, fill: '#f6f6f4', stroke: phase.color, 'stroke-width': 1.6 });
      marker.append(svg('title', {}, `Intensity ${point.level.toFixed(3)}: ${point.luminance.toFixed(3)} cd/m² (SD ${point.sd.toFixed(3)})`));
      chart.append(marker);
    }
  }
}



function pythonInfo(info) {
  state.info = info;
  $('python-path').textContent = info?.python || $('python-path').textContent;
  $('python-info').textContent = info ? `Python ${info.version} · ${info.system} · ${info.psychopy ? 'PsychoPy available' : 'PsychoPy missing'}` : 'Runtime offline. Choose a Python environment with the required packages.';
}
function clearMeasurement() {
  state.current = null; state.live = []; state.summaries = {};
  updateMetrics(); drawChart(); renderHistory(); setBusy(false);
  $('session-date').textContent = ''; $('chart-subtitle').textContent = 'Current measurement';
  $('chart-caption').textContent = 'Shared axes · Shading ±1 SD';
  $('notice').hidden = true; idle();
}
async function run() {
  if (state.busy || !$('measurement-form').reportValidity()) return;
  if (!state.info?.psychopy) { notice('PsychoPy is unavailable in the measurement runtime.', true); $('settings-dialog').showModal(); return; }
  clearMeasurement(); setBusy(true);
  $('progress').value = 0; $('progress').hidden = false; $('cancel').hidden = false;
  activity('Preparing measurement window…');
  loading('Opening measurement window', `Preparing PsychoPy on screen ${settings().screen}…`, true);
  $('chart-subtitle').textContent = 'Collecting sensor readings…';
  try { await api.start(settings()); }
  catch (error) { notice(error.message, true); }
  finally { hideLoading(); setBusy(false); $('cancel').hidden = true; $('progress').hidden = true; idle(); }
}
api.onEvent(message => {
  if (message.event === 'ready') pythonInfo(message);
  else if (message.event === 'offline') { pythonInfo(null); if (!state.busy) notice(message.message, true); }
  else if (message.event === 'device') setDevice(true, 'Connected');
  else if (message.event === 'prompt') {
    hideLoading(); state.prompt = message.prompt;
    const prompts = { dark: 'Cover the sensor', position: 'Position the sensor', complete: 'Measurement complete', error: 'Measurement needs attention' };
    activity(prompts[message.prompt] || message.prompt, 'Continue in the PsychoPy window.');
    if (message.prompt === 'dark') setDevice(true, 'Connected');
  } else if (message.event === 'prompt_done') state.prompt = null;
  else if (message.event === 'phase') {
    const names = { loading_measurement_tools: 'Loading measurement tools…', loading_psychopy: 'Loading PsychoPy…', connecting_sensor: 'Connecting to SpyderX…', calibrating_sensor: 'Dark calibration…', opening_display: `Opening screen ${settings().screen}…`, display_test_visible: 'Display test open', uncorrected: 'Measuring original response', corrected: 'Verifying corrected response', dark_complete: 'Dark calibration complete' };
    activity(`${names[message.phase] || message.phase}${message.repetition ? ` · run ${message.repetition}/${settings().repetitions}` : ''}`);
    if (['loading_measurement_tools', 'loading_psychopy', 'opening_display'].includes(message.phase)) $('loading-detail').textContent = names[message.phase];
    if (['connecting_sensor', 'display_test_visible'].includes(message.phase)) hideLoading();
    if (message.phase === 'connecting_sensor') setDevice(false, 'Connecting…');
    if (message.phase === 'display_test_visible') $('progress-detail').textContent = 'Closes in 5 seconds · Esc to close sooner';
  } else if (message.event === 'sample') {
    state.live.push(message); $('progress').value = message.completed / message.total * 100;
    $('progress-detail').textContent = `${message.completed}/${message.total} · ${fmt(message.luminance, 2)} cd/m²`; drawChart();
  } else if (message.event === 'summary') {
    state.summaries[message.phase] = message.summary; updateMetrics(); drawChart();
  } else if (message.event === 'result') addSession(message.session);
  else if (message.event === 'cancelled') notice('Operation cancelled.');
  else if (message.event === 'error' || message.event === 'fatal') {
    hideLoading(); if (message.action === 'probe') setDevice(false, 'Connection failed', true); notice(message.message, true);
  } else if (message.event === 'notice') notice(message.message);
});
$('measurement-form').addEventListener('submit', event => { event.preventDefault(); run(); });
function cancel() { api.cancel(); activity('Cancelling…'); $('loading-detail').textContent = 'Cancelling…'; $('loading-cancel').disabled = true; }
$('cancel').addEventListener('click', cancel); $('loading-cancel').addEventListener('click', cancel);
$('dismiss-notice').addEventListener('click', () => { $('notice').hidden = true; });
$('python-settings').addEventListener('click', () => $('settings-dialog').showModal());
$('settings-close').addEventListener('click', () => $('settings-dialog').close());
$('choose-python').addEventListener('click', async () => {
  $('choose-python').disabled = true;
  try { const result = await api.choosePython(); if (result) { $('python-path').textContent = result.python; pythonInfo(result.info); $('notice').hidden = true; } }
  catch (error) { pythonInfo(null); notice(error.message, true); }
  finally { $('choose-python').disabled = false; }
});
$('check-device').addEventListener('click', async () => {
  setBusy(true); activity('Connecting to SpyderX…'); setDevice(false, 'Checking…'); $('check-device').textContent = 'Connecting…';
  try { await api.probe($('libusb-path').value.trim()); }
  catch (error) { setDevice(false, 'Connection failed', true); notice(error.message, true); }
  finally { $('check-device').textContent = state.connected ? 'Reconnect' : 'Connect'; setBusy(false); idle(); }
});
$('test-screen').addEventListener('click', async () => {
  setBusy(true); $('notice').hidden = true; $('cancel').hidden = false;
  activity('Opening display test…'); loading('Opening display test', `Preparing PsychoPy on screen ${settings().screen}…`, true);
  try { const result = await api.testDisplay(Number($('screen').value)); if (result.event === 'display_tested') notice(`Screen ${result.screen} opened successfully.`); }
  catch (error) { notice(error.message, true); }
  finally { hideLoading(); setBusy(false); $('cancel').hidden = true; idle(); }
});
const metricLabels = { gamma: 'gamma', black: 'black luminance', white: 'white luminance', error: 'corrected error' };
for (const metric of Object.keys(metricLabels)) $(`copy-${metric}`).addEventListener('click', async () => {
  try { await api.copyValue(state.current.id, metric); notice(`Copied ${metricLabels[metric]}.`); }
  catch (error) { notice(error.message, true); }
});
$('export').addEventListener('click', async () => {
  setBusy(true); activity('Saving files…');
  try { const result = await api.export(state.current.id); if (result) notice(`Saved JSON, CSV, PNG and SVG. ${result.paths[0]}`); }
  catch (error) { notice(`Export failed: ${error.message}`, true); }
  finally { setBusy(false); idle(); }
});
$('open-session').addEventListener('click', async () => {
  $('open-menu').open = false; setBusy(true); activity('Opening measurement…');
  try { const session = await api.open(); if (session) addSession(session); }
  catch (error) { notice(error.message, true); }
  finally { setBusy(false); idle(); }
});
$('recent-sessions').addEventListener('click', () => { $('open-menu').open = false; renderHistory(); $('history-dialog').showModal(); });
$('history-close').addEventListener('click', () => $('history-dialog').close());
$('new-session').addEventListener('click', clearMeasurement);
document.addEventListener('pointerdown', event => { if (!$('open-menu').contains(event.target)) $('open-menu').open = false; });
$('absolute').addEventListener('click', () => { state.normalized = false; $('absolute').classList.add('active'); $('normalized').classList.remove('active'); drawChart(); });
$('normalized').addEventListener('click', () => { state.normalized = true; $('normalized').classList.add('active'); $('absolute').classList.remove('active'); drawChart(); });
$('show-runs').addEventListener('change', drawChart);
for (const id of ['levels', 'repetitions', 'pause', 'verify']) $(id).addEventListener('input', updateEstimate);
async function init() {
  setBusy(true); drawChart(); updateEstimate();
  try {
    const data = await api.init(); $('app-version').textContent = `v${data.version}`;
    $('python-path').textContent = data.python; pythonInfo(data.info);
    $('screen').replaceChildren(...data.displays.map(display => new Option(`Screen ${display.index}`, display.index)));
    state.history = data.history; renderHistory();
    if (data.error) notice(data.error, true);
    if (data.packaged) { $('choose-python').hidden = true; $('runtime-help').textContent = 'Python and PsychoPy are included in this application.'; }
  } catch (error) { notice(error.message, true); }
  finally { hideLoading(); setBusy(false); idle(); }
}
init();
const chartResize = new ResizeObserver(drawChart);
for (const id of ['chart-original', 'chart-corrected']) chartResize.observe($(id).parentElement);
