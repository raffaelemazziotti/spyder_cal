const { app, BrowserWindow, ipcMain, dialog, clipboard, screen } = require('electron');
const fs = require('node:fs/promises');
const { existsSync } = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { createHash } = require('node:crypto');
const { PythonWorker } = require('./worker.cjs');

app.setName('Spyder Cal');
const root = path.resolve(__dirname, '..');
let window, worker, config = {}, busy = false, quitting = false;
let history = [];
const uiURL = pathToFileURL(path.join(__dirname, 'index.html')).href;
const userPath = name => path.join(app.getPath('userData'), name);
const send = message => { if (window && !window.isDestroyed()) window.webContents.send('backend:event', message); };

function defaultPython() {
  const local = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  return process.env.SPYDER_PYTHON || (existsSync(local) ? local : process.platform === 'win32' ? 'python' : 'python3');
}

async function ensureWorker() {
  if (worker?.failed) { await worker.close(); worker = null; }
  if (!worker || worker.exited) {
    const backend = path.join(process.resourcesPath, 'backend', process.platform === 'win32' ? 'spyder-cal-backend.exe' : 'spyder-cal-backend');
    worker = new PythonWorker(app.isPackaged ? backend : config.python || defaultPython(), app.isPackaged ? process.resourcesPath : root, { frozen: app.isPackaged });
    worker.on('message', send);
  }
  return worker.ready;
}

async function operation(callback) {
  if (busy) throw new Error('A measurement or file operation is already running.');
  busy = true;
  try { await ensureWorker(); return await callback(); }
  finally { busy = false; }
}

async function remember(session) {
  history = [session, ...history.filter(item => item.id !== session.id)].slice(0, 100);
  const folder = userPath('sessions');
  await fs.mkdir(folder, { recursive: true });
  const name = createHash('sha256').update(String(session.id)).digest('hex');
  await fs.writeFile(path.join(folder, `${name}.json`), JSON.stringify(session), 'utf8');
}

function handle(channel, callback) {
  ipcMain.handle(channel, async (event, ...args) => {
    if (event.sender !== window.webContents || event.senderFrame.url !== uiURL) throw new Error('Untrusted window');
    try { return await callback(...args); }
    catch (error) { return { __spyderError: error.message || String(error) }; }
  });
}

function installHandlers() {
  handle('app:init', async () => {
    let info = null, error = null;
    try { info = await ensureWorker(); } catch (e) { error = e.message; }
    return {
      info, error, history, python: info?.python || config.python || defaultPython(), packaged: app.isPackaged, version: app.getVersion(),
      displays: screen.getAllDisplays().map((display, i) => ({
        index: i, label: display.label || `Display ${i + 1}`,
        size: display.size, primary: display.id === screen.getPrimaryDisplay().id,
      })),
    };
  });
  handle('python:choose', async () => {
    if (app.isPackaged) throw new Error('This application includes its measurement runtime.');
    if (busy) throw new Error('Finish or cancel the measurement first.');
    const choice = await dialog.showOpenDialog(window, {
      title: 'Select the Python executable in your PsychoPy environment', properties: ['openFile'],
      ...(process.platform === 'win32' ? { filters: [{ name: 'Python', extensions: ['exe'] }] } : {}),
    });
    if (choice.canceled) return null;
    if (worker) await worker.close();
    worker = null;
    config.python = choice.filePaths[0];
    await fs.writeFile(userPath('settings.json'), JSON.stringify(config), 'utf8');
    return { python: config.python, info: await ensureWorker() };
  });
  handle('device:probe', libusb_path => operation(() => worker.request('probe', { libusb_path })));
  handle('display:test', selectedScreen => operation(() => worker.request('test_display', { screen: selectedScreen })));
  handle('session:start', settings => operation(async () => {
    const result = await worker.request('start', { settings }, 0);
    if (result.event === 'result') {
      try { await remember(result.session); }
      catch (error) { send({ event: 'notice', message: `Session is ready to export. Local history could not be saved: ${error.message}` }); }
    }
    return result;
  }));
  handle('session:cancel', () => { worker?.cancel(); });
  handle('session:export', id => operation(async () => {
    const session = history.find(item => item.id === id);
    if (!session) throw new Error('Select a completed session to export.');
    const name = session.settings.monitor_name.replace(/[^a-zA-Z0-9_-]+/g, '-').slice(0, 60) || 'monitor';
    const date = session.created_utc.slice(0, 10);
    const choice = await dialog.showSaveDialog(window, {
      title: 'Export JSON, CSV, PNG and SVG', defaultPath: `${name}-${date}.json`,
      filters: [{ name: 'Calibration session', extensions: ['json'] }],
    });
    if (choice.canceled) return null;
    return worker.request('export', { session, path: choice.filePath });
  }));
  handle('session:open', () => operation(async () => {
    const choice = await dialog.showOpenDialog(window, {
      title: 'Open a Spyder Cal session', properties: ['openFile'], filters: [{ name: 'Calibration session', extensions: ['json'] }],
    });
    if (choice.canceled) return null;
    const filename = choice.filePaths[0];
    if ((await fs.stat(filename)).size > 2_000_000) throw new Error('This session file is too large.');
    const session = JSON.parse(await fs.readFile(filename, 'utf8'));
    if (session.simulated) throw new Error('Select a session measured with the SpyderX.');
    const result = await worker.request('validate', { session });
    await remember(result.session);
    return result.session;
  }));
  handle('value:copy', (id, metric) => {
    const session = history.find(item => item.id === id);
    const values = {
      gamma: session?.psychopy.gamma,
      black: session?.uncorrected.black_luminance,
      white: session?.uncorrected.white_luminance,
      error: session?.corrected?.linearity_rmse_percent,
    };
    if (!Object.hasOwn(values, metric) || !Number.isFinite(values[metric])) throw new Error('No completed value to copy.');
    const value = values[metric].toFixed(6);
    clipboard.writeText(value);
    if (clipboard.readText() !== value) throw new Error(`Clipboard is unavailable. The value is ${value}.`);
  });
}

const primaryInstance = app.requestSingleInstanceLock();
if (!primaryInstance) app.quit();
else {
app.on('second-instance', () => {
  if (window && !window.isDestroyed()) {
    if (window.isMinimized()) window.restore();
    window.show();
    window.focus();
  }
});
app.whenReady().then(async () => {
  await fs.mkdir(app.getPath('userData'), { recursive: true });
  try { config = JSON.parse(await fs.readFile(userPath('settings.json'), 'utf8')); } catch {}
  try {
    const folder = userPath('sessions');
    const names = (await fs.readdir(folder)).filter(name => name.endsWith('.json'));
    const files = await Promise.all(names.map(async name => ({ name, modified: (await fs.stat(path.join(folder, name))).mtimeMs })));
    files.sort((a, b) => b.modified - a.modified);
    for (const { name } of files.slice(0, 100)) {
      try {
        const file = path.join(folder, name);
        if ((await fs.stat(file)).size > 2_000_000) continue;
        const session = JSON.parse(await fs.readFile(file, 'utf8'));
        if (!session.simulated && session.kind === 'spyder_cal.session' && session.uncorrected?.mean?.length && session.psychopy?.gamma > 0) history.push(session);
      } catch {}
    }
    history.sort((a, b) => b.created_utc.localeCompare(a.created_utc));
  } catch {}
  const workArea = screen.getPrimaryDisplay().workAreaSize;
  window = new BrowserWindow({
    title: 'Spyder Cal', width: Math.min(1120, workArea.width), height: Math.min(720, workArea.height), minWidth: 960, minHeight: 600,
    backgroundColor: '#e8e9e8', autoHideMenuBar: true, show: false,
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', event => event.preventDefault());
  window.webContents.session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  installHandlers();
  window.once('ready-to-show', () => window.show());
  await window.loadFile(path.join(__dirname, 'index.html'));
});
}

app.on('window-all-closed', () => app.quit());
app.on('before-quit', event => {
  if (quitting) return;
  event.preventDefault();
  quitting = true;
  (worker ? worker.close() : Promise.resolve()).finally(() => app.quit());
});
