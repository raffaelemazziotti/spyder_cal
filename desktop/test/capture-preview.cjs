// Render an illustrative session with the real UI, without USB or native windows.
const { app, BrowserWindow, ipcMain } = require('electron');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const root = path.resolve(__dirname, '../..');
const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'spyder-preview-'));
app.setPath('userData', path.join(folder, 'profile'));
app.commandLine.appendSwitch('disable-gpu');
const python = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const fixture = path.join(folder, 'session.json');
execFileSync(python, [path.join(__dirname, 'fixture.py'), fixture]);
const session = JSON.parse(fs.readFileSync(fixture, 'utf8'));
session.settings.monitor_name = 'Example display';
ipcMain.handle('app:init', () => ({
  history: [session], packaged: true, version: require('../package.json').version,
  python: 'Included measurement runtime', info: { python: 'Included measurement runtime', version: '3.10', system: 'Windows', psychopy: true },
  displays: [{ index: 0 }],
}));
const watchdog = setTimeout(() => { console.error('Preview timed out'); app.exit(1); }, 30000);
app.whenReady().then(async () => {
  const window = new BrowserWindow({
    width: 1120, height: 680, useContentSize: true, show: false,
    webPreferences: { offscreen: true, preload: path.join(root, 'desktop/preload.cjs'), sandbox: true, contextIsolation: true, nodeIntegration: false },
  });
  try {
    await window.loadFile(path.join(root, 'desktop/index.html'));
    await window.webContents.executeJavaScript(`new Promise(resolve => {
      const wait = () => { if (document.getElementById('start').disabled) return setTimeout(wait, 25); resolve(); }; wait();
    })`);
    await window.webContents.executeJavaScript("document.getElementById('recent-sessions').click(); document.querySelector('.history-item').click()");
    window.webContents.send('backend:event', { event: 'device', connected: true });
    await window.webContents.executeJavaScript('document.fonts.ready.then(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))');
    const output = path.join(root, 'docs/assets/spyder-cal.png');
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(output, (await window.webContents.capturePage()).toPNG());
    console.log(output);
    clearTimeout(watchdog); app.quit();
  } catch (error) { console.error(error); clearTimeout(watchdog); app.exit(1); }
});
