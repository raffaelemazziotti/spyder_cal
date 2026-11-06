// Run with npm run test:ui. Fixtures and dialog destinations are test-only.
const { app, dialog, clipboard, ipcMain } = require('electron');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const { mkdtempSync, mkdirSync, symlinkSync } = require('node:fs');
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const { tmpdir } = require('node:os');
const folder = mkdtempSync(path.join(tmpdir(), 'spyder-ui-test-'));
app.setPath('userData', path.join(folder, 'profile'));
app.getVersion = () => require('../package.json').version;
const root = path.resolve(__dirname, '../..');
const python = process.env.SPYDER_PYTHON || path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
if (process.env.SPYDER_TEST_PACKAGED === '1') {
  const resources = path.join(folder, 'resources'); mkdirSync(resources);
  symlinkSync(path.join(root, 'desktop/build/spyder-cal-backend'), path.join(resources, 'backend'), process.platform === 'win32' ? 'junction' : 'dir');
  Object.defineProperty(app, 'isPackaged', { get: () => true });
  Object.defineProperty(process, 'resourcesPath', { value: resources });
}
mkdirSync(path.join(folder, 'profile/sessions'), { recursive: true });
execFileSync(python, [path.join(__dirname, 'fixture.py'), path.join(folder, 'profile/sessions/fixture.json')]);
const exportPath = path.join(folder, 'calibration.json');
let copiedValue = '';
clipboard.writeText = value => { copiedValue = value; };
clipboard.readText = () => copiedValue;
dialog.showSaveDialog = async () => ({ canceled: false, filePath: exportPath });
dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [exportPath] });
// Keep startup pending briefly to verify its visible loading state reliably.
const handle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, callback) => handle(channel, async (...args) => {
  if (channel === 'app:init') await new Promise(resolve => setTimeout(resolve, 600));
  return callback(...args);
});
const watchdog = setTimeout(() => { console.error('UI smoke test timed out'); app.exit(1); }, 120000);
async function waitFor(window, expression) {
  return window.webContents.executeJavaScript(`new Promise((resolve,reject)=>{
    const end=Date.now()+45000; const tick=()=>{ if(${expression})return resolve(true); if(Date.now()>end)return reject(new Error('Timed out: '+${JSON.stringify(expression)})); setTimeout(tick,40); }; tick();
  })`);
}
app.on('browser-window-created', (_event, window) => {
  let displayVisible = false;
  const send = window.webContents.send.bind(window.webContents);
  window.webContents.send = (channel, message, ...rest) => {
    if (channel === 'backend:event' && message.event === 'phase' && message.phase === 'display_test_visible') displayVisible = true;
    return send(channel, message, ...rest);
  };
  if (process.env.SPYDER_UI_DEBUG) {
    window.webContents.on('console-message', event => console.log('Renderer:', event.message));
    window.webContents.on('render-process-gone', (_event, details) => console.log('Renderer stopped:', details));
  }
  window.webContents.once('did-finish-load', async () => {
    const execute = expression => window.webContents.executeJavaScript(expression);
    const settle = () => execute('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
    try {
      assert.ok(await execute("!document.getElementById('loading-overlay').hidden"), 'Startup indicator is visible while initialization is pending');
      await waitFor(window, "!document.getElementById('start').disabled && document.getElementById('loading-overlay').hidden");
      assert.equal(await execute("document.getElementById('gamma-value').textContent"), '—', 'Start with a fresh measurement');
      assert.ok(await execute("document.getElementById('export').disabled && !document.querySelector('.sidebar') && !document.querySelector('.results-footer')"));
      await execute("document.getElementById('open-menu').open=true; document.getElementById('recent-sessions').click(); document.querySelector('.history-item').click()");
      await waitFor(window, "document.getElementById('gamma-value').textContent !== '—'");
      const gamma = Number(await execute("document.getElementById('gamma-value').textContent"));
      assert.ok(gamma > 2.1 && gamma < 2.3);
      assert.ok(await execute("document.querySelectorAll('#chart-original circle').length===15 && document.querySelectorAll('#chart-corrected circle').length===15"));
      assert.ok(await execute("!document.getElementById('history-dialog').open && !document.getElementById('demo')"));
      for (const metric of ['gamma','black','white','error']) {
        await execute(`document.getElementById('copy-${metric}').click()`);
        await waitFor(window, `document.getElementById('notice-text').textContent.startsWith('Copied ${metric==='black'?'black luminance':metric==='white'?'white luminance':metric==='error'?'corrected error':'gamma'}')`);
        assert.ok(Number.isFinite(Number(copiedValue)));
        if (metric === 'gamma') assert.ok(Math.abs(Number(copiedValue)-2.2)<.001);
        if (metric === 'black') assert.ok(Math.abs(Number(copiedValue)-.2)<.001);
        if (metric === 'white') assert.ok(Math.abs(Number(copiedValue)-160.2)<.001);
      }
      await execute("document.getElementById('normalized').click()");
      assert.ok(await execute("document.getElementById('normalized').classList.contains('active')"));
      await execute("document.getElementById('export').click()");
      await waitFor(window, "document.getElementById('notice-text').textContent.startsWith('Saved JSON')");
      const saved = JSON.parse(await fs.readFile(exportPath, 'utf8'));
      assert.equal(saved.kind,'spyder_cal.session'); assert.equal(saved.corrected.runs.length,3);
      await execute("document.getElementById('new-session').click(); document.getElementById('open-menu').open=true; document.getElementById('open-session').click()");
      await waitFor(window, "!document.getElementById('start').disabled && document.getElementById('gamma-value').textContent !== '—'");
      assert.equal(await execute("document.getElementById('history-count').textContent"),'1');
      for (const [width,height] of [[1120,680],[960,560]]) {
        window.setContentSize(width,height); await settle();
        if (process.env.SPYDER_UI_DEBUG) {
          console.log(await execute("JSON.stringify({inner:[innerWidth,innerHeight],parts:Object.fromEntries(['.control-bar','.activity','main','.setup','.metrics','.plots','.plots-toolbar','.chart-panel','.chart-container'].map(s=>[s,document.querySelector(s).getBoundingClientRect().height]))})"));
          await fs.writeFile(path.join(__dirname,'../.cache/compact-debug.png'),(await window.webContents.capturePage()).toPNG());
        }
        assert.ok(await execute("document.documentElement.scrollWidth<=window.innerWidth && document.documentElement.scrollHeight<=window.innerHeight"),`No page scrolling at ${width}x${height}`);
        assert.ok(await execute("document.getElementById('chart-original').getBoundingClientRect().right<document.getElementById('chart-corrected').getBoundingClientRect().left"));
        assert.ok(await execute("document.getElementById('chart-original').getBoundingClientRect().height>=150"),'Graphs remain visible');
        for (const id of ['start','check-device','export','copy-gamma']) assert.ok(await execute(`document.getElementById('${id}').getBoundingClientRect().bottom<=window.innerHeight`));
      }
      window.setContentSize(1120,680); await settle();
      await execute("document.getElementById('absolute').click(); document.getElementById('notice').hidden=true");
      await fs.mkdir(path.join(__dirname,'../.cache'),{recursive:true});
      await fs.writeFile(path.join(__dirname,'../.cache/app-preview.png'),(await window.webContents.capturePage()).toPNG());
      await execute("document.getElementById('test-screen').click()");
      assert.ok(await execute("!document.getElementById('loading-overlay').hidden && !document.getElementById('loading-cancel').hidden"),'Display startup shows progress and cancellation');
      await fs.writeFile(path.join(__dirname,'../.cache/loading-preview.png'),(await window.webContents.capturePage()).toPNG());
      await waitFor(window,"!document.getElementById('start').disabled && document.getElementById('loading-overlay').hidden");
      assert.ok(displayVisible, 'PsychoPy confirmed the real fullscreen window was shown');
      assert.ok(await execute("/opened successfully|Operation cancelled/.test(document.getElementById('notice-text').textContent)"), await execute("document.getElementById('notice-text').textContent"));
      console.log(JSON.stringify({passed:true,gamma,exportPath,checks:['startup indicator','fresh measurement','optional history','four card copy actions','two graphs','no scrolling at 1120x680 and 960x560','normalization','four exports','import','display loading indicator and real PsychoPy window']}));
      clearTimeout(watchdog); app.quit();
    } catch(error) { console.error(error); clearTimeout(watchdog); app.exit(1); }
  });
});
require('../main.cjs');
