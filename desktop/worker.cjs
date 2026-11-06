const { spawn } = require('node:child_process');
const { EventEmitter } = require('node:events');
const { createInterface } = require('node:readline');
const { randomUUID } = require('node:crypto');

class PythonWorker extends EventEmitter {
  constructor(python, root, { frozen = false } = {}) {
    super();
    this.pending = null;
    this.cancelTimer = null;
    this.logs = '';
    this.ready = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    this.startupTimer = setTimeout(() => this.stop('Python took too long to start. Check the selected environment.'), 45000);
    this.child = spawn(python, frozen ? [] : ['-u', `${root}/desktop_backend.py`], {
      cwd: root, windowsHide: true, shell: false,
      env: { ...process.env, PYTHONUTF8: '1', PYTHONUNBUFFERED: '1', MPLBACKEND: 'Agg' },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.child.stderr.setEncoding('utf8');
    this.child.stderr.on('data', text => { this.logs = (this.logs + text).slice(-5000); });
    this.child.stdin.on('error', () => {});
    this.child.on('error', error => this.fail(`Could not start Python: ${error.message}`));
    this.child.on('exit', code => {
      this.exited = true;
      this.fail(`Python stopped (${code ?? 'terminated'}). ${this.logs.slice(-1600)}`);
    });
    createInterface({ input: this.child.stdout }).on('line', line => {
      let message;
      try { message = JSON.parse(line); } catch { return; }
      if (!message || typeof message.event !== 'string') return;
      if (message.event === 'ready') {
        clearTimeout(this.startupTimer);
        this.info = message;
        this.resolveReady(message);
      }
      if (message.event === 'fatal') this.fail(message.message);
      if (this.pending && message.requestId === this.pending.id) {
        this.watchProgress(message);
        const { resolve, reject, timer } = this.pending;
        if (['result', 'device', 'display_tested', 'exported', 'validated', 'cancelled', 'error'].includes(message.event)) {
          clearTimeout(timer);
          clearTimeout(this.cancelTimer);
          this.pending = null;
          if (message.event === 'error') reject(new Error(message.message));
          else resolve(message);
        }
      }
      this.emit('message', message);
    });
  }

  fail(message) {
    this.failed = true;
    clearTimeout(this.startupTimer);
    clearTimeout(this.cancelTimer);
    this.rejectReady(new Error(message));
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.reject(new Error(message));
      this.pending = null;
    }
    this.emit('message', { event: 'offline', message });
  }

  async request(action, data = {}, timeout = 60000) {
    await this.ready;
    if (this.exited) throw new Error('Python is offline. Reconnect using Python settings.');
    if (this.pending) throw new Error('Wait for the current operation to finish.');
    return new Promise((resolve, reject) => {
      const id = randomUUID();
      const timer = timeout ? setTimeout(() => this.stop('Python did not respond in time.'), timeout) : null;
      this.pending = { id, action, stage: 'starting Python operation', resolve, reject, timer };
      if (action === 'start' || action === 'test_display') this.watchProgress({ event: 'started' });
      this.control(action, { ...data, requestId: id });
    });
  }

  control(action, data = {}) {
    if (!this.child.stdin.destroyed && !this.exited) {
      this.child.stdin.write(`${JSON.stringify({ action, ...data })}\n`);
    }
  }

  watchProgress(message) {
    const pending = this.pending;
    if (!pending || !['start', 'test_display'].includes(pending.action)) return;
    if (!['started', 'phase', 'sample', 'summary', 'prompt', 'prompt_done'].includes(message.event)) return;
    clearTimeout(pending.timer);
    if (message.event === 'prompt') return;
    if (message.event === 'phase') pending.stage = message.phase;
    if (message.event === 'sample') pending.stage = 'reading luminance';
    pending.timer = setTimeout(() => this.stop(`The operation stopped responding while ${pending.stage.replaceAll('_', ' ')}. The Python worker was stopped. Check the connection and try again.`), 30000);
  }

  cancel() {
    this.control('cancel');
    clearTimeout(this.cancelTimer);
    if (this.pending) this.cancelTimer = setTimeout(() => this.stop('Cancelled the unresponsive operation. You can reconnect and try again.'), 4000);
  }

  terminate() {
    if (!this.child.pid || this.exited) return;
    if (process.platform === 'win32') {
      const killer = spawn('taskkill.exe', ['/PID', String(this.child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
      killer.on('error', () => this.child.kill());
    } else this.child.kill();
  }

  stop(message = 'Python worker closed.') {
    this.fail(message);
    this.terminate();
  }

  async close() {
    if (this.exited) return;
    this.control('quit');
    await new Promise(resolve => {
      const timer = setTimeout(() => { this.terminate(); resolve(); }, 4000);
      this.child.once('exit', () => { clearTimeout(timer); resolve(); });
    });
  }
}

module.exports = { PythonWorker };
