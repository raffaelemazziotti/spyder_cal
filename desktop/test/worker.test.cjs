const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mkdtemp, readFile, rm } = require('node:fs/promises');
const { existsSync } = require('node:fs');
const { execFileSync } = require('node:child_process');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { PythonWorker } = require('../worker.cjs');
const root = path.resolve(__dirname, '../..');
const python = process.env.SPYDER_PYTHON || path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

test('real Python protocol: physical-only sessions, export, import and recovery', { skip: !existsSync(python), timeout: 60000 }, async () => {
  const worker = new PythonWorker(python, root);
  const folder = await mkdtemp(path.join(tmpdir(), 'spyder-app-test-'));
  try {
    assert.ok((await worker.ready).version);
    const fixture = path.join(folder, 'fixture.json');
    execFileSync(python, [path.join(__dirname, 'fixture.py'), fixture]);
    const session = JSON.parse(await readFile(fixture, 'utf8'));
    assert.ok(session.corrected.linearity_rmse_percent < 1);
    const saved = await worker.request('export', { session, path: path.join(folder, 'session.json') });
    assert.equal(saved.paths.length, 4);
    const imported = JSON.parse(await readFile(saved.paths[0], 'utf8'));
    assert.equal((await worker.request('validate', { session: imported })).event, 'validated');
    assert.equal((await readFile(saved.paths[2])).subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
    await assert.rejects(worker.request('start', { simulated: true, settings: {} }), /Only physical/);
    await assert.rejects(worker.request('start', { settings: { pause: -1 } }), /Settling time/);
    assert.equal((await worker.request('validate', { session: imported })).event, 'validated');
  } finally {
    await worker.close();
    assert.equal(path.dirname(folder), tmpdir());
    assert.ok(path.basename(folder).startsWith('spyder-app-test-'));
    await rm(folder, { recursive: true, force: true });
  }
});
