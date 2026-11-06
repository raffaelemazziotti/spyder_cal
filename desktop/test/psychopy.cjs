// Explicit native display test: opens screen 0 for five seconds, without USB.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { PythonWorker } = require('../worker.cjs');
const root = path.resolve(__dirname, '../..');
const python = process.env.SPYDER_PYTHON || path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

test('real PsychoPy imports and opens a display from the piped worker', { timeout: 45000 }, async () => {
  const worker = new PythonWorker(python, root);
  const stages = [];
  worker.on('message', message => { if (message.event === 'phase') stages.push(message.phase); });
  try {
    assert.equal((await worker.ready).psychopy, true, 'Install PsychoPy in the selected Python environment');
    assert.equal((await worker.request('test_display', { screen: 0 })).event, 'display_tested');
    assert.deepEqual(stages, ['loading_psychopy', 'opening_display', 'display_test_visible']);
  } finally { await worker.close(); }
});
