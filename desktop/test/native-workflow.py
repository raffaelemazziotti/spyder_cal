"""Real PsychoPy window, test-only sensor fixture; no USB or calibration."""
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from desktop_backend import Backend, PatchWindow


backend = Backend(io.StringIO())
drive = 0
real_show = PatchWindow.show
def show(window, value, status=None):
    global drive
    drive = value
    real_show(window, value, status)

class SensorFixture:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def calibrate(self): pass
    def get_luminance(self): return .2 + 160 * drive ** 2.2

original_emit = backend.emit
def emit(event, **data):
    original_emit(event, **data)
    if event == 'prompt':
        assert backend.presenter is not None
        backend.commands.put({'action': 'continue', 'prompt': data['prompt']})
backend.emit = emit
with patch('cal_lib.SpyderX', SensorFixture), patch.object(PatchWindow, 'show', show):
    backend.run_session({'settings': {'num_levels': 5, 'repetitions': 1, 'pause': .1}})
events = [json.loads(line) for line in backend.output.getvalue().splitlines()]
assert [e['prompt'] for e in events if e['event'] == 'prompt'] == ['dark', 'position', 'complete']
assert events[-1]['event'] == 'result'
assert events[-1]['session']['presentation']['instruction_band_fraction'] == .1
assert backend.presenter is None
print('Native fullscreen workflow passed: setup, two sweeps, completion and cleanup.')
