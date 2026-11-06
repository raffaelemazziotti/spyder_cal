import io
import json
import subprocess
import sys
import types
import unittest
from unittest import mock

from desktop_backend import Backend, Cancelled


class ProtocolInputTests(unittest.TestCase):
    def test_ready_is_available_before_loading_measurement_libraries(self):
        script = '''
import io, json, sys
from desktop_backend import Backend
output = io.StringIO()
Backend(output, io.StringIO('{"action":"quit"}\\n')).serve()
assert json.loads(output.getvalue())['event'] == 'ready'
assert not {'numpy', 'scipy', 'matplotlib', 'psychopy'}.intersection(sys.modules)
'''
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10, check=True)

    def test_library_subprocess_gets_eof_without_consuming_json_commands(self):
        script = '''
import json, subprocess, sys
from desktop_backend import isolate_protocol_input
protocol = isolate_protocol_input()
inherited = subprocess.check_output(
    [sys.executable, '-c', 'import sys; print(sys.stdin.read())'],
    text=True, timeout=5,
)
print(json.dumps({'inherited': inherited.strip(), 'protocol': protocol.readline().strip()}))
'''
        completed = subprocess.run(
            [sys.executable, "-c", script], input='{"action":"probe"}\n',
            capture_output=True, text=True, timeout=15, check=True,
        )
        self.assertEqual(json.loads(completed.stdout), {
            "inherited": "", "protocol": '{"action":"probe"}',
        })


class HardwareSessionCleanupTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.backend = Backend(self.output)
        self.backend.wait_for_prompt = mock.Mock()
        self.backend.pause = mock.Mock()
        self.context = mock.MagicMock()
        self.device = self.context.__enter__.return_value
        self.presenter = mock.Mock()
        self.presenter.event.getKeys.return_value = []
        self.drive = 0
        def show(value, status=None):
            self.drive = value
        self.presenter.show.side_effect = show
        self.device.get_luminance.side_effect = lambda: 0.2 + 100 * self.drive ** 2.2
        self.psychopy = types.SimpleNamespace(visual=mock.Mock(), event=mock.Mock())
        self.command = {"settings": {"num_levels": 5, "repetitions": 1}}

    def run_hardware(self, presenter_error=None):
        with mock.patch.dict("sys.modules", {"psychopy": self.psychopy}), mock.patch(
            "cal_lib.SpyderX", return_value=self.context
        ), mock.patch("desktop_backend.PatchWindow", return_value=self.presenter, side_effect=presenter_error):
            self.backend.run_session(self.command)

    def test_real_workflow_waits_for_setup_and_releases_sensor_and_window(self):
        self.run_hardware()
        self.assertEqual([call.args[0] for call in self.backend.wait_for_prompt.call_args_list], ["dark", "position", "complete"])
        self.device.calibrate.assert_called_once_with()
        self.assertEqual(self.device.get_luminance.call_count, 10)
        self.presenter.close.assert_called_once_with()
        self.context.__exit__.assert_called_once()
        messages = [json.loads(line) for line in self.output.getvalue().splitlines()]
        self.assertEqual(messages[-1]["event"], "result")
        self.assertFalse(messages[-1]["session"]["simulated"])

    def test_cancel_during_measurement_releases_both_resources(self):
        self.backend.pause.side_effect = Cancelled()
        with self.assertRaises(Cancelled):
            self.run_hardware()
        self.presenter.close.assert_called_once_with()
        self.context.__exit__.assert_called_once()
        self.assertIsNone(self.backend.presenter)
        self.assertNotIn('"event": "result"', self.output.getvalue())

    def test_window_failure_releases_sensor_without_fake_results(self):
        with self.assertRaisesRegex(RuntimeError, "display unavailable"):
            self.run_hardware(presenter_error=RuntimeError("display unavailable"))
        self.context.__enter__.assert_not_called()
        self.device.get_luminance.assert_not_called()
        self.assertNotIn('"event": "result"', self.output.getvalue())

    def test_cancel_at_cover_prompt_closes_fullscreen_and_usb(self):
        self.backend.wait_for_prompt.side_effect = Cancelled()
        with self.assertRaises(Cancelled):
            self.run_hardware()
        self.presenter.close.assert_called_once()
        self.context.__exit__.assert_called_once()
        self.device.calibrate.assert_not_called()


class NativePromptTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend(io.StringIO())
        self.presenter = mock.Mock()
        self.backend.presenter = self.presenter

    def test_each_setup_prompt_accepts_native_space_and_enter(self):
        for prompt, key in [("dark", "space"), ("position", "return")]:
            self.presenter.event.getKeys.return_value = [key]
            self.backend.wait_for_prompt(prompt, "Native instruction")
        self.assertEqual(self.presenter.instructions.call_count, 2)

    def test_escape_at_completion_preserves_the_finished_result(self):
        self.presenter.event.getKeys.return_value = ["escape"]
        self.backend.wait_for_prompt("complete", "Complete", completed=True)
        self.assertIn('"event": "prompt_done"', self.backend.output.getvalue())

    def test_escape_during_setup_cancels(self):
        self.presenter.event.getKeys.return_value = ["escape"]
        with self.assertRaises(Cancelled):
            self.backend.wait_for_prompt("dark", "Close cover")


if __name__ == "__main__":
    unittest.main()
