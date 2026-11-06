import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from calibration_session import SessionSettings, export_session, measure_session, validate_session


def synthetic_session(verify=True, emit=lambda *_args, **_kwargs: None):
    settings = SessionSettings(num_levels=9, repetitions=2, verify=verify)
    return measure_session(
        settings, lambda drive, phase, repetition: 0.25 + 150 * drive ** 2.2,
        emit, lambda: None, simulated=True,
    )


class CalibrationSessionTests(unittest.TestCase):
    def test_inverse_gamma_is_verified_by_independent_readings(self):
        calls = []
        def read(drive, phase, repetition):
            calls.append((drive, phase, repetition))
            return 0.25 + 150 * drive ** 2.2
        session = measure_session(SessionSettings(num_levels=9, repetitions=2), read, lambda *_a, **_k: None, lambda: None)
        self.assertEqual(len(calls), 36)
        self.assertAlmostEqual(session["psychopy"]["gamma"], 2.2, places=5)
        self.assertAlmostEqual(session["correction"]["exponent"], 1 / 2.2, places=5)
        self.assertAlmostEqual(session["corrected"]["gamma"], 1, delta=0.001)
        self.assertLess(session["corrected"]["linearity_rmse_percent"], 0.0001)
        self.assertGreater(session["uncorrected"]["linearity_rmse_percent"], 10)
        corrected = session["corrected"]["runs"][0]
        self.assertGreater(corrected["drive_values"][4], corrected["levels"][4])
        self.assertEqual(session["correction"]["lut"][0], 0)
        self.assertEqual(session["correction"]["lut"][-1], 1)

    def test_each_repetition_and_sample_is_preserved(self):
        events = []
        session = synthetic_session(emit=lambda name, **data: events.append((name, data)))
        samples = [data for name, data in events if name == "sample"]
        self.assertEqual([p["completed"] for p in samples], list(range(1, 37)))
        self.assertEqual(samples[-1]["total"], 36)
        self.assertEqual(len(session["uncorrected"]["runs"]), 2)
        self.assertEqual(len(session["corrected"]["mean"]), 9)

    def test_original_only_session_has_no_fake_corrected_measurements(self):
        session = synthetic_session(verify=False)
        self.assertIsNone(session["corrected"])
        validate_session(session)

    def test_invalid_readings_abort_session(self):
        for reading in [float("nan"), float("inf"), 10.0]:
            with self.subTest(reading=reading), self.assertRaises(ValueError):
                measure_session(SessionSettings(num_levels=5, repetitions=1), lambda *_args: reading, lambda *_a, **_k: None, lambda: None)

    def test_cancel_stops_before_another_sensor_read(self):
        readings = []
        def cancel():
            if len(readings) == 2:
                raise InterruptedError("cancelled")
        with self.assertRaises(InterruptedError):
            measure_session(
                SessionSettings(num_levels=5),
                lambda *_args: readings.append(1) or 1,
                lambda *_a, **_k: None, cancel,
            )
        self.assertEqual(len(readings), 2)

    def test_invalid_settings_are_rejected(self):
        for values in [
            {"num_levels": 3}, {"num_levels": 5.5}, {"screen": -1},
            {"repetitions": 0}, {"pause": math.nan}, {"pause": -1},
            {"verify": "false"}, {"monitor_name": " "},
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                SessionSettings(**values)

    def test_export_contains_raw_corrected_data_and_real_plot_files(self):
        session = synthetic_session()
        with tempfile.TemporaryDirectory() as directory:
            paths = export_session(session, Path(directory) / "monitor.json")
            self.assertEqual(len(paths), 4)
            saved = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
            self.assertEqual(saved, session)
            with Path(paths[1]).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 36)
            self.assertEqual({row["phase"] for row in rows}, {"uncorrected", "corrected"})
            self.assertTrue(all(row["simulated"] == "True" for row in rows))
            self.assertEqual(Path(paths[2]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertIn("TEST DATA", Path(paths[3]).read_text(encoding="utf-8"))
            self.assertIn("axes_2", Path(paths[3]).read_text(encoding="utf-8"))

    def test_foreign_incomplete_and_nonfinite_files_are_rejected(self):
        session = synthetic_session()
        invalid = [
            {}, {**session, "kind": "other"}, {**session, "simulated": None},
            {**session, "uncorrected": None},
        ]
        bad_sample = copy.deepcopy(session)
        bad_sample["uncorrected"]["runs"][0]["luminance"][0] = math.nan
        invalid.append(bad_sample)
        for candidate in invalid:
            with self.subTest(candidate=candidate.get("kind")), self.assertRaises((ValueError, KeyError)):
                validate_session(candidate)


if __name__ == "__main__":
    unittest.main()
