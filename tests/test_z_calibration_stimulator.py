import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


fake_psychopy = types.ModuleType("psychopy")
fake_psychopy.visual = mock.Mock()
fake_psychopy.event = mock.Mock()
fake_psychopy.core = mock.Mock()
sys.modules["psychopy"] = fake_psychopy

import calibration_stimulator as calibration


class ListValue:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def make_fit(gamma, luminance=None):
    return types.SimpleNamespace(
        original_intensities=ListValue([-1.0, 1.0]),
        original_luminance=ListValue(luminance or [0.1, 100.0]),
        params=ListValue([1.0, gamma, 0.0]),
        gamma=gamma,
    )


class StimulatorCalibrationTests(unittest.TestCase):
    def setUp(self):
        fake_psychopy.visual.reset_mock()
        fake_psychopy.visual.Window.return_value = mock.Mock()

    def test_uses_single_monitor_fullscreen_geometry(self):
        calibration.create_stimulator_gray_levels(mock.Mock())

        fake_psychopy.visual.Window.assert_called_once_with(
            size=(1920, 1080),
            screen=0,
            fullscr=True,
            allowGUI=False,
            waitBlanking=True,
            color=[0, 0, 0],
            units="norm",
        )

    def test_uses_non_interactive_matplotlib_backend(self):
        self.assertEqual(calibration.matplotlib.get_backend().lower(), "agg")

    def test_default_repetitions_is_three(self):
        args = calibration.parse_args([])

        self.assertEqual(args.repetitions, 3)
        self.assertEqual(args.runtime_output, calibration.DEFAULT_RUNTIME_OUTPUT)
        self.assertEqual((args.width_cm, args.height_cm), (34.4, 19.3))

    def test_cli_repetitions(self):
        self.assertEqual(
            calibration.parse_args(["--repetitions", "5"]).repetitions,
            5,
        )

    def test_runs_exactly_requested_number_of_full_measurements(self):
        levels = mock.Mock()
        levels.measure.side_effect = [make_fit(2.1), make_fit(2.2), make_fit(2.3)]

        with mock.patch("builtins.print"):
            fits = calibration.run_repetitions(levels, 3, pause=1, num_levels=12)

        self.assertEqual(levels.measure.call_count, 3)
        self.assertEqual([fit.gamma for fit in fits], [2.1, 2.2, 2.3])
        self.assertEqual(
            levels.measure.call_args_list,
            [
                mock.call(pause=1, num_levels=12, wait_user=True),
                mock.call(pause=1, num_levels=12, wait_user=False),
                mock.call(pause=1, num_levels=12, wait_user=False),
            ],
        )

    def test_mean_and_sample_standard_deviation(self):
        gamma_values, mean_gamma, std_gamma = calibration.summarize_gammas(
            [make_fit(2.0), make_fit(4.0), make_fit(6.0)]
        )

        self.assertEqual(gamma_values, [2.0, 4.0, 6.0])
        self.assertEqual(mean_gamma, 4.0)
        self.assertEqual(std_gamma, 2.0)

    def test_single_repetition_standard_deviation_is_zero(self):
        _, _, std_gamma = calibration.summarize_gammas([make_fit(2.2)])

        self.assertEqual(std_gamma, 0.0)

    def test_luminance_summary_contains_minimum_and_maximum(self):
        summary = calibration.summarize_luminance(
            [make_fit(2.1, [0.1, 90.0]), make_fit(2.3, [0.3, 110.0])]
        )

        self.assertEqual(summary["unit"], "cd/m^2")
        self.assertEqual(summary["minimum_by_repetition"], [0.1, 0.3])
        self.assertEqual(summary["maximum_by_repetition"], [90.0, 110.0])
        self.assertAlmostEqual(summary["mean_minimum"], 0.2)
        self.assertEqual(summary["mean_maximum"], 100.0)
        self.assertAlmostEqual(summary["std_minimum"], 0.1414213562373095)
        self.assertAlmostEqual(summary["std_maximum"], 14.142135623730951)

    def test_failed_repetition_aborts_without_returning_partial_results(self):
        levels = mock.Mock()
        levels.measure.side_effect = [make_fit(2.1), RuntimeError("USB failed")]

        with mock.patch("builtins.print"), self.assertRaisesRegex(
            RuntimeError, "USB failed"
        ):
            calibration.run_repetitions(levels, 3, pause=1, num_levels=12)

        self.assertEqual(levels.measure.call_count, 2)

    def test_json_contains_every_run_and_summary(self):
        fits = [make_fit(2.1, [0.1, 90.0]), make_fit(2.3, [0.2, 110.0])]
        args = types.SimpleNamespace(
            width_cm=21,
            height_cm=15,
            viewing_distance_cm=10,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration.json"
            luminance_summary = calibration.summarize_luminance(fits)
            saved_path = calibration.save_calibration(
                output, fits, 2.2, 0.1, luminance_summary, args
            )
            result = json.loads(saved_path.read_text(encoding="utf-8"))

        self.assertEqual(result["repetitions"], 2)
        self.assertEqual(result["format_version"], 2)
        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual([run["gamma"] for run in result["runs"]], [2.1, 2.3])
        self.assertEqual(result["runs"][0]["gray_levels"], [-1.0, 1.0])
        self.assertEqual(result["runs"][0]["luminance"], [0.1, 90.0])
        self.assertEqual(result["runs"][1]["fit_parameters"], [1.0, 2.3, 0.0])
        self.assertEqual(result["mean_gamma"], 2.2)
        self.assertEqual(result["std_gamma"], 0.1)
        self.assertEqual(result["luminance_summary"], luminance_summary)
        self.assertEqual(result["display"]["stimulus"]["connector"], "HDMI-1-2")
        self.assertEqual(result["display"]["stimulus"]["screen"], 0)
        self.assertTrue(result["display"]["stimulus"]["fullscr"])
        self.assertNotIn("control", result["display"])
        self.assertEqual(result["physical_geometry"]["width_cm"], 21)
        self.assertEqual(
            set(result),
            {
                "format_version",
                "created_utc",
                "repetitions",
                "runs",
                "mean_gamma",
                "std_gamma",
                "luminance_summary",
                "display",
                "physical_geometry",
            },
        )

    def test_runtime_json_contains_gamma_and_luminance_range(self):
        individual_gammas, mean_gamma, _ = calibration.summarize_gammas(
            [make_fit(2.1), make_fit(2.2), make_fit(2.4)]
        )
        luminance_summary = calibration.summarize_luminance(
            [make_fit(2.1, [0.1, 90.0]), make_fit(2.2, [0.3, 110.0])]
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stimulator_gamma.json"
            saved_path = calibration.save_runtime_calibration(
                output, mean_gamma, luminance_summary
            )
            result = json.loads(saved_path.read_text(encoding="utf-8"))

        self.assertEqual(set(result), {"schema_version", "gamma", "luminance"})
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["gamma"], mean_gamma)
        self.assertEqual(
            result["luminance"],
            {"unit": "cd/m^2", "minimum": 0.2, "maximum": 100.0},
        )
        self.assertNotIn(result["gamma"], individual_gammas)
        self.assertNotEqual(result["gamma"], 1.0 / mean_gamma)

    def test_failed_repetition_does_not_create_partial_runtime_file(self):
        levels = mock.Mock()
        levels.measure.side_effect = [make_fit(2.1), RuntimeError("USB failed")]
        spyder_context = mock.MagicMock()
        args = types.SimpleNamespace(
            width_cm=21,
            height_cm=15,
            viewing_distance_cm=10,
            num_levels=12,
            pause=1,
            repetitions=3,
        )

        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / "full.json"
            args.runtime_output = Path(directory) / "stimulator_gamma.json"
            args.plot_output = Path(directory) / "plot.png"
            with mock.patch.object(
                calibration, "parse_args", return_value=args
            ), mock.patch.object(
                calibration, "SpyderX", return_value=spyder_context
            ), mock.patch.object(
                calibration, "create_stimulator_gray_levels", return_value=levels
            ), mock.patch("builtins.print"), self.assertRaisesRegex(
                RuntimeError, "USB failed"
            ):
                calibration.main()

            self.assertFalse(args.runtime_output.exists())
            self.assertFalse(args.output.exists())
        levels.close.assert_called_once_with()

    def test_plot_and_display_happen_after_json_saving(self):
        events = []
        args = types.SimpleNamespace(
            output=Path("result.json"),
            runtime_output=Path("runtime.json"),
            plot_output=Path("result.png"),
        )

        with mock.patch.object(
            calibration,
            "save_calibration",
            side_effect=lambda *_args: events.append("json") or args.output,
        ), mock.patch.object(
            calibration,
            "save_runtime_calibration",
            side_effect=lambda *_args: events.append("runtime")
            or args.runtime_output,
        ), mock.patch.object(
            calibration,
            "save_fit_plot_safely",
            side_effect=lambda *_args: events.append("plot") or args.plot_output,
        ), mock.patch.object(
            calibration,
            "open_saved_image_safely",
            side_effect=lambda *_args: events.append("display") or True,
        ):
            calibration.save_and_display_results(
                args,
                [make_fit(2.2)],
                mean_gamma=2.2,
                std_gamma=0.0,
            )

        self.assertEqual(events, ["json", "runtime", "plot", "display"])

    def test_combined_plot_is_saved_without_interactive_ui(self):
        fit = types.SimpleNamespace(
            original_intensities=calibration.np.array([-1.0, 0.0, 1.0]),
            original_luminance=calibration.np.array([0.1, 25.0, 100.0]),
            params=calibration.np.array([1.0, 2.2, 0.0]),
            gamma=2.2,
            gamma_function=lambda x, a, b, c: a * x**b + c,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration.png"
            saved_path = calibration.save_fit_plot(output, [fit], 2.2)
            signature = saved_path.read_bytes()[:8]

        self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")

    def test_plot_failure_is_reported_without_being_raised(self):
        with mock.patch.object(
            calibration,
            "save_fit_plot",
            side_effect=RuntimeError("plot failed"),
        ), mock.patch("builtins.print") as print_mock:
            result = calibration.save_fit_plot_safely(
                Path("unused.png"), [make_fit(2.2)], 2.2
            )

        self.assertIsNone(result)
        print_mock.assert_called_once()
        self.assertIn("calibration data was saved", print_mock.call_args.args[0])

    def test_display_failure_is_non_fatal(self):
        with mock.patch.object(
            calibration,
            "open_saved_image",
            side_effect=OSError("viewer unavailable"),
        ), mock.patch("builtins.print") as print_mock:
            result = calibration.open_saved_image_safely(Path("result.png"))

        self.assertFalse(result)
        print_mock.assert_called_once()
        self.assertIn("could not be opened", print_mock.call_args.args[0])

    def test_measurement_plot_is_deferred_and_restored(self):
        original_plot = calibration.GammaFitter.plot

        with calibration.defer_fit_plot():
            self.assertIsNone(calibration.GammaFitter.plot(mock.Mock()))

        self.assertIs(calibration.GammaFitter.plot, original_plot)


if __name__ == "__main__":
    unittest.main()
