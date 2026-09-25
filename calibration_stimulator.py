"""Calibration routine for this computer's single-monitor Linux setup."""

import argparse
import json
import os
import platform
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev

import matplotlib
import numpy as np

# This must run before cal_lib imports matplotlib.pyplot. The stimulus window is
# still provided by PsychoPy; only the final fit plot uses this headless backend.
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

from cal_lib import GammaFitter, SpyderX


STIMULUS_CONNECTOR = "HDMI-1-2"
STIMULUS_SCREEN = 0
STIMULUS_SIZE = (1920, 1080)
STIMULUS_DESKTOP_POSITION = (0, 0)
DEFAULT_OUTPUT = Path("calibration_results/stimulator_calibration.json")
DEFAULT_RUNTIME_OUTPUT = Path("calibration_results/stimulator_gamma.json")
DEFAULT_PLOT_OUTPUT = Path("calibration_results/stimulator_calibration.png")


@contextmanager
def defer_fit_plot():
    """Prevent GrayLevels.measure() from plotting before its data is saved."""
    original_plot = GammaFitter.plot
    GammaFitter.plot = lambda _fit: None
    try:
        yield
    finally:
        GammaFitter.plot = original_plot


def create_stimulator_gray_levels(spyder):
    """Create a fullscreen GrayLevels adapter on the only connected monitor."""
    from psychopy import visual
    from cal_psy import GrayLevels

    class StimulatorGrayLevels(GrayLevels):
        def __init__(self, spyder_device):
            self.spyder = spyder_device
            self.win = visual.Window(
                size=STIMULUS_SIZE,
                screen=STIMULUS_SCREEN,
                fullscr=True,
                allowGUI=False,
                waitBlanking=True,
                color=[0, 0, 0],
                units="norm",
            )
            self.bg_rect = visual.Rect(
                self.win,
                width=2,
                height=2,
                fillColor=[0, 0, 0],
                lineColor=None,
            )
            self.bg_rect.draw()

    return StimulatorGrayLevels(spyder)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width-cm", type=float, default=34.4)
    parser.add_argument("--height-cm", type=float, default=19.3)
    parser.add_argument("--viewing-distance-cm", type=float, default=10)
    parser.add_argument("--num-levels", type=int, default=12)
    parser.add_argument("--pause", type=float, default=1)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--runtime-output", type=Path, default=DEFAULT_RUNTIME_OUTPUT
    )
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_OUTPUT)
    return parser.parse_args(argv)


def summarize_gammas(fits):
    gamma_values = [float(fit.gamma) for fit in fits]
    std_gamma = stdev(gamma_values) if len(gamma_values) > 1 else 0.0
    return gamma_values, fmean(gamma_values), std_gamma


def summarize_luminance(fits):
    """Summarize the measured black and white luminance across repetitions."""
    luminance_by_repetition = [
        np.asarray(fit.original_luminance.tolist(), dtype=float)
        for fit in fits
    ]
    minimum_values = [float(np.min(values)) for values in luminance_by_repetition]
    maximum_values = [float(np.max(values)) for values in luminance_by_repetition]
    std_minimum = stdev(minimum_values) if len(minimum_values) > 1 else 0.0
    std_maximum = stdev(maximum_values) if len(maximum_values) > 1 else 0.0
    return {
        "unit": "cd/m^2",
        "minimum_by_repetition": minimum_values,
        "maximum_by_repetition": maximum_values,
        "mean_minimum": fmean(minimum_values),
        "mean_maximum": fmean(maximum_values),
        "std_minimum": std_minimum,
        "std_maximum": std_maximum,
    }


def save_calibration(path, fits, mean_gamma, std_gamma, luminance_summary, args):
    """Atomically save every run returned by GrayLevels/GammaFitter."""
    result = {
        "format_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repetitions": len(fits),
        "runs": [
            {
                "repetition": index,
                "gamma": float(fit.gamma),
                "gray_levels": fit.original_intensities.tolist(),
                "luminance": fit.original_luminance.tolist(),
                "fit_parameters": fit.params.tolist(),
            }
            for index, fit in enumerate(fits, start=1)
        ],
        "mean_gamma": mean_gamma,
        "std_gamma": std_gamma,
        "luminance_summary": luminance_summary,
        "display": {
            "stimulus": {
                "connector": STIMULUS_CONNECTOR,
                "screen": STIMULUS_SCREEN,
                "resolution_px": list(STIMULUS_SIZE),
                "desktop_position_px": list(STIMULUS_DESKTOP_POSITION),
                "fullscr": True,
                "allow_gui": False,
                "wait_blanking": True,
            },
        },
        "physical_geometry": {
            "width_cm": args.width_cm,
            "height_cm": args.height_cm,
            "viewing_distance_cm": args.viewing_distance_cm,
        },
    }

    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def save_runtime_calibration(path, mean_gamma, luminance_summary):
    """Save gamma and the measured luminance range used by the stimulator."""
    result = {
        "schema_version": 2,
        "gamma": mean_gamma,
        "luminance": {
            "unit": luminance_summary["unit"],
            "minimum": luminance_summary["mean_minimum"],
            "maximum": luminance_summary["mean_maximum"],
        },
    }
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def save_fit_plot(path, fits, mean_gamma):
    """Save measured and fitted curves together without calling plt.show()."""
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure, axes = plt.subplots(figsize=(9, 6))
        for index, fit in enumerate(fits, start=1):
            intensities = fit.original_intensities
            luminance = fit.original_luminance
            measured_line = axes.plot(
                intensities,
                luminance,
                marker="o",
                label=f"Run {index} measured",
            )[0]

            x_normalized = np.linspace(0, 1, 100)
            y_normalized = fit.gamma_function(x_normalized, *fit.params)
            x_fitted = x_normalized * (
                np.max(intensities) - np.min(intensities)
            ) + np.min(intensities)
            y_fitted = y_normalized * (
                np.max(luminance) - np.min(luminance)
            ) + np.min(luminance)
            axes.plot(
                x_fitted,
                y_fitted,
                linestyle="--",
                color=measured_line.get_color(),
                label=f"Run {index} fit (gamma={fit.gamma:.6f})",
            )

        axes.set_xlabel("Gray level")
        axes.set_ylabel("Luminance (cd/m²)")
        axes.set_title(f"Stimulator gamma calibration — mean gamma={mean_gamma:.6f}")
        axes.grid(True)
        axes.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close("all")
    return output_path


def save_fit_plot_safely(path, fits, mean_gamma):
    """Save the optional plot without invalidating an already saved result."""
    try:
        return save_fit_plot(path, fits, mean_gamma)
    except Exception as error:
        print(f"Warning: calibration data was saved, but plot saving failed: {error}")
        return None


def open_saved_image(path):
    """Open an image with the operating system's default viewer."""
    system = platform.system()
    if system == "Windows" and hasattr(os, "startfile"):
        os.startfile(str(path))
        return

    command = "open" if system == "Darwin" else "xdg-open"
    subprocess.run(
        [command, str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def open_saved_image_safely(path):
    try:
        open_saved_image(path)
        return True
    except Exception as error:
        print(f"Warning: plot was saved, but could not be opened: {error}")
        return False


def run_repetitions(levels, repetitions, pause, num_levels):
    """Run complete, independent gray-level measurements."""
    fits = []
    for repetition in range(1, repetitions + 1):
        print(f"Calibration repetition {repetition}/{repetitions}")
        try:
            with defer_fit_plot():
                fit = levels.measure(
                    pause=pause,
                    num_levels=num_levels,
                    wait_user=repetition == 1,
                )
        except Exception as error:
            print(
                f"Calibration repetition {repetition}/{repetitions} failed: {error}"
            )
            raise

        fits.append(fit)
        print(f"Gamma: {fit.gamma:.6f}\n")
    return fits


def save_and_display_results(args, fits, mean_gamma, std_gamma):
    """Save both JSON files before optional plot and viewer operations."""
    luminance_summary = summarize_luminance(fits)
    output_path = save_calibration(
        args.output,
        fits,
        mean_gamma,
        std_gamma,
        luminance_summary,
        args,
    )
    runtime_output_path = save_runtime_calibration(
        args.runtime_output,
        mean_gamma,
        luminance_summary,
    )
    plot_path = save_fit_plot_safely(args.plot_output, fits, mean_gamma)
    if plot_path is not None:
        open_saved_image_safely(plot_path)
    return output_path, runtime_output_path, plot_path


def main():
    args = parse_args()

    with SpyderX() as spyder:
        print("SpyderX initialized.")
        levels = create_stimulator_gray_levels(spyder)
        print(
            f"Stimulus monitor opened: {STIMULUS_CONNECTOR}, "
            f"screen={STIMULUS_SCREEN}, size={STIMULUS_SIZE}, fullscreen."
        )
        try:
            levels.calibrate()
            fits = run_repetitions(
                levels,
                repetitions=args.repetitions,
                pause=args.pause,
                num_levels=args.num_levels,
            )
            gamma_values, mean_gamma, std_gamma = summarize_gammas(fits)
            luminance_summary = summarize_luminance(fits)
            output_path, runtime_output_path, plot_path = save_and_display_results(
                args, fits, mean_gamma, std_gamma
            )
        finally:
            levels.close()

    print("Calibration completed")
    print("Gamma repetitions: " + ", ".join(f"{value:.6f}" for value in gamma_values))
    print(f"Mean gamma: {mean_gamma:.6f}")
    print(f"Gamma SD: {std_gamma:.6f}")
    print(
        "Mean luminance range: "
        f"{luminance_summary['mean_minimum']:.6f} to "
        f"{luminance_summary['mean_maximum']:.6f} "
        f"{luminance_summary['unit']}"
    )
    print(f"Full calibration saved: {output_path}")
    print(f"Stimulator runtime calibration saved: {runtime_output_path}")
    if plot_path is None:
        print("Plot saved: unavailable (see warning above)")
    else:
        print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
