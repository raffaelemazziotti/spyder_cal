"""UI-independent repeated gamma measurements and calibration-file exports."""

import csv
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cal_lib import GammaFitter


@dataclass(frozen=True)
class SessionSettings:
    monitor_name: str = "My monitor"
    screen: int = 0
    num_levels: int = 15
    repetitions: int = 3
    pause: float = 1.0
    verify: bool = True
    libusb_path: str = ""

    def __post_init__(self):
        for name, low, high in [
            ("screen", 0, 31), ("num_levels", 5, 65), ("repetitions", 1, 10)
        ]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} to {high}")
        if not isinstance(self.monitor_name, str) or not self.monitor_name.strip():
            raise ValueError("Give this monitor a name")
        if len(self.monitor_name) > 120:
            raise ValueError("Monitor name must be at most 120 characters")
        if type(self.pause) not in (int, float) or not math.isfinite(self.pause):
            raise ValueError("Settling time must be a finite number")
        if not 0.1 <= self.pause <= 10:
            raise ValueError("Settling time must be from 0.1 to 10 seconds")
        if type(self.verify) is not bool:
            raise ValueError("verify must be true or false")
        if not isinstance(self.libusb_path, str):
            raise ValueError("libusb_path must be a library path")


def _fit_run(levels, drives, luminance, repetition):
    values = np.asarray(luminance, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("The sensor returned a non-finite luminance reading")
    if values[-1] - values[0] <= 1e-6:
        raise ValueError(
            "White was not brighter than black. Check the selected display "
            "and sensor placement, then repeat the measurement."
        )
    fit = GammaFitter(levels, values)
    fit.set_initial_guess([1.0, 2.2, 0.0])
    fit.set_bounds([0, 0.1, 0], [10, 5, 1])
    fit.fit()
    return {
        "repetition": repetition,
        "gamma": float(fit.gamma),
        "fit_parameters": fit.params.tolist(),
        "levels": list(levels),
        "drive_values": list(drives),
        "luminance": values.tolist(),
    }


def _summarize(runs, reference=None):
    matrix = np.asarray([run["luminance"] for run in runs])
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
    gamma = np.asarray([run["gamma"] for run in runs])
    black, white = reference or (mean[0], mean[-1])
    normalized = (mean - black) / (white - black)
    levels = runs[0]["levels"]
    return {
        "runs": runs,
        "gamma": float(gamma.mean()),
        "gamma_std": float(gamma.std(ddof=1)) if len(runs) > 1 else 0.0,
        "black_luminance": float(mean[0]),
        "white_luminance": float(mean[-1]),
        "linearity_rmse_percent": float(
            np.sqrt(np.mean((normalized - levels) ** 2)) * 100
        ),
        "mean": [
            {"level": float(x), "luminance": float(y), "sd": float(s)}
            for x, y, s in zip(levels, mean, sd)
        ],
    }


def measure_session(settings, read_patch, emit, check_cancel, simulated=False):
    """Collect baseline and optional *measured* correction-validation sweeps.

    read_patch receives (drive_0_to_1, phase, repetition). For verification,
    correction is applied to patch pixel values; no OS gamma ramp is changed.
    The exported PsychoPy gamma is the measured exponent, not its reciprocal.
    """
    levels = np.linspace(0, 1, settings.num_levels).tolist()
    phases = 2 if settings.verify else 1
    total = settings.num_levels * settings.repetitions * phases
    completed = 0

    def collect(phase, correction_gamma=None):
        nonlocal completed
        runs = []
        for repetition in range(1, settings.repetitions + 1):
            emit("phase", phase=phase, repetition=repetition)
            readings, drives = [], []
            for index, level in enumerate(levels):
                check_cancel()
                drive = level if correction_gamma is None else level ** (1 / correction_gamma)
                reading = float(read_patch(drive, phase, repetition))
                check_cancel()
                if not math.isfinite(reading):
                    raise ValueError("The sensor returned a non-finite luminance reading")
                drives.append(drive)
                readings.append(reading)
                completed += 1
                emit(
                    "sample", phase=phase, repetition=repetition, index=index,
                    level=level, drive=drive, luminance=reading,
                    completed=completed, total=total,
                )
            runs.append(_fit_run(levels, drives, readings, repetition))
        reference = None if correction_gamma is None else (baseline["black_luminance"], baseline["white_luminance"])
        summary = _summarize(runs, reference)
        emit("summary", phase=phase, summary=summary)
        return summary

    baseline = collect("uncorrected")
    gamma = baseline["gamma"]
    corrected = collect("corrected", gamma) if settings.verify else None
    return {
        "schema_version": 1,
        "kind": "spyder_cal.session",
        "id": uuid.uuid4().hex,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "simulated": simulated,
        "gamma": gamma,
        "gamma_std": baseline["gamma_std"],
        "settings": asdict(settings),
        "luminance_unit": "cd/m^2",
        "uncorrected": baseline,
        "corrected": corrected,
        "correction": {
            "method": "inverse_power",
            "gamma": gamma,
            "exponent": 1 / gamma,
            "validation_method": "stimulus_pixel_values",
            "lut": (np.linspace(0, 1, 256) ** (1 / gamma)).tolist(),
        },
        "psychopy": {
            "gamma": gamma,
            "gamma_std": baseline["gamma_std"],
            "note": "Grayscale gamma estimate; separate RGB channels were not measured.",
            "example": f"win = visual.Window(screen={settings.screen}, gamma={gamma:.6f})",
        },
    }


def validate_session(session):
    """Reject incomplete/foreign data before importing or exporting a session."""
    if not isinstance(session, dict) or session.get("kind") != "spyder_cal.session" or session.get("schema_version") != 1:
        raise ValueError("This is not a supported Spyder Cal session JSON file")
    settings = SessionSettings(**session["settings"])
    if not isinstance(session.get("id"), str) or not 1 <= len(session["id"]) <= 128:
        raise ValueError("Invalid session identifier")
    try:
        if datetime.fromisoformat(session["created_utc"]).tzinfo is None:
            raise ValueError("Missing time zone")
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError("Invalid session timestamp") from error
    if type(session.get("simulated")) is not bool:
        raise ValueError("Missing session source")

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    for phase in ("uncorrected", "corrected"):
        summary = session.get(phase)
        if summary is None and phase == "corrected" and not settings.verify:
            continue
        if not isinstance(summary, dict) or not summary.get("runs"):
            raise ValueError("Session has no completed measurement runs")
        gamma = summary.get("gamma")
        if not finite(gamma) or not 0 < gamma <= 5:
            raise ValueError("Invalid session gamma")
        if len(summary["runs"]) != settings.repetitions:
            raise ValueError("Session repetition count does not match its settings")
        for key in ("gamma_std", "black_luminance", "white_luminance", "linearity_rmse_percent"):
            if not finite(summary.get(key)):
                raise ValueError("Invalid session summary")
        mean = summary.get("mean")
        if not isinstance(mean, list) or len(mean) != settings.num_levels:
            raise ValueError("Invalid session mean curve")
        for point in mean:
            if not isinstance(point, dict) or not all(finite(point.get(key)) for key in ("level", "luminance", "sd")):
                raise ValueError("Invalid mean measurement")
        for run in summary["runs"]:
            lengths = [len(run[key]) for key in ("levels", "drive_values", "luminance")]
            if len(set(lengths)) != 1 or lengths[0] != settings.num_levels:
                raise ValueError("Invalid measurement samples")
            for key in ("levels", "drive_values", "luminance"):
                if not all(finite(x) for x in run[key]):
                    raise ValueError("Invalid measurement value")
            for key in ("levels", "drive_values"):
                if any(not 0 <= x <= 1 for x in run[key]) or any(b <= a for a, b in zip(run[key], run[key][1:])):
                    raise ValueError("Sample intensities must increase from black to white")
            if not finite(run.get("gamma")) or not 0 < run["gamma"] <= 5:
                raise ValueError("Invalid run gamma")
    baseline_gamma = session["uncorrected"]["gamma"]
    if not isinstance(session.get("psychopy"), dict) or not finite(session["psychopy"].get("gamma")) or not math.isclose(session["psychopy"]["gamma"], baseline_gamma):
        raise ValueError("PsychoPy gamma does not match the original measurement")
    return session


def export_session(session, path):
    """Save all numeric data, tidy CSV, and publication-friendly PNG/SVG plots."""
    validate_session(session)
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    output = Path(path).expanduser().resolve().with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path, png_path, svg_path = [output.with_suffix(s) for s in (".csv", ".png", ".svg")]
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(session, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "repetition", "requested_level", "drive_value", "luminance_cd_m2", "simulated"])
        for phase in ("uncorrected", "corrected"):
            if session.get(phase) is None:
                continue
            for run in session[phase]["runs"]:
                for x, drive, luminance in zip(run["levels"], run["drive_values"], run["luminance"]):
                    writer.writerow([phase, run["repetition"], x, drive, luminance, session["simulated"]])

    figure, panels = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    try:
        baseline = session["uncorrected"]
        x = np.linspace(0, 1, 256)
        black, white = baseline["black_luminance"], baseline["white_luminance"]
        for axes, phase, label, color in zip(panels, ("uncorrected", "corrected"), ("Luminance response", "Corrected response"), ("#9383ba", "#639d89")):
            summary = session.get(phase)
            axes.plot(x, black + (white - black) * x, "--", color="#9a9e98", label="Linear target")
            if phase == "uncorrected":
                axes.plot(x, black + (white - black) * x ** baseline["gamma"], ":", color="#82759a", label="Gamma model")
            if summary is not None:
                for run in summary["runs"]:
                    axes.plot(run["levels"], run["luminance"], color=color, alpha=0.2, linewidth=1)
                mean = summary["mean"]
                axes.errorbar([p["level"] for p in mean], [p["luminance"] for p in mean], yerr=[p["sd"] for p in mean], color=color, marker="o", markersize=4, linewidth=2, label="Measured mean ±1 SD", capsize=2)
            else:
                axes.text(.5, .5, "Correction was not measured", transform=axes.transAxes, ha="center", color="#69726e")
            axes.set(title=label, xlabel="Requested intensity (0–1)")
            axes.grid(alpha=0.2)
            axes.legend(fontsize=8)
        panels[0].set_ylabel("Luminance (cd/m²)")
        title = session["settings"]["monitor_name"]
        if session["simulated"]:
            title += " — TEST DATA"
        figure.suptitle(f"{title} · Gamma {baseline['gamma']:.4f} ± {baseline['gamma_std']:.4f}")
        figure.tight_layout()
        figure.savefig(png_path, dpi=180)
        figure.savefig(svg_path)
    finally:
        plt.close(figure)
    return [str(p) for p in (output, csv_path, png_path, svg_path)]
