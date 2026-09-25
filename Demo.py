"""Measure display gamma and luminance with a Datacolor SpyderX."""

import argparse
from pathlib import Path

from cal_lib import SpyderX


DEFAULT_OUTPUT = Path("calibration_results/stimulator_gamma.json")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def minimum_three(value):
    value = int(value)
    if value < 3:
        raise argparse.ArgumentTypeError("must be at least 3")
    return value


def nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=int, default=0, help="PsychoPy screen number")
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument(
        "--levels", type=minimum_three, default=12, help="gray levels per run"
    )
    parser.add_argument(
        "--pause", type=nonnegative_float, default=1, help="seconds per level"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="use an 800x600 window instead of fullscreen",
    )
    parser.add_argument(
        "--libusb-path",
        help="explicit libusb library path (normally needed only on Windows)",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    with SpyderX(libusb_path=args.libusb_path) as spyder:
        result = spyder.measure_gamma(
            repetitions=args.repetitions,
            num_levels=args.levels,
            pause=args.pause,
            fullscr=not args.windowed,
            screen=args.screen,
            size=(800, 600) if args.windowed else None,
        )
        output = result.save_json(args.output)

    print(f"Calibration complete: {result}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
