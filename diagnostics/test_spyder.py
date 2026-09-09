"""Non-destructive command-line check for a connected SpyderX."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cal_lib import SpyderX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libusb-path",
        help="optional explicit path to libusb (primarily useful on Windows)",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="perform black calibration followed by one luminance measurement",
    )
    args = parser.parse_args()

    with SpyderX(libusb_path=args.libusb_path) as spyder:
        print("SpyderX found and initialized successfully (USB 085c:0a00).")
        if args.measure:
            input("Cover/close the sensor for black calibration, then press Enter...")
            spyder.calibrate()
            input("Position the sensor on the display, then press Enter...")
            print(f"Luminance: {spyder.get_luminance():.3f} cd/m²")


if __name__ == "__main__":
    main()
