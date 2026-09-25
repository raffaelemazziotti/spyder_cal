# SpyderX display gamma measurement

Measure a monitor's gamma and luminance range with a Datacolor SpyderX and
PsychoPy. The library supports Linux, Windows, and macOS through PyUSB/libusb.

## Quick start

Install the Python dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Connect the SpyderX, then run:

```bash
python Demo.py
```

The program guides you through dark calibration and sensor placement on screen.
It performs three measurement runs and writes
`calibration_results/stimulator_gamma.json`.

Common command-line options can be used without editing the script:

```bash
# Use PsychoPy display 1 and collect five runs.
python Demo.py --screen 1 --repetitions 5

# Quick windowed test with a custom output file.
python Demo.py --windowed --levels 6 --output test-gamma.json

python Demo.py --help
```

The same workflow is available as a small Python API:

```python
from cal_lib import SpyderX

with SpyderX() as spyder:
    result = spyder.measure_gamma(repetitions=3)
    result.save_json("gamma.json")

print(f"Gamma: {result.gamma:.3f}")
print(
    f"Luminance: {result.luminance_min:.3f} to "
    f"{result.luminance_max:.3f} cd/m²"
)
```

`measure_gamma()` opens a fullscreen PsychoPy window on display 0 at its native
resolution. Its most useful options are:

```python
result = spyder.measure_gamma(
    repetitions=3,  # independent runs; the result contains their mean and SD
    num_levels=12,  # gray levels sampled from black to white
    pause=1,        # display stabilization time in seconds
    screen=0,       # PsychoPy display number
    fullscr=True,
)
```

The returned `GammaMeasurement` provides:

- `gamma`, `gamma_std`, and `gamma_values`
- `luminance_min`, `luminance_max`, and their standard deviations
- `runs`, containing every gray level, measured luminance, and fitted parameter
- `to_dict()` and `save_json(path)` for serialization

The JSON keeps `gamma` as a top-level scalar for simple runtime use and includes
the full measured curves for inspection:

```json
{
  "schema_version": 2,
  "gamma": 2.2,
  "gamma_std": 0.03,
  "gamma_by_repetition": [2.18, 2.22, 2.2],
  "luminance": {
    "unit": "cd/m^2",
    "minimum": 0.15,
    "maximum": 105.4
  },
  "runs": []
}
```

The actual file also contains per-repetition luminance extrema, their standard
deviations, and all samples in `runs`.

## Linux setup

Install libusb and the virtual-environment support package. On Ubuntu/Xubuntu:

```bash
sudo apt update
sudo apt install libusb-1.0-0 python3-venv
```

Linux normally needs a udev rule so your user can access the SpyderX without
running Python as root. A ready-made rule is included:

```bash
sudo cp udev/60-spyderx.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug and reconnect the SpyderX, then verify it:

```bash
lsusb -d 085c:0a00
python diagnostics/test_spyder.py
```

If access is still denied, confirm that the `plugdev` group exists on your
distribution or edit the group in `udev/60-spyderx.rules`. Do not work around
USB permissions by running the calibration with `sudo`, because a root process
may not be able to use your graphical session.

For USB diagnostics without PsychoPy, install only:

```bash
python -m pip install -r requirements-usb.txt
```

## macOS setup

The existing macOS workflow is unchanged. Install libusb with Homebrew and then
the Python dependencies:

```bash
brew install libusb
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python diagnostics/test_spyder.py
```

`SpyderX()` normally discovers Homebrew's libusb automatically.

## Windows setup

The existing Windows workflow is unchanged. The SpyderX must use a
libusb-compatible driver, such as libusbK installed with
[Zadig](https://zadig.akeo.ie/). Carefully select the SpyderX (`085c:0a00`);
replacing another device's driver can disable that device until restored.

When `libusb-1.0.dll` is on the normal DLL search path, use the same API as
Linux and macOS:

```python
with SpyderX() as spyder:
    result = spyder.measure_gamma()
```

The explicit DLL workflow remains supported:

```python
with SpyderX(
    libusb_path=r"C:\path\to\libusb-1.0.dll"
) as spyder:
    result = spyder.measure_gamma()
```

If needed, libusb can still be built or installed through vcpkg:

```bat
vcpkg install libusb:x64-windows
```

The DLL is normally under
`vcpkg\installed\x64-windows\bin\libusb-1.0.dll`.

## Direct luminance measurements

PsychoPy is not imported when using the low-level photometer API:

```python
from cal_lib import SpyderX

with SpyderX() as spyder:
    input("Cover the sensor, then press Enter...")
    spyder.calibrate()
    input("Place the sensor on the display, then press Enter...")
    print(f"Luminance: {spyder.get_luminance():.3f} cd/m²")
```

The diagnostic provides the same check:

```bash
python diagnostics/test_spyder.py --measure
```

On Windows, add `--libusb-path C:\path\to\libusb-1.0.dll` when required.

## Advanced PsychoPy control

Existing code using `GrayLevels` remains supported. This lower-level API is
useful for custom windows or measurement sequences:

```python
from cal_lib import SpyderX
from cal_psy import GrayLevels

with SpyderX() as spyder:
    levels = GrayLevels(
        spyder,
        size=(800, 600),
        pos=None,
        fullscr=False,
        screen=0,
    )
    try:
        levels.calibrate()
        fit = levels.measure(num_levels=12)
        print(f"Gamma: {fit.gamma:.3f}")
    finally:
        levels.close()
```

`GrayLevels.measure(plot=False)` suppresses the interactive fit plot, and
`GrayLevels.close(close_spyder=False)` closes only the PsychoPy window when the
device is managed elsewhere.

## Troubleshooting

- **Device not found:** check the cable and run `lsusb -d 085c:0a00` on Linux.
- **Permission denied on Linux:** install the included udev rule and reconnect.
- **No libusb backend:** install the native libusb package for your platform.
- **PsychoPy import failure:** install `requirements.txt` in a PsychoPy-supported
  Python environment.
- **Wrong display:** pass the correct PsychoPy `screen` number.
- **Unstable measurements:** minimize ambient light, allow the display to warm
  up, and use multiple repetitions.

## Tests and project layout

Run the mocked test suite without a connected device:

```bash
python -m unittest discover -s tests -v
```

- `cal_lib.py`: `SpyderX`, `GammaMeasurement`, and gamma fitting
- `cal_psy.py`: PsychoPy gray-level presentation
- `Demo.py`: recommended complete measurement example
- `calibration_stimulator.py`: machine-specific single-monitor routine
- `diagnostics/test_spyder.py`: USB and direct-luminance diagnostic

The SpyderX implementation was adapted from
[patrickmineault/spyderX](https://github.com/patrickmineault/spyderX).
