# SpyderX Monitor Calibration Library

This project uses PyUSB to communicate with a Datacolor SpyderX Pro and
PsychoPy to present gray levels for monitor gamma measurement. Linux, Windows,
and macOS use the same SpyderX protocol and calibration calculations; only the
libusb setup differs by platform.

## Project structure

- `cal_lib.py` contains `SpyderX` and the gamma-fitting code.
- `cal_psy.py` contains the PsychoPy `GrayLevels` presentation helper.
- `Demo.py` demonstrates a complete calibration run.
- `diagnostics/test_spyder.py` checks USB initialization without changing
  factory calibration data.

## Python dependencies

Create and activate a virtual environment, then install the platform-neutral
Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows cmd: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyUSB is the USB abstraction on every supported platform. The native libusb
library is a system dependency and is intentionally not a Windows-only Python
package in `requirements.txt`.

## Linux setup

On Ubuntu/Xubuntu:

```bash
sudo apt update
sudo apt install libusb-1.0-0 python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-usb.txt
```

`SpyderX()` asks PyUSB to discover the installed libusb automatically. Root is
not required. If your user cannot open the USB device, install this udev rule
for the project's SpyderX USB ID, `085c:0a00`:

```udev
SUBSYSTEM=="usb", ATTR{idVendor}=="085c", ATTR{idProduct}=="0a00", MODE="0660", GROUP="plugdev", TAG+="uaccess"
```

For example:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="085c", ATTR{idProduct}=="0a00", MODE="0660", GROUP="plugdev", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/60-spyderx.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug and reconnect the SpyderX, then verify enumeration and initialization:

```bash
lsusb -d 085c:0a00
python diagnostics/test_spyder.py
```

That smaller requirements file is sufficient for USB diagnostics and direct
measurements. To run `Demo.py` and `cal_psy.py`, also install PsychoPy with
`python -m pip install -r requirements.txt`. PsychoPy's Linux installation can
require a supported Python version and additional GUI dependencies; consult
its official Linux installation guide if pip cannot provide suitable wheels
for your Ubuntu/Python combination.

The rule is normally required only when `lsusb` sees the device but the
diagnostic reports a permission/access error. Do not run the diagnostic with
`sudo`; fix udev access instead. Some distributions use a different desktop
access policy or group, so adjust `GROUP="plugdev"` if that group is absent.

## macOS setup

Install libusb with Homebrew when it is not already available, then install the
Python dependencies:

```bash
brew install libusb
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python diagnostics/test_spyder.py
```

No library path is normally necessary: `SpyderX()` uses PyUSB automatic
backend discovery.

## Windows setup

The existing explicit-DLL workflow remains supported. Install a
libusb-compatible device driver for the SpyderX (for example libusbK using
[Zadig](https://zadig.akeo.ie/)); the vendor DataColor driver cannot be used by
PyUSB at the same time. If an unsuitable DataColor driver remains installed,
the existing [Driver Store Explorer](https://github.com/lostindark/DriverStoreExplorer)
workflow may still be needed before selecting the device driver with Zadig.

Install libusb through the existing vcpkg workflow if desired:

```bat
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg install libusb:x64-windows
```

The DLL is normally under
`vcpkg\installed\x64-windows\bin\libusb-1.0.dll`. If it is discoverable through
the normal Windows DLL search path, this works:

```python
spyder = SpyderX()
```

Otherwise preserve the explicit path:

```python
spyder = SpyderX(
    libusb_path=r"C:\path\to\libusb-1.0.dll"
)
```

When changing a USB device driver with Zadig, carefully select the SpyderX;
replacing the driver for another device can make that device unusable until its
driver is restored.

## Usage

Automatic libusb discovery is the preferred cross-platform API:

```python
from cal_lib import SpyderX

with SpyderX() as spyder:
    # Cover/close the sensor before black calibration.
    spyder.calibrate()
    print(spyder.get_luminance())
```

The context manager releases the claimed interface, restores a kernel driver
that it detached, and disposes PyUSB resources even if measurement raises an
exception. Existing non-context-manager usage remains valid; call
`spyder.close()` when finished.

For monitor calibration:

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

`size`, `pos`, `fullscr`, and `screen` are passed to PsychoPy's window. The
defaults retain the previous 800x600 windowed behavior. Display numbering and
window positioning are controlled by PsychoPy and the host window system.

## Diagnostic and tests

The default diagnostic only discovers and initializes the device. It does not
write permanent calibration data:

```bash
python diagnostics/test_spyder.py
```

An optional black calibration and one luminance measurement can be requested:

```bash
python diagnostics/test_spyder.py --measure
```

On Windows an explicit DLL can also be supplied:

```bat
python diagnostics\test_spyder.py --libusb-path C:\path\to\libusb-1.0.dll
```

Run the mocked test suite without connecting a USB device:

```bash
python -m unittest discover -s tests -v
```

## Calibration guidance

- Use `fullscr=True` for final display measurements.
- Allow the display to stabilize before measuring.
- Cover the SpyderX during black calibration.
- Minimize ambient light and repeat measurements when appropriate.

The SpyderX implementation was adapted from
[patrickmineault/spyderX](https://github.com/patrickmineault/spyderX).
