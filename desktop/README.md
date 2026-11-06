# Spyder Cal desktop app

An Electron interface for repeated SpyderX gamma measurements. The existing
Python library controls USB; PsychoPy presents the measurement patches.

## Standalone applications

Download the applications from the [Spyder Cal website](https://raffaelemazziotti.github.io/spyder_cal/).
See the [Windows/Linux installation guide](https://raffaelemazziotti.github.io/spyder_cal/installation.html)
for USB setup and first measurements, or the [Python API guide](https://raffaelemazziotti.github.io/spyder_cal/api.html)
for use in your own scripts.

For quicker Windows startup, extract the recommended ZIP download and run
`Spyder Cal.exe` directly. Keep that entire folder intact. In a local build,
the same executable is under `release/windows-0.1.2/win-unpacked/`.
The single-file portable executable shows a loading panel during extraction.

The `release` folder contains builds for **64-bit Windows and Linux**. Python,
PsychoPy, the USB library and plotting dependencies are included; Node/npm and
a separate Python installation are not needed to run them.

- **Windows:** double-click `Spyder-Cal-0.1.2-win-x64.exe`. This portable launcher
  extracts its bundled runtime to a separate temporary folder for every launch.
  Opening it again brings the existing window forward. Startup can take a moment.
  The SpyderX still needs a WinUSB or libusbK driver (see the root README).
- **Linux:** make `Spyder-Cal-0.1.2-linux-x86_64.AppImage` executable and launch it.
  It was built on Ubuntu 22.04 x86_64. A graphical desktop and the normal
  OpenGL/GTK/NSS/audio system libraries are required.

```bash
chmod +x Spyder-Cal-0.1.2-linux-x86_64.AppImage
./Spyder-Cal-0.1.2-linux-x86_64.AppImage
# If FUSE is unavailable:
./Spyder-Cal-0.1.2-linux-x86_64.AppImage --appimage-extract-and-run
```

The Linux build command also produces a `.tar.gz` distribution: extract it and
run `spyder-cal` inside the extracted directory, keeping the other files beside it.
Install the repository's `udev/60-spyderx.rules` as described in the root README
and reconnect the sensor to grant ordinary user access. Run the app as your
normal desktop user.

History is kept in the OS application-data folder, independently of the
executable. Older test/demo sessions are hidden; actual measurements remain.

## Run from source

On Windows, once dependencies are installed, double-click **Start Spyder
Cal.cmd** in the repository's top-level folder. This launcher works regardless
of the terminal's current directory. The `desktop` folder below refers to the
folder inside this repository, not your Windows Desktop folder.

Requires **Node.js 22.12 or newer** and a Python environment with the repository's
`requirements.txt` installed. You can select an existing PsychoPy Python
executable using **Runtime settings** in the app.

From the repository root, set up a local environment on Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd desktop
npm.cmd ci
npm.cmd start
```

On Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd desktop
npm ci
npm start
```

Follow the repository's Linux libusb and udev setup. Use a PsychoPy-supported
Python environment; its native display dependencies depend on the distribution.
The app discovers `.venv` in the repository first, then `python` (Windows) or
`python3` (Linux). `SPYDER_PYTHON` can specify another interpreter.


## Compact workspace

The top control bar contains **Connect**, **Test screen**, **New**, **Open**,
**Save files**, and runtime/USB settings. The app starts with an empty current
measurement. Previous measurements are available only when requested under
**Open → Recent measurements**, or import an exported JSON with **Open → Open file**.
There is no permanent history sidebar or separate PsychoPy result footer.

Copy buttons are inside the **Gamma**, **Black luminance**, **White luminance**,
and **Corrected error** cards. They copy the full value to six decimal places;
black and white are cd/m² and corrected error is percent. Copy the original
gamma exponent into PsychoPy. Save files exports JSON, CSV, PNG and SVG together.

The two plots fill the remaining window area. Axis labels retain their text
size as the graph geometry adapts. The compact layout is checked without page
scrolling at 1120×680 and 960×560 content sizes.

## Loading feedback

The Windows portable executable displays a native loading panel during runtime
extraction. Once Electron opens, a spinner remains visible while the measurement
runtime starts. Opening a PsychoPy window also shows a spinner and a Cancel
button until the fullscreen window is ready. Python starts without importing
NumPy/SciPy/Matplotlib/PsychoPy; those packages load only when needed.

Portable launch still extracts its bundled dependencies each time, so startup
can take several seconds. Keep the app open between measurements.

## Measurement workflow

1. Press **Connect** in the top bar and select the **Screen** number.
   Screen numbering follows PsychoPy's pyglet backend; confirm the target
   display when its placement window opens. **Test screen** opens a fullscreen
   PsychoPy test window for five seconds without using the sensor.
2. Name the monitor and set gray levels, repetitions and settling time.
3. Press **Measure**. The fullscreen PsychoPy window opens immediately.
   Close the sensor cover and press **Space** or **Enter** there to confirm
   dark calibration. All setup prompts stay in that window, including on
   a single-monitor computer.
4. Open the cover and place the sensor against the selected display. Press
   **Space** in the measurement window to start. **Escape** cancels the session. Progress and instructions stay in a fixed
   strip at the bottom, outside the sensor position at the center. Both sweeps
   use the same patch geometry, recorded in the JSON.
5. At completion, remove the sensor and press **Space**, **Enter** or **Escape**
   to return to the dashboard. Inspect the two plots side by side: original
   response on the left and corrected response on the right. Both share axes.
   Each shows individual runs, means and standard deviation. When verification is enabled, the app performs another set of
   readings with inverse-gamma-corrected patch values and plots those actual
   readings. The corrected error is RMSE as a percentage of the original
   black-to-white luminance range.
6. Repeat as needed. Completed sessions are saved locally. **New** clears the
   current view; **Open → Recent measurements** retrieves a previous session.

Keep monitor brightness, contrast, HDR, adaptive brightness, Night Light,
existing calibration and other display settings fixed between measurement,
verification and use in the experiment. Calibrations are specific to that
display configuration. The app measures grayscale, not independent RGB gamma.

## Use the value in PsychoPy

Use **Copy** in the **Gamma** card and enter the value in PsychoPy's Monitor Center, or use it when creating
the experiment window:

```python
from psychopy import visual

win = visual.Window(screen=0, fullscr=True, gamma=2.215)  # use YOUR measured value
```

Enter the original measured exponent, not its reciprocal and not the corrected
sweep's fitted gamma (which should approach 1). The inverse exponent is used
internally to generate corrected test patches.

Verification in this app applies correction to patch pixel values. PsychoPy's
window gamma setting uses a display gamma table and depends on the graphics
backend and OS. Validate the final experiment's output after entering the
value. The app's files do not automatically install a desktop color profile.
See PsychoPy's [gamma correction](https://psychopy.org/general/gamma.html) and
[Window](https://psychopy.org/api/visual/window.html) documentation for the
underlying correction and display-table behavior.

## Files

**Save files** saves four files with the selected base name:

| File | Contents |
| --- | --- |
| `.json` | All original and corrected samples, per-run fits, summaries, settings, timestamp, gamma and correction LUT |
| `.csv` | One row per sensor reading: phase, repetition, requested intensity, actual drive value and luminance |
| `.png` | Plot of individual runs, means, SD, fitted model and target |
| `.svg` | The same plot as editable vector graphics |

Gamma is available at `gamma` and `psychopy.gamma` in the JSON. **Open → Open file** reopens an app-exported JSON file. Existing command-line exports use
a different schema and remain usable through the original Python API.

## Build executables

Build **on each target OS**; the bundled Python runtime is platform specific.
Use Python 3.10 and Node.js 22.12+ (the release used Node 24 and Electron 45).
From the repository root, install `requirements-build.txt` into `.venv`, then:

```bash
cd desktop
npm ci
npm run dist
```

On Windows use `npm.cmd` when PowerShell blocks `npm.ps1`.
`SPYDER_PYTHON` can select another build interpreter. `dist` freezes the worker
with PyInstaller, records dependency versions and packages the native Electron
app into `release`. The backend spec includes PsychoPy resources and both
Matplotlib PNG and SVG exporters. Source builds and standalone builds use the
same measurement and export code. Third-party licenses travel with the bundle.

## Checks

From the repository root:

```bash
python -m unittest discover -s tests -v
```

From `desktop`:

```bash
npm run check
npm test
npm run test:ui
npm run test:psychopy
```

The worker integration test requires `.venv` or `SPYDER_PYTHON`. The Electron
renderer smoke test loads a fixture generated only by test code and checks
loading indicators, a fresh measurement on startup, two plots, small-window
layout without scrolling, four card copy actions, normalization, export/import,
optional history and real PsychoPy display loading. Production has no simulated measurement mode. Native file dialogs
and the OS clipboard are replaced with test destinations. A desktop session
is needed to run the Electron smoke test; it does not perform sensor calibration.

`test:psychopy` imports the real PsychoPy renderer through the worker and opens
screen 0 for five seconds. It needs PsychoPy and a graphical desktop. This
checks the real import/display path, including the Windows subprocess input
isolation which prevents an import-time Git version probe from hanging.

`SPYDER_TEST_PACKAGED=1 npm run test:ui` exercises the bundled-runtime route.
`node test/frozen.cjs` checks exports, USB error recovery and a real fullscreen
window in the frozen runtime. `python test/native-workflow.py` uses a test-only
sensor fixture with the actual PsychoPy window to check the complete flow.
Neither test performs physical sensor calibration. Linux USB hardware must be
verified on a native Linux installation with the sensor connected.

### Windows portable regression check

After building a Windows portable executable, run from the repository root:

```powershell
.venv\Scripts\python.exe desktop\test\portable-regression.py release\Spyder-Cal-0.1.2-win-x64.exe
```

This launches the actual executable with isolated test profiles, confirms that
reopening the app preserves its active runtime, closes another launch, and
checks that PsychoPy's VERSION file is still present. It then opens a real
fullscreen PsychoPy window and exports JSON/CSV/PNG/SVG from that surviving
extracted runtime. No physical sensor calibration is performed.

The builder is pinned to 26.18.0. Its `unpackDirName: true` implementation
uses the unique NSIS `$PLUGINSDIR` location; its `false` implementation uses a
shared folder despite the option documentation. Do not switch this to a
shared extraction path: one launch's cleanup can delete another's resources.
