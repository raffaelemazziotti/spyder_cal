# **SpyderX Monitor Calibration Library**

This project provides Python classes to interface with a **[SpyderX Pro colorimeter](https://www.datacolor.com/spyder/products/spyder-x-pro/)** for measuring and calibrating monitor gamma. It combines **PyUSB** for USB communication and **PsychoPy** for precise visual presentation of gray levels.

---

## **Project Structure**

The project includes the following files:

1. **`cal_lib.py`**:
   - Contains the `SpyderX` class for communicating with the SpyderX device.
   - Includes `GammaFitter` to fit gamma curves using luminance measurements.

2. **`cal_psy.py`**:
   - Contains the `GrayLevels` class for measuring luminance and fitting gamma curves using **PsychoPy**.
   - Displays gray levels and measures luminance to calibrate monitor gamma.

3. **Example Usage**:
   - In `Demo.py` file you can find a demonstration on how to use the code

---

## **Dependencies**

Ensure the following libraries are installed:

- **PyUSB** for USB communication:
   ```bash
   pip install pyusb
  
- **PsychoPy** for visual stimulus presentation:
   ```bash
   pip install psychopy

 - **NumPy** and **SciPy**  for numerical computations:
   ```bash
   pip install numpy scipy

 - **Matplotlib** for plotting gamma curves:
   ```bash
   pip install matplotlib
   
## **Installing libusb on Windows**
**IMPORTANT**: This step requires visual studio with several [components](https://github.com/microsoft/vcpkg-tool/pull/314) installed. 
1. Clone and bootstrap vcpkg:
    ```bash
   git clone https://github.com/microsoft/vcpkg.git
    cd vcpkg
    bootstrap-vcpkg.bat
   
2. Install libusb:
    ```bash
   vcpkg install libusb
3. Find libusb-1.0.dll in:
    ```python
   vcpkg\installed\x64-windows\bin\libusb-1.0.dll
   
## **How to Use**

### **Basic Usage**
Here is a demo script that performs monitor calibration:
```python
from cal_lib import SpyderX
from cal_psy import GrayLevels
import numpy as np

# Path to the libusb-1.0.dll file
libusb_path = r"C:\path\to\libusb-1.0.dll"

# Initialize the SpyderX device
spyder = SpyderX(libusb_path)

# Initialize the GrayLevels class
gl = GrayLevels(spyder, fullscr=True)

# Perform black-level calibration
gl.calibrate()

# Measure gray levels and fit gamma
gammas = []
gfit = gl.measure(num_levels=12)
gammas.append(gfit.gamma)

# Repeat measurements for better accuracy
for i in range(3):
    gfit = gl.measure(num_levels=12, wait_user=False)
    gammas.append(gfit.gamma)

print(f"Average Gamma Value: {np.mean(gammas):.3f}")

# Clean up
gl.close()
```
### **Steps to Calibrate Your Monitor** 
1. **Install the libusb driver** as described above.
2. **Connect the SpyderX device** to your computer.
3. **Change the default driver to libusbK** using [Zadig](https://zadig.akeo.ie/)
3. **Run the script**:

   - Perform black calibration by covering the SpyderX sensor.
   - Place the SpyderX on the monitor as instructed.
   - Measure luminance values for a range of gray levels.

4. **Analyze the Gamma Curve**:

    - The script will fit a gamma curve and compute the gamma value.
    - The curve will be displayed using matplotlib.

5. **Average Multiple Measurements**:

    - For better results, repeat the measurement process.
   
## **Tips for Better Calibration**
1. **Run in Full-Screen Mode**:

    - Use `fullscr=True` when initializing the GrayLevels class to ensure precise display of gray levels.

2. **Stabilize the Display**:

    - Allow a brief pause (e.g., `pause=1` second) after each gray level display to ensure the monitor luminance stabilizes.

3. **Cover the SpyderX for Black Calibration**:

    - Ensure the SpyderX sensor is covered during black calibration for accurate measurements.

4. **Avoid Ambient Light**:

    - Perform calibration in a dark room to minimize interference from ambient light.

5. **Repeat Measurements**:

    - Repeating measurements multiple times and averaging the gamma value improves accuracy.

## **Troubleshooting**

   - If the system installs DataColor Driver remove them completely using [RAPR](https://github.com/lostindark/DriverStoreExplorer?tab=readme-ov-file) 
   - Then install libusbK using Zadig

## **Credits**
 - The SpyderX class was adapted from the work (for macOS) of [patrickmineault](https://github.com/patrickmineault/spyderX).