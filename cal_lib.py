import usb.core
import usb.util
import usb.backend.libusb1
import numpy as np
import struct
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class SpyderX:
    """
    SpyderX colorimeter using PyUSB and a libusb 1.x backend.
    """

    INTERFACE = 0
    VENDOR_ID = 0x085C
    PRODUCT_ID = 0x0A00

    def __init__(self, libusb_path=None):
        """
        Initialize the SpyderX device.

        Args:
            libusb_path (str, optional): Explicit path to a libusb 1.x shared
                library. This preserves the existing Windows DLL workflow. If
                omitted, PyUSB discovers the system libusb automatically.
        """
        self.backend = None
        self.dev = None
        self._interface_claimed = False
        self._detached_kernel_driver = False
        self._closed = False

        if libusb_path is None:
            self.backend = usb.backend.libusb1.get_backend()
        else:
            self.backend = usb.backend.libusb1.get_backend(
                find_library=lambda _name: libusb_path
            )

        if self.backend is None:
            if libusb_path is None:
                raise RuntimeError(
                    "PyUSB could not find a libusb 1.x backend. Install libusb-1.0 "
                    "and make it discoverable by the operating system, or pass "
                    "libusb_path explicitly."
                )
            raise RuntimeError(
                f"PyUSB could not load the libusb 1.x library at: {libusb_path}"
            )

        # Find SpyderX device on the bus
        self.dev = usb.core.find(
            idVendor=self.VENDOR_ID,
            idProduct=self.PRODUCT_ID,
            backend=self.backend,
        )
        if self.dev is None:
            raise ValueError(
                "SpyderX device (USB 085c:0a00) not found. Check that it is "
                "connected, that your user has USB permission, and on Windows "
                "that a libusb-compatible driver is installed."
            )

        try:
            # An already configured device can reject this harmlessly.
            try:
                self.dev.set_configuration()
            except usb.core.USBError:
                pass

            self._detach_kernel_driver_if_needed()
            usb.util.claim_interface(self.dev, self.INTERFACE)
            self._interface_claimed = True

            # Initialize the device (control transfers, then get calibration data)
            self._initialize_device()
        except Exception:
            self.close()
            raise

    def _detach_kernel_driver_if_needed(self):
        """Detach an active kernel driver when the backend supports doing so."""
        is_active = getattr(self.dev, "is_kernel_driver_active", None)
        detach = getattr(self.dev, "detach_kernel_driver", None)
        if not callable(is_active) or not callable(detach):
            return

        try:
            if is_active(self.INTERFACE):
                detach(self.INTERFACE)
                self._detached_kernel_driver = True
        except (NotImplementedError, usb.core.USBError):
            # Windows and some libusb backends do not implement these methods.
            pass

    def _initialize_device(self):
        """
        Send the standard control transfers, then retrieve factory calibration and measurement setup.
        """
        # Standard set of control transfers to 'wake' the SpyderX
        self._ctrl(0x02, 1, 0, 1)
        self._ctrl(0x02, 1, 0, 129)
        self._ctrl(0x41, 2, 2, 0)

        # Retrieve factory calibration (matrix, v1, v2, etc.)
        self._get_calibration()
        # Retrieve measurement setup values (s1, s2, s3)
        self._setup_measurement()

    def _ctrl(self, bmRequestType, bRequest, wValue, wIndex):
        """
        Helper for control transfers.
        """
        self.dev.ctrl_transfer(bmRequestType, bRequest, wValue, wIndex, None)

    def _bulk(self, data, read_size, timeout=1000):
        """
        Helper for bulk transfers. Writes 'data' to endpoint 1, reads 'read_size' bytes from endpoint 0x81.
        """
        self.dev.write(1, data, timeout=timeout)
        return self.dev.read(0x81, read_size, timeout=timeout)

    def _read_ieee754(self, b):
        """
        Decode a 32-bit IEEE-754 float from 4 bytes in 'b'.
        The SpyderX data is little-endian, so we reverse and parse.
        """
        b = b[::-1]
        raw = int.from_bytes(b, "big")
        sign = (raw >> 31) & 1
        exponent = (raw >> 23) & 0xFF
        fraction = raw & 0x7FFFFF
        return (-1) ** sign * (1 + fraction / 2**23) * 2**(exponent - 127)

    def _get_calibration(self):
        """
        Retrieve factory calibration data, storing:
          - v1, v2: calibration parameters
          - calibration_matrix: 3x3 for XYZ computation
        """
        # Command 0xcb for factory calibration, read 47 bytes
        out = self._bulk([0xcb, 0x05, 0x73, 0x00, 0x01, 0x00], 47)[5:]
        self.v1 = out[1]
        self.v2 = int.from_bytes(out[2:4], 'big')

        # Build the 3x3 matrix
        matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                k = i*3 + j
                matrix[i, j] = self._read_ieee754(out[k*4 + 4 : k*4 + 8])

        self.calibration_matrix = matrix

    def _setup_measurement(self):
        """
        Retrieve s1, s2, s3 from the device, used in calibrate() and measure() calls.
        """
        # 0xc3 sets up measurement for the device
        payload = [0xc3, 0x29, 0x27, 0x00, 0x01, self.v1]
        out = self._bulk(payload, 15)

        self.s1 = out[5]
        self.s2 = out[6:10]
        self.s3 = out[10:14]

    def calibrate(self):
        """
        Perform black (dark) calibration.
        The user should cover the SpyderX lens before calling this.
        """
        # Standard control transfer first
        self._ctrl(0x41, 2, 2, 0)

        # build payload from v2, s1, s2
        payload = [self.v2 >> 8, self.v2 & 0xFF, self.s1] + list(self.s2)
        out = self._bulk([0xd2, 0x3f, 0xb9, 0x00, 0x07] + payload, 13)

        # parse the black calibration data
        raw = struct.unpack('>HHHH', out[5:])
        self.black_cal = np.array(raw[:3])

    def measure(self):
        """
        Perform a measurement, returning XYZ as a numpy array [X, Y, Z].
        Y is luminance in cd/m².
        """
        # Control transfer
        self._ctrl(0x41, 2, 2, 0)

        # build measurement payload
        payload = [self.v2 >> 8, self.v2 & 0xFF, self.s1] + list(self.s2)
        out = self._bulk([0xd2, 0x3f, 0xb9, 0x00, 0x07] + payload, 13)

        raw = np.array(struct.unpack('>HHHH', out[5:]))[:3]
        corrected = raw - self.black_cal
        xyz = np.dot(corrected, self.calibration_matrix)
        return xyz

    def get_luminance(self):
        """
        Return just the Y (luminance) component in cd/m².
        """
        return self.measure()[1]

    def get_rgb_luminance(self):
        """
        Convert measured XYZ to approximate linear RGB intensities.
        Useful for relative channel brightness checks.
        """
        xyz = self.measure()
        xyz_to_rgb = np.array([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.2040,  1.0570]
        ])
        rgb = np.dot(xyz_to_rgb, xyz)
        return tuple(rgb)

    def close(self):
        """
        Release the interface, restore its kernel driver, and dispose resources.

        This method is idempotent and safe after partially failed initialization.
        """
        if self._closed:
            return
        self._closed = True

        if self.dev is None:
            return

        if self._interface_claimed:
            try:
                usb.util.release_interface(self.dev, self.INTERFACE)
            except usb.core.USBError:
                pass
            finally:
                self._interface_claimed = False

        if self._detached_kernel_driver:
            attach = getattr(self.dev, "attach_kernel_driver", None)
            if callable(attach):
                try:
                    attach(self.INTERFACE)
                except (NotImplementedError, usb.core.USBError):
                    pass
            self._detached_kernel_driver = False

        try:
            usb.util.dispose_resources(self.dev)
        except usb.core.USBError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

def xyz_to_lms(xyz):
    # XYZ to LMS conversion matrix (Hunt-Pointer-Estevez)
    xyz_to_lms_matrix = np.array([
        [0.4002, 0.7076, -0.0808],
        [-0.2263, 1.1653, 0.0457],
        [0.0, 0.0, 0.9182]
    ])
    return np.dot(xyz_to_lms_matrix, xyz)

class GammaFitter:
    """
    A class to fit a gamma function to luminance data for monitor calibration.

    This class takes grayscale intensity values and corresponding luminance measurements,
    normalizes the data, and fits a gamma function to estimate the display gamma.
    It also provides tools for visualizing the fit and customizing fit parameters.

    Attributes:
        original_intensities (numpy.ndarray): Original grayscale intensity values.
        original_luminance (numpy.ndarray): Original luminance measurements.
        intensities (numpy.ndarray): Normalized grayscale intensities [0, 1].
        luminance (numpy.ndarray): Normalized luminance values [0, 1].
        params (list): Parameters of the fitted gamma function [a, b, c].
        gamma (float): The gamma value (b) obtained from the fitted curve.
        lower_bounds (list): Lower bounds for curve fitting parameters.
        upper_bounds (list): Upper bounds for curve fitting parameters.
        initial_guess (list): Initial guess for the curve fitting parameters.
    """

    def __init__(self, intensities, luminance):
        """
        Initializes the GammaFitter class.

        Args:
            intensities (list or array): The grayscale intensity values.
            luminance (list or array): The corresponding luminance measurements.
        """
        # Store original values for plotting
        self.original_intensities = np.array(intensities)
        self.original_luminance = np.array(luminance)

        # Normalize input data to the range [0, 1]
        self.intensities = (self.original_intensities - np.min(self.original_intensities)) / (
                    np.max(self.original_intensities) - np.min(self.original_intensities))
        self.luminance = (self.original_luminance - np.min(self.original_luminance)) / (
                    np.max(self.original_luminance) - np.min(self.original_luminance))

        self.params = None
        self.gamma = None

        # Default values for bounds and initial guess based on monitor gamma calibration
        self.lower_bounds = [0, 0, 0]  # Lower bounds: a ≥ 0, b ≥ 1 (physical gamma), c ≥ 0
        self.upper_bounds = [10, 5, 1]  # Upper bounds: a ≤ 10, b ≤ 5, c ≤ 0.2
        self.initial_guess = [0, 1, 0.01]  # Typical monitor gamma: a=1, b=2.2, c=0.01

    def gamma_function(self, x, a, b, c):
        """
        Gamma function to model luminance as a function of intensity.

        Args:
            x (float or array): The normalized intensity values.
            a (float): Scaling factor for luminance.
            b (float): Gamma value (exponent).
            c (float): Offset or baseline luminance.

        Returns:
            float or array: Modeled luminance values.
        """
        return a * (x ** b) + c

    def set_bounds(self, lower_bounds, upper_bounds):
        """
        Set custom lower and upper bounds for the gamma fit parameters.

        Args:
            lower_bounds (list): Lower bounds for the parameters [a, b, c].
            upper_bounds (list): Upper bounds for the parameters [a, b, c].
        """
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def set_initial_guess(self, initial_guess):
        """
        Set a custom initial guess for the gamma fit parameters.

        Args:
            initial_guess (list): Initial guess for the parameters [a, b, c].
        """
        self.initial_guess = initial_guess

    def fit(self):
        """
        Fits the gamma function to the normalized intensity and luminance data.

        Returns:
            list: The fitted parameters [a, b, c], where 'b' is the gamma value.
        """
        bounds = (self.lower_bounds, self.upper_bounds)
        self.params, _ = curve_fit(self.gamma_function, self.intensities, self.luminance, p0=self.initial_guess,
                                   bounds=bounds)
        self.gamma = self.params[1]
        return self.params

    def plot(self):
        """
        Plots the original data and the fitted gamma curve.

        Raises:
            ValueError: If the fit has not been performed before calling this method.
        """
        if self.params is None:
            raise ValueError("Fit the data first using the 'fit' method before plotting.")

        plt.figure(figsize=(8, 6))
        plt.scatter(self.original_intensities, self.original_luminance, label='Original Data', color='blue', alpha=0.6)

        # Generate the fit curve in the normalized range
        x_fit_normalized = np.linspace(0, 1, 100)
        y_fit_normalized = self.gamma_function(x_fit_normalized, *self.params)

        # Convert back to the original scale for plotting
        x_fit_original = x_fit_normalized * (
                    np.max(self.original_intensities) - np.min(self.original_intensities)) + np.min(
            self.original_intensities)
        y_fit_original = y_fit_normalized * (
                    np.max(self.original_luminance) - np.min(self.original_luminance)) + np.min(self.original_luminance)

        plt.plot(x_fit_original, y_fit_original,
                 label=f'Gamma={self.params[1]:.3f}, Base luminance={self.params[2]:.3f}',
                 color='red')
        plt.xlabel('Original Intensity')
        plt.ylabel('Original Luminance')
        plt.title('Gamma Function Fit for Monitor Calibration')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    try:
        spyder = SpyderX()
        print("Performing Dark calibration...")
        spyder.calibrate()
        print("Starting measurements...")
        while True:
            xyz = spyder.measure()
            lms = xyz_to_lms(xyz)
            print(f"XYZ values: {xyz}")
            print(f"LMS values: {lms}")
            time.sleep(2)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    finally:
        if "spyder" in locals():
            spyder.close()
            print("SpyderX closed.")
