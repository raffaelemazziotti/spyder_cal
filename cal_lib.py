import usb.core
import usb.util
import usb.backend.libusb1
import numpy as np
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# to run this code on windows10 [other OS not tested] you have to install the usblib drivers
# git clone https://github.com/microsoft/vcpkg.git
# cd vcpkg
# bootstrap-vcpkg.bat
# vcpkg install libusb
# then the library is located in ...\vcpkg\installed\x64-windows\bin\libusb-1.0.dll


class SpyderX:
    """
    A class to interface with the SpyderX colorimeter for monitor calibration.

    This class provides methods for initializing the SpyderX device, performing black calibration,
    measuring luminance values, and extracting factory calibration data. The SpyderX communicates
    through USB, requiring `libusb` as the backend.

    Attributes:
        dev (usb.core.Device): The USB device representing the SpyderX.
        spyderData (dict): A dictionary storing SpyderX calibration and measurement data.
        backend (usb.backend): The USB backend used for communication.
        Code freely modified from: https://github.com/patrickmineault/spyderX
    """

    def __init__(self, libusb_path):
        """
        Initializes the SpyderX class.

        Args:
            libusb_path (str): Full path to the `libusb-1.0.dll` file for Windows.
        """
        self.dev = None
        self.spyderData = {}
        self.backend = usb.backend.libusb1.get_backend(find_library=lambda x: libusb_path)
        if self.backend is None:
            raise ValueError("Libusb backend not found. Check if the path is correct.")

    def initialize(self):
        """
        Initializes the SpyderX device and performs initial USB setup.

        Returns:
            bool: True if the device was successfully initialized, False otherwise.
        """
        try:
            self.dev = usb.core.find(idVendor=0x085C, idProduct=0x0A00, backend=self.backend)
            if self.dev is None:
                print("SpyderX device not found. Is it plugged in?")
                return False

            # Set USB configuration
            try:
                self.dev.set_configuration()
            except NotImplementedError:
                print("Set configuration not supported on this platform.")
            except usb.core.USBError as e:
                print(f"Error setting configuration: {e}")

            # Claim USB interface
            try:
                if self.dev.is_kernel_driver_active(0):
                    self.dev.detach_kernel_driver(0)
                usb.util.claim_interface(self.dev, 0)
            except NotImplementedError:
                print("Claim interface not supported on this platform.")
            except usb.core.USBError as e:
                print(f"Error claiming interface: {e}")

            # Perform setup transfers
            self._control_transfer(0x02, 1, 0, 1, None)
            self._control_transfer(0x02, 1, 0, 129, None)
            self._control_transfer(0x41, 2, 2, 0, None)

            # Retrieve hardware and calibration data
            self._get_hardware_version()
            self._get_serial_number()
            self._get_factory_calibration()
            self._get_amb_measure()
            self._setup_measurement()

            self.spyderData['isOpen'] = True
            return True
        except usb.core.USBError as e:
            print(f"USB error occurred: {str(e)}")
            return False

    def _control_transfer(self, bmRequestType, bRequest, wValue, wIndex, data_or_wLength):
        """
        Performs a USB control transfer.

        Args:
            bmRequestType (int): The request type.
            bRequest (int): The specific request.
            wValue (int): Value parameter for the request.
            wIndex (int): Index parameter for the request.
            data_or_wLength (int or None): Data length or None.

        Returns:
            int or None: Result of the control transfer.
        """
        try:
            return self.dev.ctrl_transfer(bmRequestType, bRequest, wValue, wIndex, data_or_wLength)
        except NotImplementedError:
            print("Control transfer not supported on this platform.")
            return None

    def _bulk_transfer(self, cmd, outSize):
        """
        Performs a USB bulk transfer for sending commands and reading data.

        Args:
            cmd (list): List of bytes to send as a command.
            outSize (int): Number of bytes to read as a response.

        Returns:
            numpy.ndarray: The response data read from the device.
        """
        try:
            self.dev.write(1, cmd)
            return self.dev.read(0x81, outSize)
        except NotImplementedError:
            print("Bulk transfer not supported on this platform.")
            return None

    def _get_hardware_version(self):
        """
        Retrieves the hardware version of the SpyderX device.
        """
        out = self._bulk_transfer([0xd9, 0x42, 0x33, 0x00, 0x00], 28)
        if out is not None:
            self.spyderData['HWvn'] = out[5:9].tobytes().decode()

    def _get_serial_number(self):
        """
        Retrieves the serial number of the SpyderX device.
        """
        out = self._bulk_transfer([0xc2, 0x5c, 0x37, 0x00, 0x00], 42)
        if out is not None:
            self.spyderData['serNo'] = out[9:17].tobytes().decode()

    def _get_factory_calibration(self):
        """
        Retrieves the factory calibration data, including the calibration matrix.
        """
        out = self._bulk_transfer([0xcb, 0x05, 0x73, 0x00, 0x01, 0x00], 47)
        if out is not None:
            out = out[5:]
            matrix = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    k = i * 3 + j
                    matrix[i, j] = self._read_IEEE754(out[k * 4 + 4:k * 4 + 8])
            self.spyderData['calibration'] = {'matrix': matrix}

    @staticmethod
    def _read_nORD_be(input_bytes):
        """
        Reads a big-endian integer from bytes.

        Args:
            input_bytes (bytes): Input bytes.

        Returns:
            int: Big-endian integer value.
        """
        return int.from_bytes(input_bytes, byteorder='big')

    @staticmethod
    def _read_IEEE754(input_bytes):
        """
        Decodes an IEEE-754 floating-point value from bytes.

        Args:
            input_bytes (bytes): Input bytes in little-endian format.

        Returns:
            float: The decoded floating-point value.
        """
        input_bytes = input_bytes[::-1]
        binary = ''.join(f'{byte:08b}' for byte in input_bytes)
        sign = int(binary[0])
        exponent = int(binary[1:9], 2)
        fraction = int(binary[9:], 2) / 2 ** 23
        return (-1) ** sign * (1 + fraction) * 2 ** (exponent - 127)

    def calibrate(self):
        """
        Performs black-level calibration of the SpyderX device.
        """
        if not self.spyderData.get('isOpen', False):
            self.initialize()
        self._control_transfer(0x41, 2, 2, 0, None)

    def measure(self):
        """
        Measures the luminance and color values from the monitor.

        Returns:
            numpy.ndarray: The measured XYZ color values, where Y is luminance.
        """
        self._control_transfer(0x41, 2, 2, 0, None)
        out = self._bulk_transfer([0xd2, 0x3f, 0xb9, 0x00, 0x07], 13)
        if out is not None:
            raw = np.array(struct.unpack('>HHHH', out[5:]))
            XYZ = np.dot(raw[:3], self.spyderData['calibration']['matrix'])
            return XYZ

    def close(self):
        """
        Releases the USB resources and closes the connection to the SpyderX device.
        """
        if self.dev:
            usb.util.dispose_resources(self.dev)
        self.spyderData['isOpen'] = False

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

if __name__=='__main__':
    spyder = SpyderX(r"C:\cancellami\vcpkg\installed\x64-windows\bin\libusb-1.0.dll")
    try:
        print("Initializing SpyderX...")
        if spyder.initialize():
            print("Performing Dark calibration...")
            spyder.calibrate()
            print("Starting measurements...")
            while True:
                xyz = spyder.measure()
                lms = xyz_to_lms(xyz)
                print(f"XYZ values: {xyz}")
                print(f"LMS values: {lms}")
                time.sleep(2)
        else:
            print("Failed to initialize SpyderX. Please check the connection and try again.")
    except KeyboardInterrupt:
        print("\nMeasurement stopped by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
    finally:
        spyder.close()
        print("SpyderX closed.")
