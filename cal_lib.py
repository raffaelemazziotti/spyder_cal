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
    # code modified from https://github.com/patrickmineault/spyderX
    def __init__(self, libusb_path):
        """
        Initializes the SpyderX class.

        Args:
            libusb_path (str): Full path to the libusb-1.0.dll file.
        """
        self.dev = None
        self.spyderData = {}
        self.backend = usb.backend.libusb1.get_backend(find_library=lambda x: libusb_path)
        if self.backend is None:
            raise ValueError("Libusb backend not found. Check if the path is correct.")

    def initialize(self):
        try:
            self.dev = usb.core.find(idVendor=0x085C, idProduct=0x0A00, backend=self.backend)
            if self.dev is None:
                print("SpyderX device not found. Is it plugged in?")
                return False

            # Try to set configuration
            try:
                self.dev.set_configuration()
            except NotImplementedError:
                print("Set configuration not supported on this platform.")
            except usb.core.USBError as e:
                print(f"Error setting configuration: {e}")

            # Try to claim interface
            try:
                if self.dev.is_kernel_driver_active(0):
                    self.dev.detach_kernel_driver(0)
                usb.util.claim_interface(self.dev, 0)
            except NotImplementedError:
                print("Claim interface not supported on this platform.")
            except usb.core.USBError as e:
                print(f"Error claiming interface: {e}")

            self._control_transfer(0x02, 1, 0, 1, None)
            self._control_transfer(0x02, 1, 0, 129, None)
            self._control_transfer(0x41, 2, 2, 0, None)

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
        try:
            return self.dev.ctrl_transfer(bmRequestType, bRequest, wValue, wIndex, data_or_wLength)
        except NotImplementedError:
            print("Control transfer not supported on this platform.")
            return None

    def _bulk_transfer(self, cmd, outSize):
        try:
            self.dev.write(1, cmd)
            return self.dev.read(0x81, outSize)
        except NotImplementedError:
            print("Bulk transfer not supported on this platform.")
            return None

    def _get_hardware_version(self):
        out = self._bulk_transfer([0xd9, 0x42, 0x33, 0x00, 0x00], 28)
        if out is not None:
            self.spyderData['HWvn'] = out[5:9].tobytes().decode()

    def _get_serial_number(self):
        out = self._bulk_transfer([0xc2, 0x5c, 0x37, 0x00, 0x00], 42)
        if out is not None:
            self.spyderData['serNo'] = out[9:17].tobytes().decode()

    def _get_factory_calibration(self):
        out = self._bulk_transfer([0xcb, 0x05, 0x73, 0x00, 0x01, 0x00], 47)
        if out is not None:
            print(f"Factory calibration raw data: {out}")
            out = out[5:]  # Remove first 5 bytes as in MATLAB code

            matrix = np.zeros((3, 3))
            v1 = out[1]  # MATLAB uses 1-based indexing, so this is correct
            v2 = self._read_nORD_be(out[2:4])
            v3 = out[40]  # 41 in MATLAB, but 40 in 0-based Python indexing

            for i in range(3):
                for j in range(3):
                    k = i * 3 + j
                    matrix[i, j] = self._read_IEEE754(out[k * 4 + 4:k * 4 + 8])  # +4 because MATLAB starts at 5

            self.spyderData['calibration'] = {
                'matrix': matrix,
                'v1': v1,
                'v2': v2,
                'v3': v3,
                'ccmat': np.eye(3)  # This is diag([1 1 1]) in MATLAB
            }
            #print(f"Calibration data: {self.spyderData['calibration']}")

    @staticmethod
    def _read_nORD_be(input_bytes):
        return int.from_bytes(input_bytes, byteorder='big')

    @staticmethod
    def _read_IEEE754(input_bytes):
        # Reverse the byte order as in MATLAB code
        input_bytes = input_bytes[::-1]

        # Convert to binary string
        binary = ''.join(f'{byte:08b}' for byte in input_bytes)

        sign = int(binary[0])
        exponent = int(binary[1:9], 2)
        fraction = int(binary[9:], 2) / 2 ** 23

        return (-1) ** sign * (1 + fraction) * 2 ** (exponent - 127)

    def _get_amb_measure(self):
        out = self._bulk_transfer([0xd4, 0xa1, 0xc5, 0x00, 0x02, 0x65, 0x10], 11)
        if out is not None:
            self.spyderData['amb'] = struct.unpack('>HHBB', out[5:])

    def _setup_measurement(self):
        out = self._bulk_transfer([0xc3, 0x29, 0x27, 0x00, 0x01, self.spyderData['calibration']['v1']], 15)
        if out is not None:
            self.spyderData['settUp'] = {
                's1': out[5],
                's2': out[6:10],
                's3': out[10:14]
            }

    def calibrate(self):
        if not self.spyderData.get('isOpen', False):
            self.initialize()

        self._control_transfer(0x41, 2, 2, 0, None)
        v2 = self.spyderData['calibration']['v2']
        s1 = self.spyderData['settUp']['s1']
        s2 = self.spyderData['settUp']['s2']

        send = bytes([v2 >> 8, v2 & 0xFF, s1] + list(s2))
        out = self._bulk_transfer([0xd2, 0x3f, 0xb9, 0x00, 0x07] + list(send), 13)
        if out is not None:
            raw = struct.unpack('>HHHH', out[5:])
            self.spyderData['bcal'] = np.array(raw[:3]) - np.array(self.spyderData['settUp']['s3'][:3])
            self.spyderData['isBlackCal'] = True

    def measure(self):
        if not self.spyderData.get('isOpen', False):
            raise ValueError("SpyderX not initialized")
        if not self.spyderData.get('isBlackCal', False):
            raise ValueError("Black calibration not performed")

        self._control_transfer(0x41, 2, 2, 0, None)
        v2 = self.spyderData['calibration']['v2']
        s1 = self.spyderData['settUp']['s1']
        s2 = self.spyderData['settUp']['s2']

        send = bytes([v2 >> 8, v2 & 0xFF, s1] + list(s2))
        out = self._bulk_transfer([0xd2, 0x3f, 0xb9, 0x00, 0x07] + list(send), 13)
        if out is not None:
            #print(out)
            raw = np.array(struct.unpack('>HHHH', out[5:]))
            #print(raw)

            raw[:3] = raw[:3] - np.array(self.spyderData['settUp']['s3'][:3]) - self.spyderData['bcal']
            #print(raw[:3])
            #print(self.spyderData['calibration']['matrix'])
            XYZ = np.dot(raw[:3], self.spyderData['calibration']['matrix'])
            return XYZ

    def close(self):
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

    def __init__(self, intensities, luminance):
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
        return a * (x ** b) + c

    def set_bounds(self, lower_bounds, upper_bounds):
        """Set custom lower and upper bounds for the gamma fit."""
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def set_initial_guess(self, initial_guess):
        """Set custom initial guess for the gamma fit."""
        self.initial_guess = initial_guess

    def fit(self):
        # Fit the gamma function to the normalized data with specified bounds and initial guess
        bounds = (self.lower_bounds, self.upper_bounds)
        self.params, _ = curve_fit(self.gamma_function, self.intensities, self.luminance, p0=self.initial_guess,
                                   bounds=bounds)
        self.gamma = self.params[1]
        return self.params

    def plot(self):
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
