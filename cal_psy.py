from psychopy import visual, event, core
import numpy as np
from cal_lib import GammaFitter

class GrayLevels:
    """
    A class to measure and calibrate monitor gamma using a SpyderX colorimeter and PsychoPy.

    This class allows calibration of the SpyderX device, measurement of luminance levels for
    various grayscale levels, and fitting of the gamma curve using the provided luminance data.

    Attributes:
        spyder (object): An instance of the SpyderX class used for photometric measurements.
        win (psychopy.visual.Window): The PsychoPy window for displaying gray levels.
        bg_rect (psychopy.visual.Rect): A full-screen rectangle used to simulate background color.
    """

    def __init__(self, spyder, fullscr=False):
        """
        Initializes the GrayLevels class.

        Args:
            spyder (SpyderX): An initialized SpyderX object for luminance measurements.
            fullscr (bool): Whether to open the PsychoPy window in full-screen mode.
                            Defaults to False.
        """
        self.spyder = spyder
        self.win = visual.Window([800, 600], color=[0, 0, 0], units="norm", waitBlanking=True, fullscr=fullscr)
        self.bg_rect = visual.Rect(self.win, width=2, height=2, fillColor=[0, 0, 0], lineColor=None)
        self.bg_rect.draw()

    def calibrate(self):
        """
        Calibrates the SpyderX device to measure the baseline luminance.

        Displays instructions for the user to close the SpyderX sensor to perform a black
        level calibration and waits for user confirmation before proceeding.
        """
        print('### GRAYLEVELS ### Spyder calibration', end=' ')
        instruction = visual.TextStim(self.win,
                                      text="Close the SpyderX to measure baseline and press space to start.",
                                      color=[1, 1, 1])
        self.bg_rect.draw()
        instruction.draw()
        self.win.flip()
        event.waitKeys(keyList=["space"])

        self.spyder.calibrate()
        print('DONE')

    def measure(self, pause=1, gamma=None, num_levels=12,wait_user=True):
        """
        Measures luminance levels across a range of grayscale values.

        Displays a series of gray levels on the monitor and uses the SpyderX device to measure
        the corresponding luminance values. The gamma curve is then fitted using the GammaFitter class.

        Args:
            pause (float): The time (in seconds) to pause after each gray level display to ensure stabilization.
                           Defaults to 1 second.
            gamma (float, optional): A predefined gamma value to set for the monitor (for testing the linearity of the monitor after correction). If None, gamma remains unchanged.
                                     Defaults to None. This parameter works on Windows only with one monitor or in mirror mode.
            num_levels (int): The number of gray levels to display and measure. Defaults to 12.
            wait_user (bool): wait for keypress to start

        Returns:
            GammaFitter: An instance of the GammaFitter class containing the gamma value and the fit result.
        """
        if gamma is not None:
            self.win.setGamma(gamma)
        gray_levels = np.linspace(-1, 1, num_levels)
        luminance_readings = []

        if wait_user:
            instruction = visual.TextStim(self.win,
                                          text="Position the SpyderX on the monitor and press space to start.",
                                          color=[1, 1, 1])
            self.bg_rect.fillColor = [-1, -1, -1]
            self.bg_rect.draw()
            instruction.draw()
            self.win.flip()
            event.waitKeys(keyList=["space"])
        self.bg_rect.fillColor = [-1, -1, -1]
        self.bg_rect.draw()
        self.win.flip()
        core.wait(pause)

        for gray in gray_levels:
            self.bg_rect.fillColor = [gray, gray, gray]
            self.bg_rect.draw()
            self.win.flip()
            core.wait(pause)  # Wait for the screen to stabilize

            XYZ = self.spyder.measure()
            luminance = XYZ[1]  # Y component of XYZ is luminance
            luminance_readings.append(luminance)
            print(f"### GRAYLEVELS ### The luminance for gray level {gray:.3f} is {luminance:.3f}")

        gfit = GammaFitter(gray_levels, luminance_readings)
        gfit.fit()
        print(f"### GRAYLEVELS ### Monitor Gamma value: {gfit.gamma}")
        gfit.plot()

        if gamma is not None:
            # Reset monitor gamma to default
            self.win.setGamma(1)
        return gfit

    def close(self):
        """
        Closes the PsychoPy window and releases the SpyderX device.

        Ensures that all resources are properly cleaned up.
        """
        self.win.close()
        self.spyder.close()

if __name__ == '__main__':
    from cal_lib import SpyderX
    libusb_path = r"C:\cancellami\vcpkg\installed\x64-windows\bin\libusb-1.0.dll"  # Replace with actual path
    spyder = SpyderX(libusb_path)
    gl = GrayLevels(spyder)
    gl.calibrate()
    gammas = list()
    gfit = gl.measure(num_levels=12)
    gammas.append(gfit.gamma)
    for i in range(0,3):
        gfit = gl.measure(num_levels=12,wait_user=False)
        gammas.append(gfit.gamma)
    #gl.measure(gamma=np.mean(gammas),num_levels=12,wait_user=False)
    gl.close()
    print(f'Display Gamma avg: {np.mean(gammas)}')