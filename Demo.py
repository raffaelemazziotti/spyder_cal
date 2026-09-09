from cal_lib import SpyderX
from cal_psy import GrayLevels
import numpy as np


with SpyderX() as spyder:
    # On Windows, an explicit DLL remains supported:
    # SpyderX(libusb_path=r"C:\path\to\libusb-1.0.dll")
    gl = GrayLevels(spyder, size=(800, 600), screen=0, fullscr=False)
    try:
        gl.calibrate()
        gammas = list()
        gfit = gl.measure(num_levels=12)
        gammas.append(gfit.gamma)
        for i in range(0, 3):
            gfit = gl.measure(num_levels=12, wait_user=False)
            gammas.append(gfit.gamma)
        # gl.measure(gamma=np.mean(gammas), num_levels=12, wait_user=False)
    finally:
        gl.close()
print(f'Display Gamma avg: {np.mean(gammas)}')
