from cal_lib import SpyderX
from cal_psy import GrayLevels
import numpy as np

libusb_path = r"path_to_lib"  # Replace with actual path
spyder = SpyderX(libusb_path)
gl = GrayLevels(spyder)
gl.calibrate()
gammas = list()
gfit = gl.measure(num_levels=12)
gammas.append(gfit.gamma)
for i in range(0,3): # repeat measurements 3 times
    gfit = gl.measure(num_levels=12,wait_user=False)
    gammas.append(gfit.gamma)
#gl.measure(gamma=np.mean(gammas),num_levels=12,wait_user=False) # uncomment only if you have one monitor [on windows10] or is in mirroring mode
gl.close()
print(f'Display Gamma avg: {np.mean(gammas)}') # display average gamma.