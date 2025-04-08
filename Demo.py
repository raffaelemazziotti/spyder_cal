from cal_lib import SpyderX
from cal_psy import GrayLevels


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