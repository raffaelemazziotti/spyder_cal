import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import usb.core

from cal_lib import GammaMeasurement, SpyderX


def make_fit(gamma, luminance):
    return types.SimpleNamespace(
        gamma=gamma,
        original_intensities=[-1.0, 0.0, 1.0],
        original_luminance=luminance,
        params=[1.0, gamma, 0.0],
    )


class FakeDevice:
    def __init__(self, kernel_active=False, kernel_api=True):
        self.kernel_active = kernel_active
        self.set_configuration_calls = 0
        self.detached = []
        self.attached = []
        if not kernel_api:
            self.is_kernel_driver_active = None
            self.detach_kernel_driver = None
            self.attach_kernel_driver = None

    def set_configuration(self):
        self.set_configuration_calls += 1

    def is_kernel_driver_active(self, interface):
        return self.kernel_active

    def detach_kernel_driver(self, interface):
        self.detached.append(interface)

    def attach_kernel_driver(self, interface):
        self.attached.append(interface)


class SpyderXUSBTests(unittest.TestCase):
    def setUp(self):
        self.backend = object()
        self.device = FakeDevice()
        self.backend_patcher = mock.patch(
            "cal_lib.usb.backend.libusb1.get_backend", return_value=self.backend
        )
        self.find_patcher = mock.patch(
            "cal_lib.usb.core.find", return_value=self.device
        )
        self.claim_patcher = mock.patch("cal_lib.usb.util.claim_interface")
        self.release_patcher = mock.patch("cal_lib.usb.util.release_interface")
        self.dispose_patcher = mock.patch("cal_lib.usb.util.dispose_resources")
        self.initialize_patcher = mock.patch.object(SpyderX, "_initialize_device")

        self.get_backend = self.backend_patcher.start()
        self.find = self.find_patcher.start()
        self.claim = self.claim_patcher.start()
        self.release = self.release_patcher.start()
        self.dispose = self.dispose_patcher.start()
        self.initialize = self.initialize_patcher.start()
        self.addCleanup(mock.patch.stopall)

    def test_automatic_backend_discovery(self):
        spyder = SpyderX()

        self.get_backend.assert_called_once_with()
        self.find.assert_called_once_with(
            idVendor=0x085C, idProduct=0x0A00, backend=self.backend
        )
        spyder.close()

    def test_explicit_libusb_path_backend(self):
        path = r"C:\\libusb\\libusb-1.0.dll"
        spyder = SpyderX(libusb_path=path)

        find_library = self.get_backend.call_args.kwargs["find_library"]
        self.assertEqual(find_library("libusb-1.0"), path)
        spyder.close()

    def test_missing_libusb_has_useful_error(self):
        self.get_backend.return_value = None

        with self.assertRaisesRegex(RuntimeError, "libusb 1.x backend"):
            SpyderX()

        self.find.assert_not_called()

    def test_missing_explicit_libusb_has_path_in_error(self):
        self.get_backend.return_value = None

        with self.assertRaisesRegex(RuntimeError, "missing-libusb.dll"):
            SpyderX(libusb_path="missing-libusb.dll")

    def test_device_not_found_has_vid_pid_in_error(self):
        self.find.return_value = None

        with self.assertRaisesRegex(ValueError, "085c:0a00"):
            SpyderX()

    def test_interface_is_claimed_and_released(self):
        spyder = SpyderX()

        self.claim.assert_called_once_with(self.device, 0)
        spyder.close()
        self.release.assert_called_once_with(self.device, 0)
        self.dispose.assert_called_once_with(self.device)

    def test_active_kernel_driver_is_detached_and_reattached(self):
        self.device.kernel_active = True

        spyder = SpyderX()
        self.assertEqual(self.device.detached, [0])

        spyder.close()
        self.assertEqual(self.device.attached, [0])

    def test_inactive_kernel_driver_is_not_reattached(self):
        spyder = SpyderX()
        spyder.close()

        self.assertEqual(self.device.detached, [])
        self.assertEqual(self.device.attached, [])

    def test_missing_kernel_driver_api_is_safe(self):
        self.device = FakeDevice(kernel_api=False)
        self.find.return_value = self.device

        spyder = SpyderX()
        spyder.close()

        self.release.assert_called_once_with(self.device, 0)
        self.dispose.assert_called_once_with(self.device)

    def test_unsupported_kernel_driver_api_is_safe(self):
        self.device.is_kernel_driver_active = mock.Mock(
            side_effect=NotImplementedError
        )

        spyder = SpyderX()
        spyder.close()

        self.claim.assert_called_once_with(self.device, 0)
        self.dispose.assert_called_once_with(self.device)

    def test_context_manager_cleans_up_when_measurement_raises(self):
        with self.assertRaisesRegex(RuntimeError, "measurement failed"):
            with SpyderX() as spyder:
                spyder.measure = mock.Mock(
                    side_effect=RuntimeError("measurement failed")
                )
                spyder.measure()

        self.release.assert_called_once_with(self.device, 0)
        self.dispose.assert_called_once_with(self.device)

    def test_initialization_failure_cleans_up(self):
        self.initialize.side_effect = usb.core.USBError("initialization failed")

        with self.assertRaises(usb.core.USBError):
            SpyderX()

        self.release.assert_called_once_with(self.device, 0)
        self.dispose.assert_called_once_with(self.device)

    def test_claim_failure_reattaches_driver_detached_by_this_instance(self):
        self.device.kernel_active = True
        self.claim.side_effect = usb.core.USBError("claim failed")

        with self.assertRaises(usb.core.USBError):
            SpyderX()

        self.assertEqual(self.device.detached, [0])
        self.assertEqual(self.device.attached, [0])
        self.release.assert_not_called()
        self.dispose.assert_called_once_with(self.device)

    def test_linux_permission_error_explains_udev_setup(self):
        error = usb.core.USBError("access denied")
        error.errno = 13
        self.claim.side_effect = error

        with mock.patch("cal_lib.platform.system", return_value="Linux"):
            with self.assertRaisesRegex(PermissionError, "udev/60-spyderx.rules"):
                SpyderX()

        self.dispose.assert_called_once_with(self.device)

    def test_measure_gamma_runs_repetitions_and_keeps_device_open(self):
        spyder = SpyderX()
        levels = mock.Mock()
        levels.measure.side_effect = [
            make_fit(2.1, [0.1, 20.0, 90.0]),
            make_fit(2.3, [0.3, 25.0, 110.0]),
        ]
        fake_cal_psy = types.ModuleType("cal_psy")
        fake_cal_psy.GrayLevels = mock.Mock(return_value=levels)

        with mock.patch.dict(sys.modules, {"cal_psy": fake_cal_psy}), mock.patch(
            "builtins.print"
        ):
            result = spyder.measure_gamma(
                repetitions=2,
                num_levels=3,
                pause=0.5,
                fullscr=True,
                screen=1,
            )

        fake_cal_psy.GrayLevels.assert_called_once_with(
            spyder,
            fullscr=True,
            screen=1,
            size=None,
            pos=None,
        )
        levels.calibrate.assert_called_once_with()
        self.assertEqual(
            levels.measure.call_args_list,
            [
                mock.call(
                    pause=0.5,
                    num_levels=3,
                    wait_user=True,
                    plot=False,
                ),
                mock.call(
                    pause=0.5,
                    num_levels=3,
                    wait_user=False,
                    plot=False,
                ),
            ],
        )
        levels.close.assert_called_once_with(close_spyder=False)
        self.assertAlmostEqual(result.gamma, 2.2)
        self.assertAlmostEqual(result.luminance_min, 0.2)
        self.assertEqual(result.luminance_max, 100.0)
        self.assertFalse(spyder._closed)
        spyder.close()

    def test_close_is_idempotent(self):
        spyder = SpyderX()

        spyder.close()
        spyder.close()

        self.release.assert_called_once()
        self.dispose.assert_called_once()


class GammaMeasurementTests(unittest.TestCase):
    def test_result_can_be_saved_as_json(self):
        result = GammaMeasurement(
            [
                make_fit(2.1, [0.1, 20.0, 90.0]),
                make_fit(2.3, [0.3, 25.0, 110.0]),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            output = result.save_json(Path(directory) / "gamma.json")
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(saved["schema_version"], 2)
        self.assertAlmostEqual(saved["gamma"], 2.2)
        self.assertEqual(saved["luminance"]["minimum"], 0.2)
        self.assertEqual(saved["luminance"]["maximum"], 100.0)
        self.assertEqual(len(saved["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
