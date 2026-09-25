import sys
import types
import unittest
from unittest import mock


fake_psychopy = types.ModuleType("psychopy")
fake_psychopy.visual = mock.Mock()
fake_psychopy.event = mock.Mock()
fake_psychopy.core = mock.Mock()
sys.modules["psychopy"] = fake_psychopy

from cal_psy import GrayLevels


class GrayLevelsWindowTests(unittest.TestCase):
    def setUp(self):
        fake_psychopy.visual.reset_mock()
        self.window = mock.Mock()
        fake_psychopy.visual.Window.return_value = self.window

    def test_default_window_configuration_is_preserved(self):
        GrayLevels(mock.Mock())

        fake_psychopy.visual.Window.assert_called_once_with(
            size=(800, 600),
            pos=None,
            color=[0, 0, 0],
            units="norm",
            waitBlanking=True,
            fullscr=False,
            screen=0,
        )

    def test_window_configuration_is_forwarded(self):
        GrayLevels(
            mock.Mock(),
            size=(1920, 1080),
            pos=(100, 50),
            fullscr=True,
            screen=2,
        )

        fake_psychopy.visual.Window.assert_called_once_with(
            size=(1920, 1080),
            pos=(100, 50),
            color=[0, 0, 0],
            units="norm",
            waitBlanking=True,
            fullscr=True,
            screen=2,
        )

    def test_fullscreen_native_resolution_omits_size_and_position(self):
        GrayLevels(mock.Mock(), size=None, fullscr=True, screen=0)

        fake_psychopy.visual.Window.assert_called_once_with(
            color=[0, 0, 0],
            units="norm",
            waitBlanking=True,
            fullscr=True,
            screen=0,
        )

    def test_measurement_plot_can_be_disabled(self):
        spyder = mock.Mock()
        spyder.measure.side_effect = [
            [0.0, 0.1, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 100.0, 0.0],
        ]
        levels = GrayLevels(spyder)
        fit = mock.Mock(gamma=2.2)

        with mock.patch("cal_psy.GammaFitter", return_value=fit):
            result = levels.measure(
                pause=0,
                num_levels=3,
                wait_user=False,
                plot=False,
            )

        self.assertIs(result, fit)
        fit.fit.assert_called_once_with()
        fit.plot.assert_not_called()

    def test_window_can_close_without_closing_shared_spyder(self):
        spyder = mock.Mock()
        levels = GrayLevels(spyder)

        levels.close(close_spyder=False)

        levels.win.close.assert_called_once_with()
        spyder.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
