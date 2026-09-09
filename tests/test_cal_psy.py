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


if __name__ == "__main__":
    unittest.main()
