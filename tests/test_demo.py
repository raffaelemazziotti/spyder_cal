import unittest

import Demo


class DemoArgumentTests(unittest.TestCase):
    def test_user_friendly_defaults(self):
        args = Demo.parse_args([])

        self.assertEqual(args.screen, 0)
        self.assertEqual(args.repetitions, 3)
        self.assertEqual(args.levels, 12)
        self.assertFalse(args.windowed)
        self.assertEqual(args.output, Demo.DEFAULT_OUTPUT)

    def test_common_options(self):
        args = Demo.parse_args(
            [
                "--screen",
                "1",
                "--repetitions",
                "5",
                "--windowed",
                "--output",
                "result.json",
            ]
        )

        self.assertEqual(args.screen, 1)
        self.assertEqual(args.repetitions, 5)
        self.assertTrue(args.windowed)
        self.assertEqual(str(args.output), "result.json")


if __name__ == "__main__":
    unittest.main()
