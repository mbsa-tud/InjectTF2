#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import unittest
import tensorflow as tf

import inject_tf2.inject_tf2_util as itfutil


class TestInjectTF2Util(unittest.TestCase):
    def test_float_to_bin32(self):
        self.assertEqual(
            itfutil.float_to_bin32(1.0), "00111111100000000000000000000000"
        )

        self.assertEqual(
            itfutil.float_to_bin32(-1.0), "10111111100000000000000000000000"
        )

        self.assertEqual(itfutil.float_to_bin32(0), "00000000000000000000000000000000")

    def test_bin_to_float32(self):
        self.assertEqual(
            itfutil.bin_to_float32("00111111100000000000000000000000"), 1.0
        )

        self.assertEqual(
            itfutil.bin_to_float32("10111111100000000000000000000000"), -1.0
        )

        self.assertEqual(
            itfutil.bin_to_float32("00000000000000000000000000000000"), 0.0
        )

    def test_float_to_bin64(self):
        self.assertEqual(
            itfutil.float_to_bin64(1.0),
            "0011111111110000000000000000000000000000000000000000000000000000",
        )

        self.assertEqual(
            itfutil.float_to_bin64(-1.0),
            "1011111111110000000000000000000000000000000000000000000000000000",
        )

        self.assertEqual(
            itfutil.float_to_bin64(0),
            "0000000000000000000000000000000000000000000000000000000000000000",
        )

    def test_bin_to_float64(self):
        self.assertEqual(
            itfutil.bin_to_float64(
                "0011111111110000000000000000000000000000000000000000000000000000"
            ),
            1.0,
        )

        self.assertEqual(
            itfutil.bin_to_float64(
                "1011111111110000000000000000000000000000000000000000000000000000"
            ),
            -1.0,
        )

        self.assertEqual(
            itfutil.bin_to_float64(
                "0000000000000000000000000000000000000000000000000000000000000000"
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
