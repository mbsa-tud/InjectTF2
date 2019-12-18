#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import unittest
import tensorflow as tf

from inject_tf2.injection_manager import InjectionManager


class TestInjectionManager(unittest.TestCase):
    def setUp(self):
        self.im = InjectionManager()

    def test_get_bit_flipped_value(self):

        a = tf.constant(1.0, dtype=tf.float32)
        b = tf.constant(2.0, dtype=tf.float32)
        c = tf.constant(0.0, dtype=tf.float32)
        d = tf.constant(-1.0, dtype=tf.float32)

        # Check correct flipping
        self.assertNotEqual(a, self.im._get_bit_flipped_value(a))
        self.assertEqual(b, self.im._get_bit_flipped_value(c, 30))
        self.assertEqual(a, self.im._get_bit_flipped_value(d, 31))

        e = tf.constant(-1.0, dtype=tf.float16)

        # Check error raise for dtypes that are not implemented
        self.assertRaises(NotImplementedError, self.im._get_bit_flipped_value, e)


if __name__ == "__main__":
    unittest.main()
