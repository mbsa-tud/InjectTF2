#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import unittest

import tensorflow as tf
import numpy as np

from inject_tf2.injection_manager import InjectionManager
from inject_tf2.config_manager import ConfigurationManager


class TestInjectionManager(unittest.TestCase):
    def setUp(self):
        self.im = InjectionManager()
        self.cm_random_bit = ConfigurationManager(
            "./tests/configuration_files/injection_manager_random_bit_test.yml"
        )
        self.cm_specific_bit = ConfigurationManager(
            "./tests/configuration_files/injection_manager_specific_bit_test.yml"
        )

    def test_inject_batch(self):

        # Simulated layer output: a batch of 32 samples with the dimension 32x32x3
        test_arr = np.zeros((32, 32, 32, 3))

        test_config_random = self.cm_random_bit.get_data()
        test_config_specific = self.cm_specific_bit.get_data()

        inj_test_arr_random = self.im.inject_batch(test_arr, test_config_random)
        inj_test_arr_specific = self.im.inject_batch(test_arr, test_config_specific)

        # Values should be different after injection.
        self.assertFalse(np.array_equal(test_arr, inj_test_arr_random))
        self.assertFalse(np.array_equal(test_arr, inj_test_arr_specific))

        # Each sample in the batch should be injected, therefore there should
        # be test_arr.shape[0] non-zero elements.
        self.assertEqual(test_arr.shape[0], np.count_nonzero(inj_test_arr_specific))
        print(np.amax(inj_test_arr_specific))

        # Each sample should only have one non-zero element
        for sample in inj_test_arr_specific:
            self.assertEqual(1, np.count_nonzero(sample))

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

    def test_inject_bit_flip(self):

        test_config_random = self.cm_random_bit.get_data()
        test_config_specific = self.cm_specific_bit.get_data()

        a = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        b = np.zeros((2, 2), dtype=np.float32)

        # Inject `a`.
        # np.copy is necessary here, otherwise the original `a` array will be modified.
        # During execution of the program the function `inject_batch` will create the
        # copy of the input values that should be injected.
        a_random = self.im._inject_bit_flip(np.copy(a), test_config_random)
        a_specific = self.im._inject_bit_flip(np.copy(a), test_config_specific)
        b_specific = self.im._inject_bit_flip(np.copy(b), test_config_specific)

        self.assertFalse(np.array_equal(a, a_random))
        self.assertFalse(np.array_equal(a, a_specific))

        # Flipping the 31 bit of `0` for a 32 bit float results in 2.0.
        self.assertEqual(2.0, np.amax(b_specific))

        # Only one element in the array should be equal to 2.0
        self.assertEqual(1, np.count_nonzero(a_specific.flatten() == 2.0))


if __name__ == "__main__":
    unittest.main()
