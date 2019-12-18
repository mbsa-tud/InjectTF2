#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import unittest
import tensorflow as tf

from inject_tf2.config_manager import ConfigurationManager
import inject_tf2.string_res as str_res


class TestConfigurationManager(unittest.TestCase):
    def setUp(self):
        self.cm = ConfigurationManager(
            "./tests/configuration_files/config_manager_test.yml"
        )

    def test_get_data(self):

        true_data = {
            str_res.inject_layer_str: {
                str_res.layer_name_str: "test_layer_name",
                str_res.fault_type_str: "test_fault_type",
                str_res.bit_flip_type_str: "RandomBit",
                str_res.bit_number_str: 31,
                str_res.probability_str: 1.0,
            }
        }

        self.assertEqual(true_data, self.cm.get_data())

    def test_get_selected_layer(self):

        true_layer_name = "test_layer_name"

        self.assertEqual(true_layer_name, self.cm.get_selected_layer())


if __name__ == "__main__":
    unittest.main()
