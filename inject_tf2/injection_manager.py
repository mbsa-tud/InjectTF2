#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import logging
import numpy as np

import inject_tf2.inject_tf2_util as itfutil
import inject_tf2.string_res as str_res


class InjectionManager:
    def __init__(self):
        logging.debug("initialized InjectionManager")

    def inject_batch(self, batch_output_values, layer_inj_config):

        # Get the probability for fault injection for the selected layer.
        layer_inj_probability = layer_inj_config[str_res.inject_layer_str][
            str_res.probability_str
        ]

        # For each sample in the batch generate a random number and compare it
        # to the specified fault injection probabilty.
        for i, sample in enumerate(batch_output_values):
            random_number = np.random.rand()

            if layer_inj_probability > random_number:

                # Call correct injection function based on the configuration.
                result = self._injected_functions[
                    layer_inj_config[str_res.inject_layer_str][str_res.fault_type_str]
                ](sample, layer_inj_config)

                batch_output_values[i] = result

        return batch_output_values

    @staticmethod
    def _get_bit_flipped_value(element_val, bit_number=None):
        """Flips one random bit of a value and returns it."""

        # convert float according to IEEE 754,
        # cast to integer for xor operation,
        # convert back to float
        if element_val.dtype == np.float32:

            if bit_number is None:
                # No specific bit specified, select random bit
                bit_number = np.random.randint(0, 32)

            element_val_bin = int(itfutil.float_to_bin32(element_val), 2)
            element_val_bin ^= 1 << bit_number
            element_val_bin = bin(element_val_bin)[2:].zfill(32)
            element_val = itfutil.bin_to_float32(element_val_bin)

        elif element_val.dtype == np.float64:

            if bit_number is None:
                # No specific bit specified, select random bit
                bit_number = np.random.randint(0, 64)

            element_val_bin = int(itfutil.float_to_bin64(element_val), 2)
            element_val_bin ^= 1 << bit_number
            element_val_bin = bin(element_val_bin)[2:].zfill(64)
            element_val = itfutil.bin_to_float64(element_val_bin)

        else:
            raise NotImplementedError(
                "Bit flip is not supported for dtype: ", element_val.dtype
            )

        return element_val

    @staticmethod
    def _inject_bit_flip_random_element(tensor, bit_number=None):
        """Flips one bit of a random element of the tensor and returns the updated tensor.

        If `bit_number = None`, a random bit will be flipped. For `bit_number = i`,
        the ith bit will be flipped (zero indexed)."""

        # select random element from tensor by generating
        # random indices based on tensor dimensions
        element = []
        for dimension in tensor.shape:
            element.append(np.random.randint(0, dimension))

        element_val = InjectionManager._get_element_from_tensor(tensor, *element)

        new_element_val = InjectionManager._get_bit_flipped_value(
            element_val, bit_number
        )

        tensor = InjectionManager._set_element_in_tensor(
            new_element_val, tensor, *element
        )
        return tensor

    @staticmethod
    def _inject_bit_flip(tensor, inj_config):
        """Handle the bit flip injection based on the selected bit flip type."""

        # Determine the bit flip type that should be executed
        bf_type = inj_config[str_res.inject_layer_str][str_res.bit_flip_type_str]

        if bf_type == str_res.specific_bit_str:

            # Get the bit number
            bit_num = inj_config[str_res.inject_layer_str][str_res.bit_number_str]
            return InjectionManager._inject_bit_flip_random_element(tensor, bit_num)

        else:
            return InjectionManager._inject_bit_flip_random_element(tensor)

    @staticmethod
    def _get_element_from_tensor(tensor, *e_indices):
        return tensor[e_indices]

    @staticmethod
    def _set_element_in_tensor(value, tensor, *e_indices):
        tensor[e_indices] = value
        return tensor

    _injected_functions = {str_res.bit_flip_str: _inject_bit_flip.__func__}
