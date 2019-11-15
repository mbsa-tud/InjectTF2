#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import inject_tf2.inject_tf2_util as itfutil
import inject_tf2.string_res as str_res
import logging
import numpy as np


class InjectionManager:
    def __init__(self):
        logging.debug("initialized InjectionManager")

    def inject(self, output_val, layer_inj_config):

        # TODO
        # - Generate random number, compare with provided injection probability
        random_number = np.random.rand()
        if layer_inj_config[str_res.probability_str] > random_number:
            # - Call correct injection function based on configuration
            return self._injected_functions[layer_inj_config[str_res.fault_type_str]](
                output_val,
                layer_inj_config
            )

        # - Return default output of layer
        return output_val

    @staticmethod
    def _get_BitFlip_from_value(element_val):
        """Flips one random bit of a value and returns it."""

        # convert float according to IEEE 754,
        # cast to integer for xor operation,
        # convert back to float
        if element_val.dtype == np.float32:

            # select random bit
            bit_num = np.random.randint(0, 32)
            element_val_bin = int(itfutil.float_to_bin32(element_val), 2)
            element_val_bin ^= 1 << bit_num
            element_val_bin = bin(element_val_bin)[2:].zfill(32)
            element_val = itfutil.bin_to_float32(element_val_bin)

        elif element_val.dtype == np.float64:

            # select random bit
            bit_num = np.random.randint(0, 64)
            element_val_bin = int(itfutil.float_to_bin64(element_val), 2)
            element_val_bin ^= 1 << bit_num
            element_val_bin = bin(element_val_bin)[2:].zfill(64)
            element_val = itfutil.bin_to_float64(element_val_bin)

        else:
            raise NotImplementedError("Bit flip is not supported for dtype: ", element_val.dtype)

        return element_val

    @staticmethod
    def _injectBitFlip_all(tensor):
        """Flips one random bit of an each element of the tensor and returns the updated tensor."""
        dimensions = []
        for dimension in tensor.shape:
            dimensions.append(dimension)

        if len(dimensions) == 1:
            # Do BitFlip for each value takes so much time, so just bitflipping one random value in each dimension
            # for i in range(len(tensor)):
                #tensor[i] = InjectionManager._get_BitFlip_from_value(tensor[i])
            index = np.random.randint(0,len(tensor))
            tensor[index] = InjectionManager._get_BitFlip_from_value(tensor[index])
        else:
            for i in range(len(tensor)):
                tensor[i] = InjectionManager._injectBitFlip_all(tensor[i])

        return tensor

    @staticmethod
    def _injectBitFlip_random(tensor):
        """Flips one random bit of a random element of the tensor and returns the updated tensor."""

        # select random element from tensor by generating
        # random indices based on tensor dimensions
        element = []
        for dimension in tensor.shape:
            element.append(np.random.randint(0, dimension))

        element_val = InjectionManager._get_element(tensor, *element)

        new_element_val = InjectionManager._get_BitFlip_from_value(element_val)

        tensor = InjectionManager._set_element(new_element_val, tensor, *element)
        return tensor

    @staticmethod
    def _injectBitFlip(tensor, inj_config):
        """Flips one random bit of a random element of the tensor and returns the updated tensor."""

        if inj_config[str_res.elements_str] == str_res.random_values_str:
            return InjectionManager._injectBitFlip_random(tensor)
        else:
            return InjectionManager._injectBitFlip_all(tensor)

    @staticmethod
    def _random_tensor(tensor, inj_config):
        """Replacement of a tensor value with random value"""
        # The tensor.shape is a tuple, while rand needs linear arguments
        # So we need to unpack the tensor.shape tuples as arguments using *

        # If needs to replace only one random value
        if inj_config[str_res.elements_str] == str_res.random_values_str:
            # Generate random value for injection
            random_value = np.random.rand(1)
            # Inject random_value in random position into random index
            dimensions = []
            for dimension in tensor.shape:
                dimensions.append(np.random.randint(0, dimension))
            injected_result = InjectionManager._set_element(random_value, tensor, *dimensions)
            return injected_result
        # If needs to replace all values with random values
        else:
            return np.random.rand(*tensor.shape)

    @staticmethod
    def _zero_tensor(tensor, inj_config):
        """Replacement of a tensor value with zero"""
        # The tensor.shape is a tuple, while rand needs linear arguments
        # So we need to unpack the tensor.shape tuples as arguments using *

        # If needs to replace with zero only one random value
        if inj_config[str_res.elements_str] == str_res.random_values_str:
            # Inject zero value in random position into random index
            dimensions = []
            for dimension in tensor.shape:
                dimensions.append(np.random.randint(0, dimension))
            injected_result = InjectionManager._set_element(0.0, tensor, *dimensions)
            return injected_result
        # If needs to replace all values with zeros
        else:
            return np.zeros(tensor.shape)

    @staticmethod
    def _get_element(tens, *e_indices):
        return tens[e_indices]

    @staticmethod
    def _set_element(val, tens, *e_indices):
        tens[e_indices] = val
        return tens

    _injected_functions = {
        str_res.bit_flip_str: _injectBitFlip.__func__,
        str_res.random_str: _random_tensor.__func__,
        str_res.zero_str: _zero_tensor.__func__,
    }
