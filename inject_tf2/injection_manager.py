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
import time
from itertools import product


class InjectionManager:
    def __init__(self):
        logging.debug("initialized InjectionManager")

    def inject(self, output_val, layer_inj_config):

        # TODO
        # - Generate random number, compare with provided injection probability
        # - Coin flip was replaced to injection function to be able to inject for each dimension
        #random_number = np.random.rand()
        #if layer_inj_config[str_res.probability_str] > random_number:
            # - Call correct injection function based on configuration
        return self._injected_functions[layer_inj_config[str_res.fault_type_str]](
                output_val,
                layer_inj_config
            )

        # - Return default output of layer
        #return output_val

    def custom_inject(self, output_val, probability):

        # TODO
        # - Generate random number, compare with provided injection probability
        random_number = np.random.rand()
        if probability > random_number:
            # - Call correct injection function based on configuration
            return self._injectBitFlip_random_each_dimension(output_val)

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
        # """Flips one random bit of an each element of the tensor and returns the updated tensor."""
        dimensions = []
        for dimension in tensor.shape:
            dimensions.append(dimension)

        if len(dimensions) == 1:
            # Do BitFlip for each value takes so much time, so just bitflipping one random value in each dimension
            # time_start = time.time()
            for i in range(len(tensor)):
                tensor[i] = InjectionManager._get_BitFlip_from_value(tensor[i])
            # time_end = time.time()
            # print("\n Time diff = ", time_end - time_start)
            # index = np.random.randint(0,len(tensor))
            # tensor[index] = InjectionManager._get_BitFlip_from_value(tensor[index])
        else:
            for i in range(len(tensor)):
                tensor[i] = InjectionManager._injectBitFlip_all(tensor[i])

        return tensor

    @staticmethod
    def _injectBitFlip_random_each_dimension(tensor, layer_inj_config):
        """Flips one random bit of a random element of the tensor and returns the updated tensor."""
        for i in range(tensor.shape[0]):
            # select random element from tensor by generating
            # random indices based on tensor dimensions
            element = [i]
            for dimension in tensor.shape[1:]:
                element.append(np.random.randint(0, dimension))

            random_number = np.random.rand()
            if layer_inj_config[str_res.probability_str] > random_number:
                element_val = InjectionManager._get_element(tensor, *element)
                new_element_val = InjectionManager._get_BitFlip_from_value(element_val)
                tensor = InjectionManager._set_element(new_element_val, tensor, *element)

        return tensor

    @staticmethod
    def _injectBitFlip(tensor, inj_config):
        """Flips one random bit of a random element of the tensor and returns the updated tensor."""

        if inj_config[str_res.elements_str] == str_res.random_values_str:
            return InjectionManager._injectBitFlip_random_each_dimension(tensor, inj_config)
        else:
            return InjectionManager._injectBitFlip_all(tensor, inj_config)

    @staticmethod
    def _random_tensor(tensor, layer_inj_config):
        """Replacement of a tensor value with random value"""
        # The tensor.shape is a tuple, while rand needs linear arguments
        # So we need to unpack the tensor.shape tuples as arguments using *

        # If needs to replace only one random value
        if layer_inj_config[str_res.elements_str] == str_res.random_values_str:
            for i in range(tensor.shape[0]):
                # Generate random value for injection

                # Inject random_value in random position into random index
                dimensions = [i]
                for dimension in tensor.shape[1:]:
                    dimensions.append(np.random.randint(0, dimension))

                if layer_inj_config[str_res.probability_str] > np.random.rand():
                    random_value = np.random.rand(1)
                    tensor = InjectionManager._set_element(random_value, tensor, *dimensions)

        # If needs to replace all values with random values
        else:
            tensor = np.random.rand(*tensor.shape)
        return tensor

    @staticmethod
    def _zero_tensor(tensor, layer_inj_config):
        """Replacement of a tensor value with zero"""
        # The tensor.shape is a tuple, while rand needs linear arguments
        # So we need to unpack the tensor.shape tuples as arguments using *

        # If needs to replace with zero only one random value
        if layer_inj_config[str_res.elements_str] == str_res.random_values_str:
            for i in range(tensor.shape[0]):
                # Inject zero value in random position into random index
                dimensions = [i]
                for dimension in tensor.shape:
                    dimensions.append(np.random.randint(0, dimension))

                if layer_inj_config[str_res.probability_str] > np.random.rand():
                    tensor = InjectionManager._set_element(0.0, tensor, *dimensions)
        # If needs to replace all values with zeros
        else:
            tensor = np.zeros(tensor.shape)
        return tensor

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
