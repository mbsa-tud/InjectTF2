#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import logging

import tensorflow as tf
import numpy as np

from inject_tf2.config_manager import ConfigurationManager
from inject_tf2.model_manager import ModelManager
import inject_tf2.inject_tf2_util as itfutil


class InjectionManager:
    def __init__(self):
        logging.debug("initialized InjectionManager")

    def inject(self, output_val, layer_inj_config):

        # TODO
        # - Generate random number, compare with provided injection probability
        # - Call correct injection function based on configuration
        return self._random_tensor(output_val)  # self._injectBitFlip(output_val, "")

    def _injectBitFlip(self, tensor, inj_config):
        """Flips one random bit of a random element of the tensor and returns the updated tensor."""

        # select random element from tensor by generating
        # random indices based on tensor dimensions
        element = []
        for dimension in tensor.shape:
            element.append(np.random.randint(0, dimension))

        def get_element(tens, *e_indices):
            return tens[e_indices]

        def set_element(val, tens, *e_indices):
            tens[e_indices] = val
            return tens

        element_val = get_element(tensor, *element)

        # convert float according to IEEE 754,
        # cast to integer for xor operation,
        # convert back to float
        if tensor.dtype == np.float32:

            # select random bit
            bit_num = np.random.randint(0, 32)
            element_val_bin = int(itfutil.float_to_bin32(element_val), 2)
            element_val_bin ^= 1 << bit_num
            element_val_bin = bin(element_val_bin)[2:].zfill(32)
            element_val = itfutil.bin_to_float32(element_val_bin)

        elif tensor.dtype == np.float64:

            # select random bit
            bit_num = np.random.randint(0, 64)
            element_val_bin = int(itfutil.float_to_bin64(element_val), 2)
            element_val_bin ^= 1 << bit_num
            element_val_bin = bin(element_val_bin)[2:].zfill(64)
            element_val = itfutil.bin_to_float64(element_val_bin)

        else:
            raise NotImplementedError("Bit flip is not supported for dtype: ", dtype)

        tensor = set_element(element_val, tensor, *element)
        return tensor

    def _random_tensor(self, tensor):
        """Random replacement of a tensor value with another one"""
        # The tensor.shape is a tuple, while rand needs linear arguments
        # So we need to unpack the tensor.shape tuples as arguments using *
        return np.random.rand(*tensor.shape)

    _injected_functions = {"BitFlip": _injectBitFlip}
