# region import block
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import InjectFaultsInLayer.InjectTFUtil as itfutil

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer, Dropout
from tensorflow.keras import Model

import numpy as np


# endregion


def bit_flip_scalar(dtype, val):
    """Flips one random bit of the input value `val` and returns the updated value."""
    # convert float according to IEEE 754,
    # cast to integer for xor operation,
    # convert back to float
    if dtype == np.float32:

        # select random bit
        bitNum = np.random.randint(0, 32)
        val_bin = int(itfutil.float_to_bin32(val), 2)
        val_bin ^= 1 << bitNum
        val_bin = bin(val_bin)[2:].zfill(32)
        val = itfutil.bin_to_float32(val_bin)

    elif dtype == np.float64:

        # select random bit
        bitNum = np.random.randint(0, 64)
        val_bin = int(itfutil.float_to_bin64(val), 2)
        val_bin ^= 1 << bitNum
        val_bin = bin(val_bin)[2:].zfill(64)
        val = itfutil.bin_to_float64(val_bin)

    else:
        raise NotImplementedError("Bit flip is not supported for dtype: ", dtype)

    return tf.cast(val, dtype)


class LayerWrapperFlatten(Flatten):

    def __init__(self, layer, **kwargs):
        kwargs['autocast']=False
        self.wrapped_layer = layer
        super(LayerWrapperFlatten, self).__init__(input_shape=self.wrapped_layer.input_shape,**kwargs)
        self.compute_output_shape(self.wrapped_layer.input_shape)

    def call(self, inp):
        outputs = self.wrapped_layer.call(inp)
        const_probability = 0.5
        type_of_inputs = self.wrapped_layer.dtype
        result_array = []
        for value in outputs:
            fault_value = 0
            if np.random.rand() > const_probability:
                result_array.append(fault_value)
            else:
                result_array.append(value)
        return inp


class LayerWrapperDense(Dense):

    def __init__(self, layer):
        self.wrapped_layer = layer
        super(LayerWrapperDense, self).__init__(self.wrapped_layer.units, activation=self.wrapped_layer.activation)
        self.compute_output_shape(self.wrapped_layer.input_shape)

    def build(self, input_shape):
        super(LayerWrapperDense, self).build(input_shape)

    def call(self, inp):
        outputs = self.wrapped_layer.call(inp)
        const_probability = 0.5
        type_of_inputs = self.wrapped_layer.dtype
        result_array = []
        for value in outputs:
            fault_value = 0
            if np.random.rand() > const_probability:
                result_array.append(fault_value)
            else:
                result_array.append(value)
        return inp


class LayerWrapperDropout(Dropout):

    def __init__(self, layer):
        self.wrapped_layer = layer
        super(LayerWrapperDropout, self).__init__(self.wrapped_layer.rate)
        self.compute_output_shape(self.wrapped_layer.input_shape)

    def call(self, inp):
        outputs = self.wrapped_layer.call(inp)
        const_probability = 0.5
        type_of_inputs = self.wrapped_layer.dtype
        result_array = []
        for value in outputs:
            fault_value = 0
            if np.random.rand() > const_probability:
                result_array.append(fault_value)
            else:
                result_array.append(value)
        return inp

class LayerWrapper(Layer):

    def __init__(self, layer):
        self.wrapped_layer = layer
        #self.output_shape = self.wrappedLayer.output_shape
        super(LayerWrapper, self).__init__()


    def call(self, inp):
        outputs = self.wrapped_layer.call(inp)
        const_probability = 0.5
        type_of_inputs = self.wrapped_layer.dtype
        result_array = []
        for value in outputs:
            fault_value = 0
            if np.random.rand() > const_probability:
                result_array.append(fault_value)
            else:
                result_array.append(value)
        return inp
