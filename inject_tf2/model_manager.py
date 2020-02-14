#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K


class ModelManager:
    def __init__(self, path_to_model, layer_name, batched_tf_dataset, batch_size):
        self._org_model = tf.keras.models.load_model(path_to_model)

        self._selected_layer = self._org_model.get_layer(layer_name)

        self._layer_output = self._get_output_values_for_selected_layer(
            batched_tf_dataset, batch_size
        )

    def _get_output_values_for_selected_layer(self, batched_tf_dataset, batch_size):

        # Get the output tensor for the selected layer.
        layer_output_tensor = self._selected_layer.output

        # Create a function which evaluates the output tensor for
        # the selected layer and returns its values
        functor = K.function(self._org_model.inputs, layer_output_tensor)

        # Allocate memory for the output values
        num_of_batches = 0
        for _ in batched_tf_dataset:
            num_of_batches += 1

        # Determine the dataset's dtype.
        ds_dtype =  batched_tf_dataset.element_spec[0].dtype.as_numpy_dtype

        layer_output_values = np.zeros(
            (num_of_batches, batch_size, *layer_output_tensor.shape[1:]), dtype=ds_dtype)

        image_names_for_output_values = []

        # Get the output values for the input data.
        for i, batch in enumerate(batched_tf_dataset):
            layer_output_values[i] = functor(batch)
            image_names_for_output_values.append(batch[1])

        return (layer_output_values, images_names_for_output_values)

    def get_org_model(self):
        """Returns the provided model"""
        return self._org_model

    def get_selected_layer_output_values(self):
        """Returns the output values for the selected layer
        for the provided test data."""
        return self._layer_output[1]

    def get_selected_layer_output(self):
        """Returns the output values and corresponding image names
        for the selected layer for the provided test data."""
        return self._layer_output

    def predict_func_from_layer(self):
        """Returns a prediction function which feed its input argument into the
        selected layer and returns the prediction of the model."""
        return K.function(self._selected_layer.output, self._org_model.outputs)
