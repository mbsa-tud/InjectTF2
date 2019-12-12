#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import logging

import tensorflow as tf
from tensorflow.python.keras import backend as K


class ModelManager:
    def __init__(self, path_to_model, batched_tf_dataset, batch_size):
        self.org_model = tf.keras.models.load_model(path_to_model)

        self.layer_output_values = self._get_layer_output_values_for_layer(
            self.org_model, layer_name, batched_tf_dataset, batch_size
        )

    def _get_layer_output_values_for_layer(
        self, model, layer_name, batched_tf_dataset, batch_size
    ):

        # Get the output tensor for the selected layer.
        layer_output_tensor = model.get_layer(layer_name).output

        # Create a function which evaluates the output tensor for
        # the selected layer and returns its values
        functor = K.function(model.inputs, layer_output_tensor)

        # Allocate memory for the output values
        num_of_batches = 0
        for _ in batched_tf_dataset:
            num_of_batches += 1

        layer_output_values = np.zeros(
            (num_of_batches, batch_size, *layer_output_tensor.shape[1:])
        )

        # Get the output values for the input data.
        for i, batch in enumerate(batched_tf_dataset):
            layer_output_values[i] = functor(batch)

        return layer_output_values

    def get_org_model(self):
        """Returns the provided model"""
        return self.org_model

    def get_layer_output_tensors_and_values(self):
        """Returns a list containing the output tensors for each layer and the
        layers output for the provided test data."""
        return self.layer_output_values
