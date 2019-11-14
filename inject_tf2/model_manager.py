#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import logging

import tensorflow as tf


class ModelManager:
    def __init__(self, path_to_model):
        self.org_model = tf.keras.models.load_model(path_to_model)

        self.layer_models = self._create_layer_models(self.org_model)

    # Splits the provided model into its individual layers and
    # creats a model for each layer. That way each layer can be
    # executed separately.
    def _create_layer_models(self, model):

        logging.debug("Creating models for each layer...")

        layer_models = []

        def clone_function(layer):

            logging.debug("Creating model for layer {0}".format(layer))

            layer_models.append(tf.keras.models.Sequential([layer]))
            return layer

        tf.keras.models.clone_model(model=self.org_model, clone_function=clone_function)

        logging.debug(
            "All models created! Resulting layer model list is:\n{0}".format(
                layer_models
            )
        )

        return layer_models

    def get_org_model(self):
        """Returns the original, provided model"""
        return self.org_model

    def get_layer_models(self):
        """Returns a list containing the individual layers of the original model
        as models."""
        return self.layer_models
