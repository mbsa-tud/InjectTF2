#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import os
import logging

import tensorflow as tf

from inject_tf2.injection_manager import InjectionManager
from inject_tf2.model_manager import ModelManager
from inject_tf2.config_manager import ConfigurationManager


class InjectTF2:
    def __init__(
        self, path_to_model, path_to_config, input_data, logging_level="ERROR"
    ):

        # Setup logging
        logging.basicConfig()
        logging.getLogger().setLevel(logging_level)
        logging.debug("Logging level set to {0}".format(logging_level))

        self.im = InjectionManager()
        self.mm = ModelManager(path_to_model)
        self.cm = ConfigurationManager(path_to_config)

        self.input_data = input_data

        self.golden_run = self._execute_golden_run(input_data)
        self.golden_run_layers = self._execute_golden_run_layers(input_data)

    def _execute_golden_run(self, data):
        return self.mm.get_org_model().predict(data)

    def _execute_golden_run_layers(self, data):
        result = []

        for i, layer_model in enumerate(self.mm.get_layer_models()):

            # First iteration. Pass the dataset into the first layer.
            if not result:
                result.append(layer_model.predict(data))

            # Use the output values from the previous layer
            else:
                result.append(layer_model.predict(result[i - 1]))

        return result

    def run_experiments(self, input_data):

        # selected_layers = self.cm.get_selected_layers()
        #
        # logging.debug(
        #     "The following layers are selected for injection: {0}".format(
        #         selected_layers
        #     )
        # )

        results = []

        for i, layer in enumerate(self.mm.get_layer_models()):

            if self.cm.is_selected_for_inj(i + 1, layer):

                # Compute output of layer
                # If first layer
                if i == 0:
                    output_val = layer.predict(input_data)

                else:
                    output_val = layer.predict(results[i - 1])

                inj_res = self.im.inject(
                    output_val, self.cm.get_config_for_layer(i + 1, layer)
                )

                results.append(inj_res)

            else:

                # If first layer
                if i == 0:
                    results.append(layer.predict(input_data))

                else:
                    results.append(layer.predict(results[i - 1]))

        return results
