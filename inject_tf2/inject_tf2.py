#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

from inject_tf2.injection_manager import InjectionManager
from inject_tf2.model_manager import ModelManager
from inject_tf2.config_manager import ConfigurationManager
import os
import logging
import tensorflow as tf
import time


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

        # TODO
        # Validate user inputs

        self.golden_run = self._execute_golden_run(input_data)
        self.golden_run_layers = self._execute_golden_run_layers(input_data)

    def _execute_golden_run(self, data):
        logging.debug("Model summary {0}".format(self.mm.get_org_model().summary()))
        return self.mm.get_org_model().predict(data)

    def _execute_golden_run_layers(self, data):
        start_time = time.time()
        result = []

        for i, layer_model in enumerate(self.mm.get_layer_models()):

            # First iteration. Pass the dataset into the first layer.
            if not result:
                result.append(layer_model.predict(data))

            # Use the output values from the previous layer
            else:
                result.append(layer_model.predict(result[i - 1]))

        finish_time = time.time()
        self.time_golden_layers = finish_time - start_time

        return result

    def run_experiments(self, batch_size = 0, offset = 0):

        results = []
        # if noone layer was injected before
        first_inject = True
        for i, layer in enumerate(self.mm.get_layer_models()):

            if self.cm.is_selected_for_inj(i + 1, layer):

                #
                first_inject = False

                # Compute output of layer
                # If first layer
                if i == 0:
                    output_val = layer.predict(self.input_data)

                else:
                    output_val = layer.predict(results[i - 1])

                inj_res = self.im.inject(
                    output_val, self.cm.get_config_for_layer(i + 1, layer)
                )

                results.append(inj_res)

            else:

                # take results of golden run instead of executing layer another time for better performance
                if first_inject:
                    if batch_size == 0 and offset == 0:
                        results.append(self.golden_run_layers[i])
                    else:
                        results.append(self.golden_run_layers[i][offset: batch_size + offset])
                    continue
                # If first layer
                results.append(layer.predict(results[i - 1]))

        return results


