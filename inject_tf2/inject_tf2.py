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
        self,
        path_to_model,
        path_to_config,
        tf_dataset,
        batch_size,
        logging_level="ERROR",
    ):

        # Setup logging.
        logging.basicConfig()
        logging.getLogger().setLevel(logging_level)
        logging.debug("Logging level set to {0}".format(logging_level))

        self.im = InjectionManager()
        self.cm = ConfigurationManager(path_to_config)

        # Create a batched dataset. Since the values for the layer that is selected
        # for injection are stored later on, `drop_remainder` is set to `True`
        # (that way a uniform numpy can be pre-allocated).
        self.batched_ds = tf_dataset.batch(batch_size, drop_remainder=True)

        self.mm = ModelManager(
            path_to_model, self.cm.get_selected_layer(), self.batched_ds, batch_size
        )

        self.golden_run = self._execute_golden_run(self.batched_ds)

    def _execute_golden_run(self, batched_tf_dataset):
        return self.mm.get_org_model().evaluate(batched_tf_dataset)

    # TODO Refactor, DEPRECATED
    def run_experiments(self, input_data):

        # results = []

        # for i, layer in enumerate(self.mm.get_layer_models()):

        #     if self.cm.is_selected_for_inj(i + 1, layer):

        #         # Compute output of layer
        #         # If first layer
        #         if i == 0:
        #             output_val = layer.predict(input_data)

        #         else:
        #             output_val = layer.predict(results[i - 1])

        #         inj_res = self.im.inject(
        #             output_val, self.cm.get_config_for_layer(i + 1, layer)
        #         )

        #         results.append(inj_res)

        #     else:

        #         # If first layer
        #         if i == 0:
        #             results.append(layer.predict(input_data))

        #         else:
        #             results.append(layer.predict(results[i - 1]))

        # return results
        return
