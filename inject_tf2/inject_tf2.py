#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import os
import logging

import numpy as np
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

    def _calc_accuracy_for_batch(self, pred_batch, truth_batch):

        pred_batch = np.argmax(pred_batch, axis=1)

        # Check for one-hot encoded ground truth labels.
        if len(truth_batch.shape) > 1:
            truth_batch = np.argmax(truth_batch, axis=1)

        ret = np.zeros(pred_batch.shape, np.float)
        correct_pred = np.equal(pred_batch, truth_batch, out=ret)
        accuracy = np.mean(correct_pred)
        return accuracy

    def run_experiments(self):

        logging.info("Starting experiment...")

        # Compute reference "golden run" `accuracy` without fault injection
        # and the `inj_accuracy` for the predictions obtained when executing
        # the model from the selected layer onward.
        accuracy = 0
        inj_accuracy = 0

        for input_values, batch in zip(
            self.mm.get_selected_layer_output_values(), self.batched_ds
        ):

            # Get the prediction function for the selected layer.
            predict = self.mm.predict_func_from_layer()

            # `predict()` returns the predictions for the current
            # `input_values` batch wrapped in a list.
            result = predict(input_values)

            # Predict again with (possibly) fault injected values
            inj_result = predict(self.im.inject_batch(input_values, self.cm.get_data()))

            # Caluculate accuracy and inj_accuracy for current batch.
            # result[0]: array containing the predictions
            # inj_result[0]: array containing the injected predictions
            # batch[1]: ground truth labels from the dataset
            accuracy += self._calc_accuracy_for_batch(result[0], batch[1])
            inj_accuracy += self._calc_accuracy_for_batch(inj_result[0], batch[1])

        # Divide accumulated accuracy by number of batches.
        accuracy = accuracy / self.mm.get_selected_layer_output_values().shape[0]
        inj_accuracy = (
            inj_accuracy / self.mm.get_selected_layer_output_values().shape[0]
        )

        logging.info("Done.")
        logging.info(
            "Golden run accuracy for original model is: {}".format(self.golden_run[1])
        )
        logging.info(
            "Golden run accuracy for predictions from selected layer is: {}".format(
                accuracy
            )
        )
        logging.info("Resulting accuracy after injection is: {}".format(inj_accuracy))

        return accuracy
