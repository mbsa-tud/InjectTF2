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
from tensorflow.python.keras import backend as K

from inject_tf2.injection_manager import InjectionManager
from inject_tf2.model_manager import ModelManager
from inject_tf2.config_manager import ConfigurationManager


class InjectTF2:
    def __init__(
        self,
        model,
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

        # WIP: Not using Model Manger anymore, since itermediate values of the
        # selected layer are no longer saved due to memory issues.
        # self.mm = ModelManager(
        #     model, self.cm.get_selected_layer(), self.batched_ds, batch_size
        # )
        self._model = model

    def evaluate_golden_run_model(self):
        """Evaluates the model using the `.evaluate()` function with the provided dataset."""
        return self.mm.get_org_model().evaluate(self.batched_ds)

    def evaluate_golden_run_layer(self):
        """Returns the reference accuracy for the predictions obtained when executing from the selected layer onwards."""
        return self._get_predictions_from_layer()

    def evaluate_layer_with_injections(self):
        """Returns the accuracy for the predictions obtained when when executing from the selected layer onwards with fault injections."""
        return self._get_predictions_from_layer(inject=True)

    def _calc_accuracy_for_batch(self, pred_batch, truth_batch):

        pred_batch = np.argmax(pred_batch, axis=1)

        # Check for one-hot encoded ground truth labels.
        if len(truth_batch.shape) > 1:
            truth_batch = np.argmax(truth_batch, axis=1)

        ret = np.zeros(pred_batch.shape, np.float)
        correct_pred = np.equal(pred_batch, truth_batch, out=ret)
        accuracy = np.mean(correct_pred, dtype=np.float64)
        return accuracy

    def _get_predictions_from_layer(self, inject=False):

        logging.info("Starting prediction from selected layer...")

        # Compute `accuracy` for the predictions obtained when executing
        # the model from the selected layer onwards.
        accuracy = 0

        for input_values, batch in zip(
            self.mm.get_selected_layer_output_values(), self.batched_ds
        ):

            # Get the prediction function for the selected layer.
            predict = self.mm.predict_func_from_layer()

            result = None

            # `predict()` returns the predictions for the current
            # `input_values` batch wrapped in a list.
            if not inject:
                result = predict(input_values)

            else:
                result = predict(self.im.inject_batch(input_values, self.cm.get_data()))

            # Caluculate accuracy for current batch.
            # result[0]: array containing the predictions
            # batch[1]: ground truth labels from the dataset
            accuracy += self._calc_accuracy_for_batch(result[0], batch[1])

        # Divide accumulated accuracy by number of batches.
        accuracy = accuracy / self.mm.get_selected_layer_output_values().shape[0]

        logging.info("Done.")
        logging.info("Resulting accuracy is: {}".format(accuracy))

        return accuracy

    def get_top_k_from_layer(self, inject=False, k=1):
        logging.info("Starting top k prediction from selected layer...")

        res_top_k_batched = []

        for input_values, image_name in self.batched_ds:

            if not inject:

                # Predict on current batch.
                result = self._model.predict(input_values)

            else:

                # Run classificaion on current batch until selected layer.
                selected_layer = self._model.get_layer(self.cm.get_selected_layer())

                # Create prediction function until selected layer to catch the values.
                pred_till_layer = K.function(self._model.inputs, selected_layer.output)

                # Catch values.
                values_of_layer_for_batch = pred_till_layer(input_values)


                # Create prediction function from selected layer
                # to the output of the network.
                pred_from_layer = K.function(selected_layer.output, self._model.outputs)

                # Inject and predict on injected values from the selected layer onward.
                result = pred_from_layer(self.im.inject_batch(
                    values_of_layer_for_batch,
                    self.cm.get_data()
                    ))

            # Collect top k elements.
            res_top_k = tf.math.top_k(result, k=k)

            res_current_batch = (res_top_k, image_name)

            res_top_k_batched.append(res_current_batch)

        logging.info("Done.")

        return res_top_k_batched
