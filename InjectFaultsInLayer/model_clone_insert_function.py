# Build a simple model to classify the MNIST dataset
# The model is a combination of:
# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.tensorflow.org/tutorials/quickstart/advanced


# region import block
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer, Dropout
from tensorflow.keras import Model

import InjectFaultsInLayer.LayerWrapper as LayerWrapper

# endregion

# Print the TensorFlow version
print(f"TensorFlow version: {tf.version.VERSION}")

# - - - - - Variables - - - - -
save_path = "./InjectFaultsInLayer/model/"
model_name = "testModel.h5"
cloned_model_name = "clonedTestModel.h5"
# - - - - - - - - - - - - - - -


# region setup simple model and save it
mnist = tf.keras.datasets.mnist
#
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
#
#
# model = tf.keras.models.Sequential(
#     [
#         tf.keras.layers.Flatten(input_shape=(28, 28)),
#         tf.keras.layers.Dense(128, activation="relu"),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(10, activation="softmax"),
#     ]
# )
#
# model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
#
# model.fit(x_train, y_train, epochs=5)
#
# model.evaluate(x_test, y_test, verbose=2)
#
# model.save(save_path + model_name)
# endregion

model = tf.keras.models.load_model(save_path + model_name)

model.summary()


# basicaly we try to inject another function to run in the layer

def cloneFunction(layer):
    argument_layer = layer.__class__.from_config(layer.get_config())
    # if type(layer) is Flatten:
    #     wrapped_layer = LayerWrapper.LayerWrapperFlatten(layer)
    # if type(layer) is Dense:
    #     wrapped_layer = LayerWrapper.LayerWrapperDense(layer)
    # if type(layer) is Dropout:
    #     wrapped_layer = LayerWrapper.LayerWrapperDropout(layer)
    wrapped_layer = LayerWrapper.LayerWrapper(layer)
    return wrapped_layer


clonedModel = tf.keras.models.clone_model(
    model,
    input_tensors=None,
    clone_function=cloneFunction
)



#model.fit(x_train, y_train, epochs=5)

#model.evaluate(x_test, y_test, verbose=2)

clonedModel.compile(
     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

clonedModel.summary()

#clonedModel.fit(x_train, y_train, epochs=5)

#clonedModel.evaluate(x_test, y_test, verbose=2)

# new_model = tf.keras.models.load_model(model_name)
