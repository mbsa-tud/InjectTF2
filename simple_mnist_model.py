# Build a simple model to classify the MNIST dataset
# The model is a combination of:
# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.tensorflow.org/tutorials/quickstart/advanced

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Print the TensorFlow version
print(f"TensorFlow version: {tf.version.VERSION}")

# - - - - - Variables - - - - -
save_path = "./model"
model_name = "simple_mnist_model.h5"
# - - - - - - - - - - - - - - -

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

model.save(save_path + "/" + model_name)

model.summary()
