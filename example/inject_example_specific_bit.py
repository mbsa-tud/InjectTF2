import sys

import tensorflow as tf

Dataset = tf.data.Dataset
mnist = tf.keras.datasets.mnist

# Add InjectTF2 to python path.
inject_tf2_path = "../"

if inject_tf2_path not in sys.path:
    sys.path.append(inject_tf2_path)

from inject_tf2 import InjectTF2


# Prepare data set
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

# Add a channels dimension
x_test = x_test[..., tf.newaxis].astype("float32")
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))


model = tf.keras.models.load_model("mnist_model.h5")

model.summary()

itf2 = InjectTF2(model, "./example_config_2.yml", test_ds, 32, "INFO")

itf2.evaluate_layer_with_injections()
