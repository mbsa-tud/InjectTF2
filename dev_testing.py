# dev testing setup
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


from inject_tf2.inject_tf2 import InjectTF2

# Print the TensorFlow version
print(f"TensorFlow version: {tf.version.VERSION}")

# - - - - - Variables - - - - -
save_path = "./model"
model_name = "simple_mnist_model.h5"
# - - - - - - - - - - - - - - -

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


itf2 = InjectTF2(
    "./model/simple_mnist_model.h5", "./config/dev_config.yml", "", "DEBUG"
)

itf2.run_experiments()
