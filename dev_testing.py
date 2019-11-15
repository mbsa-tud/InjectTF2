# dev testing setup
import tensorflow as tf
import numpy as np

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
    "./model/simple_mnist_model.h5", "./config/dev_config.yml", x_test, "DEBUG"
)

exp_res = itf2.run_experiments(x_test)
print(len(exp_res))
print(len(exp_res[0]))


# print("Exp res arg max: {0}".format(np.argmax(exp_res[-1])))
# print(np.argmax(itf2.golden_run_layers))
print(len(itf2.golden_run_layers))
print(len(itf2.golden_run_layers[0]))

diff_expt_gold_layers = exp_res[-1] - itf2.golden_run_layers[-1]
print("\n\nDifference is: \n {0}".format(exp_res[-1] - itf2.golden_run_layers[-1]))
print("Golden run minus exp: {0}".format(exp_res[-1] - itf2.golden_run))
print("Difference golden run: {0}".format(itf2.golden_run - itf2.golden_run_layers[-1]))

print(exp_res[-1].shape)
diff = np.argmax(diff_expt_gold_layers)
print(diff)
ind = np.unravel_index(np.argmax(diff_expt_gold_layers, axis=None), diff_expt_gold_layers.shape)
print(diff_expt_gold_layers[ind])
