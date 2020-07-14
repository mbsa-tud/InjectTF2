import tensorflow as tf

Dataset = tf.data.Dataset
mnist = tf.keras.datasets.mnist

# Prepare data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

test_ds = Dataset.from_tensor_slices((x_test, y_test)).batch(32)


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPool2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Softmax()
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

print("Model architecture:")

model.summary()

model.fit(train_ds, epochs=4)

test_loss, test_acc = model.evaluate(test_ds, verbose=2)

print("\nTest accuracy is:", test_acc)

save_path = "./mnist_model.h5"
model.save(save_path)

print(f"Saved model: {save_path}.")
