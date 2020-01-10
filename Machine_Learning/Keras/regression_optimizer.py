import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Tensorflow version = ", tf.__version__)

x_data = list(np.arange(-1, 1, 0.01))

np.random.seed(2020)

mu = 0
sigma = 0.2
n = len(x_data)

noises = np.random.normal(mu, sigma, n)


y_temp = list(np.array(x_data)**3 - np.array(x_data)**2 + 3 * np.array(x_data) + 8)
plt.plot(x_data, y_temp)
y_data = list(np.array(y_temp) + np.array(noises))
plt.scatter(x_data, y_data)
plt.show()

train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 4

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(n, activation='relu'))
model.add(tf.keras.layers.Dense(int(1.5*n), activation='relu'))
model.add(tf.keras.layers.Dense(int(1.5*n), activation='relu'))
model.add(tf.keras.layers.Dense(int(1.5*n), activation='relu'))
model.add(tf.keras.layers.Dense(len(y_data)))

adam = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse', metrics=['mae', 'mse'])

model.fit(train_dataset, epochs=50)

print(model(input))
