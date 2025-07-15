import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

X = np.linspace(1, 10, 100)
Y = 2 * X + 10 + np.random.randn(100)

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

model.compile(optimizer='sgd', loss='mse')

model.fit(X, Y, epochs=50, verbose=1)

pred = model.predict(X)

plt.scatter(X, Y, label='Original Data', color='blue', alpha=0.6)
plt.plot(X, pred, color='red', label='Model Prediction')
plt.title("Linear Regression with Keras")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
