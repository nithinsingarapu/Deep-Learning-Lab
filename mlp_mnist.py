# Import libraries
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))   
model.add(Dropout(0.2))                        
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',            
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train,
          epochs=20,
          batch_size=64,
          validation_split=0.1,
          verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")

#predictions
sample = X_test[:5]
sample_labels = y_test[:5]

predictions = model.predict(sample, sample_labels)
print(predictions)
result = np.argmax(predictions, axis = 1)
print(result)

for i in range(sample):
  plt.subplot(1,5,i+1)
  plt.title(f"Actual label : {sample_labels[i]}\ predicted labels : {result[i]}")
  plt.imshow(sample[i])
