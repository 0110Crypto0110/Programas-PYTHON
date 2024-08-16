import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
import matplotlib.pyplot as plt

SEED_VALUE = 42

# Fix seed to make training deterministic.
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Load the Boston housing dataset.
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print(X_train.shape)
print("\nInput features: ", X_train[0])
print("\nOutput target: ", y_train[0])

# Feature of interest: Average Number of Rooms (column index 5)
boston_features = {
    "Average Number of Rooms": 5,
}

X_train_1d = X_train[:, boston_features["Average Number of Rooms"]]
X_test_1d = X_test[:, boston_features["Average Number of Rooms"]]

print(X_train_1d.shape)

plt.figure(figsize=(15, 5))
plt.xlabel("Average Number of Rooms")
plt.ylabel("Median Price [$K]")
plt.grid(True)
plt.scatter(X_train_1d[:], y_train, color="green", alpha=0.5)
plt.show()

# Define the model consisting of a single neuron.
model = Sequential()
model.add(Dense(units=1, input_shape=(1,)))

# Display a summary of the model architecture.
model.summary()

# Compile the model with RMSprop optimizer and mean squared error loss function.
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005), loss="mse")

# Train the model
history = model.fit(
    X_train_1d, 
    y_train, 
    batch_size=16, 
    epochs=101, 
    validation_split=0.3,
)

def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlim([0, 100])
    plt.ylim([0, 300])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)

# Predict the median price of a home with [3, 4, 5, 6, 7] rooms.
x = [3, 4, 5, 6, 7]
y_pred = model.predict(x)
for idx in range(len(x)):
    print(f"Predicted price of a home with {x[idx]} rooms: ${int(y_pred[idx] * 10) / 10}K")

# Generate feature data that spans the range of interest for the independent variable.
x_range = np.linspace(3, 9, 10)

# Use the model to predict the dependent variable.
y_range_pred = model.predict(x_range)

# Function to plot data
def plot_data(x_data, y_data, x, y, title=None):
    plt.figure(figsize=(15,5))
    plt.scatter(x_data, y_data, label='Ground Truth', color='green', alpha=0.5)
    plt.plot(x, y, color='k', label='Model Predictions')
    plt.xlim([3,9])
    plt.ylim([0,60])
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Price [$K]')
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot training data predictions
plot_data(X_train_1d, y_train, x_range, y_range_pred, title='Training Dataset')

# Plot test data predictions
plot_data(X_test_1d, y_test, x_range, y_range_pred, title='Test Dataset')
