
pip install pandas numpy scikit-learn tensorflow flask

"""Load the Dataset"""

import pandas as pd

# Load the dataset
df = pd.read_csv('/content/all-sensors-16.csv')

# Display the first few rows of the data
print(df.head())

""" Preprocess the Data"""

# Drop unnecessary columns
df = df.drop(['Local Time (Sensor)', 'Date', 'countlineName', 'direction', 'Time'], axis=1)

# Define features (X) and target (y)
X = df.drop(columns=['Car'])  # Predicting car volume (Car)
y = df['Car']

"""Train-Test Split"""

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Data Scaling"""

from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""Build the Baseline Neural Network Model"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Single output for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32)

""" Evaluate the Model"""

# Evaluate on test data
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"Test MAE: {test_mae}")

"""Save the Trained Model"""

# Save the model
model.save('traffic_volume_model.h5')

"""Track Training and Validation Metrics"""

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

# Plot training & validation MAE values
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend(['Train', 'Validation'])

plt.show()

# Evaluate on training data
train_loss, train_mae = model.evaluate(X_train_scaled, y_train)
print(f"Training Loss: {train_loss}")
print(f"Training MAE: {train_mae}")

# Evaluate on test data (validation set)
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

from tensorflow.keras.regularizers import l2
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])

from tensorflow.keras.layers import Dropout
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

# Save the model
model.save('traffic_volume_model.h5')

