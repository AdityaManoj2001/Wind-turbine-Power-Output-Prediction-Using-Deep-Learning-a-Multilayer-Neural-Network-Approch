#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

# Load SCADA dataset
df = pd.read_csv("C:\\Users\\adity\\OneDrive\\Desktop\\Data Driven Manufacturing 11\\Google Professional Certificate\\Wind Turbine Power Output Prediction Using Deep Learning A Multilayer Neural Network Approach\\archive (1)\\T1.csv")

# Convert Date/Time to datetime format
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format="%d %m %Y %H:%M")

# Drop rows with missing values
df = df.dropna()

# Select relevant features and target
features = ['Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
target = 'LV ActivePower (kW)'

X = df[features]
y = df[target]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network model using TensorFlow
model_tf = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input layer with 3 features
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer 1
    tf.keras.layers.Dense(64, activation='relu'),   # Hidden layer 2
    tf.keras.layers.Dense(32, activation='relu'),   # Hidden layer 3
    tf.keras.layers.Dense(16, activation='relu'),   # Hidden layer 4
    tf.keras.layers.Dense(1, activation='linear')   # Output layer for regression
])

# Compile the model
model_tf.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Custom callback to print predictions after each epoch
class PredictionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Predict on test data after each epoch
        predictions = self.model.predict(X_test)
        print(f"\nEpoch {epoch+1} Predictions:")
        print(predictions[:5])  # Display first 5 predictions for brevity
        print(f"MAE: {logs['mae']:.4f}, Validation MAE: {logs['val_mae']:.4f}\n")

# Train the TensorFlow model and print continuous predictions
history_tf = model_tf.fit(
    X_train, 
    y_train, 
    validation_data=(X_test, y_test), 
    epochs=100, 
    batch_size=32, 
    callbacks=[PredictionCallback()]  # Add the callback for continuous predictions
)

# Evaluate the TensorFlow model
test_loss_tf, test_mae_tf = model_tf.evaluate(X_test, y_test)
print(f'\nFinal TensorFlow Test MAE: {test_mae_tf}')

# Make final predictions after training
final_predictions = model_tf.predict(X_test)

# Show final continuous predictions
print("\nFinal Predictions:")
print(final_predictions[:10])  # Display first 10 predictions


# In[ ]:




