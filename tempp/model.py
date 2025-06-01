import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define paths
base_dir = 'C:/Users/mahes/OneDrive/Desktop/FinalYearProjects/Projects/ASD-Detection-App/app/'
FEATURES_FOLDER = os.path.join(base_dir, 'tempp', 'features')
MODEL_PATH = os.path.join(base_dir, "models", "lstm_asd_model.h5")

# Load dataset
X, y = [], []
for file in os.listdir(FEATURES_FOLDER):
    if file.endswith(".npy"):
        feature_path = os.path.join(FEATURES_FOLDER, file)
        features = np.load(feature_path)
        if features.shape == (100, 40):  # Expect (timesteps, features)
            X.append(features)
            label = 1 if file.startswith("aut_") else 0  # ASD = 1, Non-ASD = 0
            y.append(label)
        else:
            print(f"Skipping {file}, incorrect shape: {features.shape}")

# Convert to numpy arrays
X = np.array(X).astype(np.float32)  # Shape: (num_samples, 100, 40)
y = np.array(y).astype(int)
y = to_categorical(y, num_classes=2)  # Shape: (num_samples, 2)

if len(X) == 0:
    raise ValueError("No valid features found in FEATURES_FOLDER. Please run mfcc_extract.py to generate features.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(100, 40)),  # (timesteps, features)
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(2, activation="softmax")  # 2 classes: ASD vs. Non-ASD
])

# Compile and Train Model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print("Training LSTM model...")
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Save Model
model.save(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Validation Accuracy: {accuracy:.4f}")