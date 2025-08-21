# train_gesture_classifier.py
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import joblib

DATA_DIR = "gesture_data"
GESTURES = ["play", "pause", "next", "previous", "volume_up", "volume_down"]
X = []
y = []

for idx, gesture in enumerate(GESTURES):
    file_path = os.path.join(DATA_DIR, f"{gesture}.npy")
    if os.path.exists(file_path):
        samples = np.load(file_path)
        X.extend(samples)
        y.extend([gesture] * len(samples))
    else:
        print(f"Warning: {gesture}.npy not found")

X = np.array(X)
y = np.array(y)

print(f"Training on {len(X)} samples...")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

joblib.dump(model, "gesture_knn_model.pkl")
print("✅ Model saved to gesture_knn_model.pkl")
