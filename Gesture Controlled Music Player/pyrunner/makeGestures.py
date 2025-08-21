# collect_gesture_data.py
import cv2
import numpy as np
import mediapipe as mp
import os

GESTURES = ["play", "pause", "next", "previous", "volume_up", "volume_down"]
SAVE_DIR = "gesture_data"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
current_label = "play"
samples = []

print("Press SPACE to capture a frame for gesture:", current_label)
print("Press keys 1-6 to change label")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            current_landmarks = np.array(landmarks)

            key = cv2.waitKey(1)
            if key == ord(' '):  # Space to save
                print(f"Saved frame for label: {current_label}")
                samples.append(current_landmarks)
            elif key in [ord(str(i)) for i in range(1, 7)]:
                current_label = GESTURES[key - ord('1')]
                print("Switched to gesture:", current_label)

    cv2.putText(image, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collecting Gestures", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save collected data
if samples:
    samples = np.array(samples)
    np.save(os.path.join(SAVE_DIR, f"{current_label}.npy"), samples)

cap.release()
cv2.destroyAllWindows()
