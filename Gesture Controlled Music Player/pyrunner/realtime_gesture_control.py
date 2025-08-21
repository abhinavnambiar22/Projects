# # realtime_gesture_control.py

# import cv2
# import numpy as np
# import mediapipe as mp
# import joblib
# import time
# import asyncio
# import django
# import os
# import sys

# # Setup Django to access channel layer
# sys.path.append("/home/aadi/Documents/sixth_sem/projects/web_tech_project/web2/music_project")
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "music_project.settings")
# django.setup()

# from channels.layers import get_channel_layer

# # Load the gesture recognition model
# model = joblib.load("gesture_knn_model.pkl")
# GESTURES = ["play", "pause", "next", "previous", "volume_up", "volume_down"]

# # Set up MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.7
# )
# mp_draw = mp.solutions.drawing_utils

# # Start the webcam
# cap = cv2.VideoCapture(0)
# last_gesture = None
# last_sent_time = time.time()

# channel_layer = get_channel_layer()

# if channel_layer is None:
#     print("❌ Channel layer not initialized properly!")
#     sys.exit(1)
# else:
#     print("✅ Channel layer is ready.")

# print("🎮 Real-time Gesture Detection Started (press 'q' to quit)")

# # Async sender for channel layer
# async def send_gesture(gesture):
#     await channel_layer.group_send(
#         "gesture_group",
#         {
#             "type": "gesture_message",
#             "gesture": gesture
#         }
#     )
#     print(f"🖐️ Gesture sent to group: {gesture}")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("⚠️ Failed to grab frame from webcam")
#         break

#     # Flip and convert to RGB
#     image = cv2.flip(frame, 1)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)
#     prediction = "none"

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
#             landmarks = []
#             for lm in handLms.landmark:
#                 landmarks.extend([lm.x, lm.y, lm.z])
#             landmarks = np.array(landmarks).reshape(1, -1)

#             # Predict gesture
#             prediction = model.predict(landmarks)[0]

#             # Only send if gesture has changed or 1.5 sec passed
#             current_time = time.time()
#             if prediction in GESTURES and (prediction != last_gesture or current_time - last_sent_time > 1.5):
#                 try:
#                     asyncio.run(send_gesture(prediction))
#                     last_gesture = prediction
#                     last_sent_time = current_time
#                 except Exception as e:
#                     print("Channel send error:", e)

#     # Show feedback on video
#     cv2.putText(image, f"Gesture: {prediction}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

#     resized_image = cv2.resize(image, (800, 600))  # Resize window to make it visible
#     cv2.imshow("🎮 Real-time Gesture Control", resized_image)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# print("🛑 Webcam closed.")


import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import asyncio
import django
import os
import sys

# Setup Django to access channel layer
sys.path.append("/home/aadi/Documents/sixth_sem/projects/web_tech_project/web2/music_project")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "music_project.settings")
django.setup()

from channels.layers import get_channel_layer

# Load the gesture recognition model
model = joblib.load("gesture_knn_model.pkl")
GESTURES = ["play", "pause", "next", "previous", "volume_up", "volume_down"]

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Start the webcam
cap = cv2.VideoCapture(0)
last_gesture = None
last_sent_time = time.time()

channel_layer = get_channel_layer()
if channel_layer is None:
    print("❌ Channel layer not initialized properly!")
    sys.exit(1)
else:
    print("✅ Channel layer is ready.")

# Wait a short while to ensure the frontend has connected
print("⏳ Waiting 2 seconds for frontend to connect...")
time.sleep(2)

print("🎮 Real-time Gesture Detection Started (press 'q' to quit)")

# Async sender for channel layer
async def send_gesture(gesture):
    await channel_layer.group_send(
        "gesture_group",
        {
            "type": "gesture_message",  # <- Must match the consumer method name
            "gesture": gesture
        }
    )
    print(f"🖐️ Gesture sent to group: {gesture}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame from webcam")
        break

    # Flip and convert to RGB
    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    prediction = "none"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict gesture
            prediction = model.predict(landmarks)[0]

            # Only send if gesture has changed or 1.5 sec passed
            current_time = time.time()
            if prediction in GESTURES and (prediction != last_gesture or current_time - last_sent_time > 1.5):
                try:
                    asyncio.run(send_gesture(prediction))
                    last_gesture = prediction
                    last_sent_time = current_time
                except Exception as e:
                    print("❗ Channel send error:", e)

    # Show feedback on video
    cv2.putText(image, f"Gesture: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    resized_image = cv2.resize(image, (800, 600))  # Resize window to make it visible
    cv2.imshow("🎮 Real-time Gesture Control", resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam closed.")
