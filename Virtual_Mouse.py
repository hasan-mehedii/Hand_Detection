import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import logging

# Suppress Mediapipe/TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get the screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

# Variables for smoothing
prev_x, prev_y = 0, 0
smoothening = 7

# Click detection
click_distance_threshold = 40
click_cooldown = 1.0
last_click_time = 0

# Main loop
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark
        index_finger = landmarks[8]
        thumb_finger = landmarks[4]

        index_x = int(index_finger.x * screen_width)
        index_y = int(index_finger.y * screen_height)
        thumb_x = int(thumb_finger.x * screen_width)
        thumb_y = int(thumb_finger.y * screen_height)

        cv2.circle(frame, (int(index_finger.x * frame_width), int(index_finger.y * frame_height)), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (int(thumb_finger.x * frame_width), int(thumb_finger.y * frame_height)), 10, (0, 255, 255), cv2.FILLED)

        curr_x = prev_x + (index_x - prev_x) / smoothening
        curr_y = prev_y + (index_y - prev_y) / smoothening
        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
        current_time = time.time()

        if distance < click_distance_threshold and (current_time - last_click_time) > click_cooldown:
            pyautogui.click()
            last_click_time = current_time

    cv2.imshow("Virtual Mouse", frame)

    # Handle window close events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # If user clicks the X button
    if cv2.getWindowProperty("Virtual Mouse", cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
