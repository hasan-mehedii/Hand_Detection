import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

# For smoothing cursor movement
prev_x, prev_y = 0, 0
smoothening = 7  # Higher value means more smoothing

click_cooldown = 1.0  # seconds
last_click_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)

    hands = output.multi_hand_landmarks

    if hands:
        hand = hands[0]  # Single hand for simplicity
        drawing_utils.draw_landmarks(frame, hand)

        landmarks = hand.landmark

        # Get coordinates of index finger tip (id=8) and thumb tip (id=4)
        index_finger = landmarks[8]
        thumb_finger = landmarks[4]

        # Convert to screen coordinates
        index_x = int(index_finger.x * screen_width)
        index_y = int(index_finger.y * screen_height)
        thumb_x = int(thumb_finger.x * screen_width)
        thumb_y = int(thumb_finger.y * screen_height)

        # Draw circles on fingertips
        cv2.circle(frame, (int(index_finger.x * frame_width), int(index_finger.y * frame_height)), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (int(thumb_finger.x * frame_width), int(thumb_finger.y * frame_height)), 10, (0, 255, 255), cv2.FILLED)

        # Calculate distance between index finger and thumb
        distance = np.hypot(index_x - thumb_x, index_y - thumb_y)

        # Move mouse smoothly
        curr_x = prev_x + (index_x - prev_x) / smoothening
        curr_y = prev_y + (index_y - prev_y) / smoothening
        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Click if fingertips are close enough and cooldown passed
        if distance < 40:
            current_time = time.time()
            if (current_time - last_click_time) > click_cooldown:
                pyautogui.click()
                last_click_time = current_time

    cv2.imshow('Virtual Mouse', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
