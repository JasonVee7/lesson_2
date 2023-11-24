import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
hands = mp.solutions.hands.Hands()

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# OpenCV setup
cap = cv2.VideoCapture(0)  # You can specify a different index if you have multiple cameras
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    point_9_y = None
    point_12_y = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                if idx == 9:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at landmark 9
                    point_9_y = y
                elif idx == 12:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Draw a blue circle at landmark 12
                    point_12_y = y

    # Check if y-coordinate of point 12 is greater than point 9
    if point_9_y is not None and point_12_y is not None:
        if point_12_y < point_9_y:
            print('Open hand!')
            pyautogui.press('space')


    # Show the frame with detections
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()


