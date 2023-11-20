import cv2 as cv
import mediapipe as mp
import pyautogui
import threading

def gesture_detection():
    global y9, y12
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx == 9:
                        x9, y9 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    elif idx == 12:
                        x12, y12 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

# condition to go here!!!

def display_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

     #      cv.imshow('Webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap = cv.VideoCapture(0)
hands = mp.solutions.hands.Hands()

y9, y12 = 0, 0

gesture_thread = threading.Thread(target=gesture_detection)
display_thread = threading.Thread(target=display_frames)

gesture_thread.start()
display_thread.start()

gesture_thread.join()
display_thread.join()

cap.release()
cv.destroyAllWindows()

