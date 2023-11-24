import cv2
import mediapipe as mp

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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                if idx == 9:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at landmark 9
                    print('9: ', y)
                    # Display the y coordinate of landmark 9 on the frame window
                    cv2.putText(frame, f"y: {y}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                elif idx == 12:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Draw a blue circle at landmark 12
                   # print('12: ', y)

                    # Display the y coordinate of landmark 9 on the frame window
                    cv2.putText(frame, f"y: {y}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


    # Show the frame with detections
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()



