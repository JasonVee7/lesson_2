import cv2
import mediapipe as mp
import math
import pyautogui

# Initialize MediaPipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Open default camera (webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmark coordinates for points 12 and 16 (top and bottom lips)
            top_lip = face_landmarks.landmark[12]
            bottom_lip = face_landmarks.landmark[16]

            # Get landmark coordinates
            ih, iw, _ = frame.shape
            top_lip_x, top_lip_y = int(top_lip.x * iw), int(top_lip.y * ih)
            bottom_lip_x, bottom_lip_y = int(bottom_lip.x * iw), int(bottom_lip.y * ih)

            # Draw circles around the points (landmark 12 and 16)
            cv2.circle(frame, (top_lip_x, top_lip_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (bottom_lip_x, bottom_lip_y), 5, (0, 255, 0), -1)

            # Draw a line connecting the points
            cv2.line(frame, (top_lip_x, top_lip_y), (bottom_lip_x, bottom_lip_y), (0, 255, 0), 2)

            # Calculate Euclidean distance between points 12 and 16
            distance = math.sqrt((bottom_lip_x - top_lip_x) ** 2 + (bottom_lip_y - top_lip_y) ** 2)
            distance = round(distance, 2)  # Round the distance to 2 decimal places

             # set threshold
            if distance > 23:
                print('mouth open')
                pyautogui.press('space')

            # Display the distance on the webcam window
            cv2.putText(frame, f"Distance: {distance}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

           


    # Display the webcam window with landmarks and distance
    cv2.imshow('Lips Landmark Detection', frame)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()


