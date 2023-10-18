import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Create a VideoCapture object to access the webcam (usually, 0 represents the default camera)
cap = cv2.VideoCapture(0)

# Create a FaceMesh instance
face_mesh = mp.solutions.face_mesh

# Initialize the FaceMesh model
with face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh_model:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame with FaceMesh
        results = face_mesh_model.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Access landmarks for the lips (points 12 and 16)
                lip_top = face_landmarks.landmark[12]
                lip_bottom = face_landmarks.landmark[16]

                # Convert landmarks to pixel coordinates
                height, width, _ = frame.shape
                lip_top_x = int(lip_top.x * width)
                lip_top_y = int(lip_top.y * height)
                lip_bottom_x = int(lip_bottom.x * width)
                lip_bottom_y = int(lip_bottom.y * height)

                # Calculate the Euclidean distance between points 12 and 16
                distance = np.sqrt((lip_bottom_x - lip_top_x) ** 2 + (lip_bottom_y - lip_top_y) ** 2)

                # Display the distance on the frame
                cv2.putText(frame, f"Distance: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks on the frame
                cv2.circle(frame, (lip_top_x, lip_top_y), 2, (0, 255, 0), -1)
                cv2.circle(frame, (lip_bottom_x, lip_bottom_y), 2, (0, 255, 0), -1)

                # Check if the mouth is open (distance > 30)
                if distance > 20:
                    print('Mouth open')
                    # Simulate pressing the space key when mouth is open
                    pyautogui.press('space')

        # Display the frame with added landmarks and distance
        cv2.imshow('Video with Mouth Landmarks and Distance', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()




    
                              





















