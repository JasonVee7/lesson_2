import cv2
import mediapipe as mp

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)  # 0 represents the default webcam, change it if you have multiple cameras

# Create a FaceMesh instance
face_mesh = mp.solutions.face_mesh

# Initialize the FaceMesh model
with face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
) as face_mesh_model:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Process the frame using FaceMesh
        results = face_mesh_model.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Access iris landmarks for the left eye (landmarks 468-477)
                left_eye_iris_landmarks = face_landmarks.landmark[468:478]

                # Access iris landmarks for the right eye (landmarks 478-487)
                right_eye_iris_landmarks = face_landmarks.landmark[478:488]

                # Convert landmarks to pixel coordinates
                height, width, _ = frame.shape
                for landmark in left_eye_iris_landmarks + right_eye_iris_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    # Draw landmarks on the frame
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Display the frame with iris landmarks
            cv2.imshow('Webcam with Iris Landmarks', frame)

        if cv2.waitKey(1) == ord('q'):
            break

# Release the VideoCapture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
