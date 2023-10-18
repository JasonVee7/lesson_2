import cv2
import mediapipe as mp

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

        # Convert the frame to grayscale (optional, if not already in grayscale)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY

        # Process the frame with FaceMesh
        results = face_mesh_model.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Access landmarks for the lips (points 12 and 16)
                lip_top = face_landmarks.landmark[12]
                lip_bottom = face_landmarks.landmark[16]

                # Access landmarks for additional points (308 and 78)
                additional_point_1 = face_landmarks.landmark[308]
                additional_point_2 = face_landmarks.landmark[78]

                # Convert landmarks to pixel coordinates
                height, width, _ = frame.shape
                lip_top_x = int(lip_top.x * width)
                lip_top_y = int(lip_top.y * height)
                lip_bottom_x = int(lip_bottom.x * width)
                lip_bottom_y = int(lip_bottom.y * height)
                additional_point_1_x = int(additional_point_1.x * width)
                additional_point_1_y = int(additional_point_1.y * height)
                additional_point_2_x = int(additional_point_2.x * width)
                additional_point_2_y = int(additional_point_2.y * height)

                # Draw landmarks on the frame
                cv2.circle(frame, (lip_top_x, lip_top_y), 2, (0, 255, 0), -1)
                cv2.circle(frame, (lip_bottom_x, lip_bottom_y), 2, (0, 255, 0), -1)
                cv2.circle(frame, (additional_point_1_x, additional_point_1_y), 2, (0, 255, 0), -1)
                cv2.circle(frame, (additional_point_2_x, additional_point_2_y), 2, (0, 255, 0), -1)

        # Display the frame with added landmarks
        cv2.imshow('Video with Mouth Landmarks', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
