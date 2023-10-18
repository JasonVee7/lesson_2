import cv2
import mediapipe as mp
import numpy as np
import requests

# URL of the image
image_url = "https://github.com/JasonVee7/lesson_2/raw/main/640x960.jpg"

# Load the image from the URL
response = requests.get(image_url)
image_array = np.frombuffer(response.content, dtype=np.uint8)
image = cv2.imdecode(image_array, -1)

# Create a FaceMesh instance
face_mesh = mp.solutions.face_mesh

# Initialize the FaceMesh model
with face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,    #required paramater to access iris points 469 - 473 and 474 - 478
) as face_mesh_model:
    results = face_mesh_model.process(image)

    if results.multi_face_landmarks:
        # Access iris landmarks for the left eye (landmarks 468-477)
        left_eye_iris_landmarks = results.multi_face_landmarks[0].landmark[468:478]

        # Access iris landmarks for the right eye (landmarks 478-487)
        right_eye_iris_landmarks = results.multi_face_landmarks[0].landmark[478:488]

        # Convert landmarks to pixel coordinates
        height, width, _ = image.shape
        for landmark in left_eye_iris_landmarks + right_eye_iris_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # Draw landmarks on the image
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Display the image with iris landmarks
        cv2.imshow('Image with Iris Landmarks', image)

        # Wait for the window to be closed
        while True:
            key = cv2.waitKey(1)
            if key == 27:  # 27 is the ASCII code for the 'Esc' key
                break

# Close the windows
cv2.destroyAllWindows()
