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

# Create a window and display the image
cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image Window', 640, 960)
cv2.imshow('Image Window', image)

# Create a FaceMesh instance
face_mesh = mp.solutions.face_mesh

# Initialize the FaceMesh model
with face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh_model:
    results = face_mesh_model.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert landmarks to pixel coordinates
            height, width, _ = image.shape
            landmark_points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmark_points.append((x, y))

            # Draw landmarks on the image
            for x, y in landmark_points:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Display the image with landmarks
cv2.imshow('Image with Face Landmarks', image)

# Wait for the window to be closed
while True:
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Close the windows
cv2.destroyAllWindows()
