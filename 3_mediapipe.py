import cv2
import numpy as np
import requests
import subprocess

try:
    import mediapipe as mp
    print("Mediapipe version:", mp.__version__)
except ImportError:
    install_mediapipe = input("Mediapipe is not installed. Install it? (y/n): ")
    if install_mediapipe.lower() == 'y':
        subprocess.call(['pip', 'install', 'mediapipe'])
        import mediapipe as mp
    else:
        print("Mediapipe is required to run this script.")

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

# Perform face detection (Mediapipe)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

results = face_detection.process(image)
if results.detections:
    for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

# Display the image with face detection
cv2.imshow('Image with Face Detection', image)

# Wait for the window to be closed
while True:
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Close the windows
cv2.destroyAllWindows()
