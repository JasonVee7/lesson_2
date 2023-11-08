import cv2
import mediapipe as mp

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Initialize the face detection model
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5
) as face_detection:

    while cap.isOpened():
        # Read frame
        success, image = cap.read()
        if not success:
            break

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        results = face_detection.process(image_rgb)

        # If a face is detected, highlight it with bounding box
        if results.detections:
            for detection in results.detections:
                # Get the bounding box data
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                
                # Draw the bounding box
                mp_drawing.draw_detection(image, detection)
                
                # Print the bounding box coordinates
                print("Bounding Box (x, y, width, height):", bbox)

        # Show the output
        cv2.imshow('Face Detection', image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture and close the window
cap.release()
cv2.destroyAllWindows()
