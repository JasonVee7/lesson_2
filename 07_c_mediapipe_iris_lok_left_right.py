import cv2
import mediapipe as mp
import math

# Create a FaceMesh instance
face_mesh = mp.solutions.face_mesh

# Initialize the FaceMesh model
with face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
) as face_mesh_model:
    # Access the webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for a mirrored view
        image = cv2.flip(image, 1)

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and get the face landmarks
        results = face_mesh_model.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Access the specific landmarks by their indices
                landmark_168 = face_landmarks.landmark[168]  # Point on the center line
                landmark_469 = face_landmarks.landmark[469]  # Iris landmark associated with the eye

                # Calculate the Euclidean distance between points 168 and 469
                distance = math.sqrt(
                    (landmark_168.x - landmark_469.x) ** 2 + (landmark_168.y - landmark_469.y) ** 2
                )

                # Check the distance and take action accordingly
                if distance < 0.03:
                    # Change background color to green
                    image[:] = (0, 255, 0)
                    # Display 'Right!' in red text
                    cv2.putText(image, 'RIGHT!', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif distance > 0.04:
                    # Change background color to purple
                    image[:] = (255, 0, 255)
                    # Display 'Left!' in red text
                    cv2.putText(image, 'LEFT!', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                else:
                    # Do nothing if the distance is exactly 0.04
                    pass

                # Display the distance value on the frame window
             #   cv2.putText(image, f"Distance: {distance:.4f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



        # Display the modified image
        cv2.imshow('Gaze Detection', image)

        # Wait for the window to be closed
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Release the VideoCapture and destroy windows
    cap.release()
    cv2.destroyAllWindows()


