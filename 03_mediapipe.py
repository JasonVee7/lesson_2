import cv2 as cv

# Open webcam
cap = cv.VideoCapture(0)  # Use 0 for default webcam, or change to another number if you have multiple cameras

while cap.isOpened():
    # Read frames from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display the frame in a window
    cv.imshow('Webcam', frame)

    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv.destroyAllWindows()
