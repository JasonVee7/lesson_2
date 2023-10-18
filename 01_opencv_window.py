import cv2

# Get user input for customisation
width = int(input("Enter window width: "))
height = int(input("Enter window height: "))
x_position = int(input("Enter starting X position: "))
y_position = int(input("Enter starting Y position: "))


# Create a frame window with custom parameters
cv2.namedWindow('OpenCV Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('OpenCV Window', width, height)
cv2.moveWindow('OpenCV Window', x_position, y_position)
print("Your customised window is ready!")

# Set window properties to bring it to the front
cv2.setWindowProperty('OpenCV Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty('OpenCV Window', cv2.WND_PROP_TOPMOST, 1)

# Wait for the window to be closed
while True:
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Close the window
cv2.destroyAllWindows()
