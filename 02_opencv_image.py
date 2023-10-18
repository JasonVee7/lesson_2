import cv2

# Check OpenCV version
print("OpenCV version:", cv2.__version__)

try:
    import numpy as np
    print("NumPy version:", np.__version__)
except ImportError:
    install_numpy = input("NumPy is not installed. Install it? (y/n): ")
    if install_numpy.lower() == 'y':
        import subprocess
        subprocess.call(['pip', 'install', 'numpy'])
        import numpy as np
    else:
        print("NumPy is required to run this script.")

try:
    import requests
    print("Requests version:", requests.__version__)
except ImportError:
    install_requests = input("Requests library is not installed. Install it? (y/n): ")
    if install_requests.lower() == 'y':
        import subprocess
        subprocess.call(['pip', 'install', 'requests'])
        import requests
    else:
        print("Requests library is required to run this script.")

# URL of the image
image_url = "https://github.com/JasonVee7/lesson_2/raw/main/640x960.jpg"


# Load the image from the URL
response = requests.get(image_url)
image_array = np.frombuffer(response.content, dtype=np.uint8)

# Decode the image
image = cv2.imdecode(image_array, -1)

# Create a window and display the image
cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image Window', 640, 960)


cv2.imshow('Image Window', image)

# Wait for the window to be closed
while True:
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Close the window
cv2.destroyAllWindows()
