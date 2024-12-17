import os
import cv2

# Set an environment variable to suppress macOS AVFoundation-related warnings.
# This prevents unnecessary warning messages in the terminal.
os.environ["QT_MAC_WANTS_LAYER"] = "1"

# Load the pre-trained Haar Cascade model for face detection.
# Haar Cascade is a machine learning-based approach where the cascade function 
# is trained using positive (faces) and negative (non-faces) images.
# OpenCV provides various pre-trained Haar Cascade models, including models for 
# frontal face detection, profile face detection, and more.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam for capturing video frames.
# On macOS, we explicitly specify the AVFoundation backend (cv2.CAP_AVFOUNDATION)
# for better compatibility with modern camera APIs.
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Verify if the webcam is successfully opened.
# If the webcam fails to open (e.g., due to permission issues), exit the program.
if not cap.isOpened():
    print("Error: Unable to access the camera. Ensure permissions are granted.")
    exit()

# Start a continuous loop to process video frames in real time.
while True:
    # Capture the current frame from the webcam.
    ret, frame = cap.read()
    if not ret:
        # If the frame couldn't be read, log an error and exit the loop.
        print("Error: Couldn't read frame from camera.")
        break

    # Convert the captured frame to grayscale.
    # Face detection works faster and more efficiently on grayscale images 
    # compared to colored (RGB) images.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the Haar Cascade model.
    # The detectMultiScale function scans the image at multiple scales and 
    # detects objects that match the Haar Cascade features for faces.
    # - scaleFactor: Specifies how much the image size is reduced at each scale.
    # - minNeighbors: Specifies the number of neighbors each rectangle should 
    #   have to be considered a face (higher values reduce false positives).
    # - minSize: Specifies the minimum size of the detected face.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each detected face, draw a rectangle around it.
    # The rectangle is drawn using the (x, y) coordinates of the top-left corner 
    # and the width (w) and height (h) of the bounding box.
    # The color of the rectangle is blue (BGR format: 255, 0, 0), and its thickness is 2 pixels.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the processed frame with the drawn rectangles in a window.
    # The window will show the real-time video feed with detected faces highlighted.
    cv2.imshow('Real-Time Face Detection', frame)

    # Wait for a key press for 1 millisecond. If the 'q' key is pressed, exit the loop.
    # The bitwise AND operation ensures compatibility with different platforms.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop ends, release the webcam resources.
# This step ensures that the camera is properly closed and can be used by other programs.
cap.release()

# Close all OpenCV windows that were created during the program's execution.
cv2.destroyAllWindows()

