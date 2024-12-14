import os
import cv2

# Set environment variables to avoid macOS AVFoundation warnings
os.environ["QT_MAC_WANTS_LAYER"] = "1"

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam using VideoCapture with a fallback to ensure macOS compatibility
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Use AVFoundation backend explicitly for macOS

if not cap.isOpened():
    print("Error: Unable to access the camera. Ensure permissions are granted.")
    exit()

while True:
    # Read frames from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame from camera.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Real-Time Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
