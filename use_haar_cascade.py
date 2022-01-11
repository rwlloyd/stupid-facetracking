import cv2
import numpy as np
from math import floor

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html
b_cascade = cv2.CascadeClassifier("./haarcascade_fullbody.xml")
f_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Function to get the centre of the detected object
def getCentre(x, y, w, h):
    return floor(x+(w/2)), floor(y+(h/2))


while(True):
    ret, frame = cap.read()

    if ret == True:
        # Convert frame to Gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # find the bodies
        bodies = b_cascade.detectMultiScale(gray)
        # find the faces
        faces = f_cascade.detectMultiScale(gray, 1.3, 5)
        print(bodies, faces)

        # Draw the positions of the faces on the frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawMarker(frame, getCentre(x, y, w, h), (0, 0, 255), 0, 20, 1)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
