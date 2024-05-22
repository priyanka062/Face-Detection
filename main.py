import argparse
import cv2
import numpy as np

# construct the argument parser and parse the arguments
arg = argparse.ArgumentParser()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
arg.add_argument("-f", "--face", required=True, help="path to where the face cascade resides")
arg.add_argument("-e", "--eye", required=True, help="path to where the eye cascade resides")

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w] # type: ignore
        roi_color = img[y:y+h, x:x+w]
        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #To draw a rectangle in eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
        cv2.putText(img, "lodu Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    # Display an image in a window
    cv2.imshow('img',img)
    
    # Wait for Esc key to stop
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
