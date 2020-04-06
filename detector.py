import numpy as np
import cv2
import pandas as pd

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


# READ VIDEO
capture = cv2.VideoCapture(0)

# READ CAPTURE FROM VIDEO
while True:
    ret, frame = capture.read()

    #CONVERT IMAGE TO GRAYSCALE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #DETECT FACE
    faces = faceCascade.detectMultiScale(gray, 1.3, 2)

    # DRAW RECTANGLES ON EACH FACE DETECTED
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    # SHOW IMAGE
    cv2.imshow("image", frame)


    k = cv2.waitKey(5) & 0xff
    if k == 'q':
        break

# RELEASE VIDEO
capture.release()

cv2.destroyAllWindows()

