import numpy as np
import cv2
import pandas as pd

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# to save the face pic to path
path = "dataset"

face_id = input("\n Type the face ID number: ")

count = 0

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

        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + "." + str(count) + ".jpg", roi_gray)


    # SHOW IMAGE
    cv2.imshow("image", frame)


    k = cv2.waitKey(30) & 0xff
    if k == 'q':
        break
    elif count == 30:
        break

# RELEASE VIDEO
capture.release()

cv2.destroyAllWindows()

