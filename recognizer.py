import cv2
import numpy as numpy
import os

detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("trainer/trainer.yml")
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
name = ['none', 'Godswill', 'Ebere', 'Godswill', 'handle']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.1, 2)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+y])

        if (confidence <= 100):
            id = name[id]
            confidence = "{0}%".format(round(100-confidence))

        else: 
            id = "Unknown"
            confidence = "{}%".format(round(100-confidence))

        cv2.putText(frame, str(id), (x+5, y-5), font, 1, (255, 0, 0), 2)
        cv2.putText(frame, str(confidence), (x+5, y+h-5), font, 1, (255, 0, 0), 2)
        
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(30) & 0xff
    if k == "q":
        break

cap.release()
cv2.destroyAllWindows()



