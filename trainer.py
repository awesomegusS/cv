import cv2
import numpy as np
import os
from PIL import Image

detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# create a face recognizer object
recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'dataset'

# Function iterates through the pictures in the dataset folder
def ImageAndLabels(path):
    ImagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    for ImagePath in ImagePaths:
        
        # using the pillow module, open an image and convert it to grayscale
        PIL_img = Image.open(ImagePath).convert('L')
        
        # an array of numbers in the unit8 data type representing the image
        img_numpy = np.array(PIL_img, 'uint8') 

        faceSamples = []

        ids = []

        # this would extract the id number from the Image path string
        id = int(os.path.split(ImagePath)[-1].split(".")[1])

        faces = detector.detectMultiScale(img_numpy)
        
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[x:x+w, y:y+h])
            ids.append(id)

    return faceSamples, ids


faces, ids = ImageAndLabels(path)

recognizer.train(faces, np.array(ids))

recognizer.write("trainer/trainer.yml")


print('Done with Training')


