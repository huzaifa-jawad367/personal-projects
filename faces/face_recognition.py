from cv2 import INTER_AREA, INTER_LINEAR
import numpy as np
import cv2 as cv
import os 

haar_cascade = cv.CascadeClassifier('openCV/code/faces/haar_face.xml')

people = []
for i in os.listdir(r'D:\Docs (huzaifa)\semester 1\focp\project\openCV\photos\celebs'):
    people.append(i)
people.pop()

features = np.load('features.npy')
labels = np.load('labels.npy')

face_Recognizer = cv.face.LBPHFaceRecognizer_create()
face_Recognizer.read('face_trained.yml')

img = cv.imread('openCV/photos/celebs/test_data/IMG_3211.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('person',gray)

# detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 7)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_Recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidince of {confidence}')

    cv.putText(img, str(people[label]),(20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected image',img)

cv.waitKey(0)