import cv2 as cv
import os
import numpy as np

people = []
for i in os.listdir(r'D:\Docs (huzaifa)\semester 1\focp\project\openCV\photos\celebs'):
    people.append(i)
people.pop()

DIR = r'D:\Docs (huzaifa)\semester 1\focp\project\openCV\photos\celebs'

haar_cascade = cv.CascadeClassifier('openCV/code/faces/haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        label = people.index(person)
        path = os.path.join(DIR, person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 4)

            for x,y,w,h in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('training done-----------------------')

print(f'Length of features = {len(features)}')
print(f'Length of labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_Recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on the features list and the labels list
face_Recognizer.train(features,labels)

face_Recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('Labels.npy',labels)