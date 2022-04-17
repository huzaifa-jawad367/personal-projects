# there are 2 types of classifiers; haar cascades and local binary patterns(lower noise as compared to haar cascades)
import cv2 as cv
from cv2 import INTER_AREA

img = cv.imread('photos/person.jpg')
# img = cv.resize(img, (img.shape[0]//2,img.shape[0]//2),interpolation=INTER_AREA)
#cv.imshow('person',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray person', gray)

haar_cascade = cv.CascadeClassifier('code/faces/haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 7) # by minimising these values openCV's haar_cascades are more prone to noise. not the most effective.

for x,y,w,h in faces_rect:
    cv.rectangle(img, (x,y),(x+w, y+h), (0,255,0), thickness=2)
cv.imshow('face highlighted', img)

print(faces_rect)
print(f'Number of faces found = {len(faces_rect)}')

cv.waitKey(0)