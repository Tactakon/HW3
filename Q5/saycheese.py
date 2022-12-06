import numpy as np
import cv2
import os
if not os.path.exists('images'):
    os.makedirs('images')

faceface = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
camera.set(3,640)
camera.set(4,480)
count = 0
font = cv2.FONT_HERSHEY_SIMPLEX

facefind = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n enter user id (MUST be an integer) and press <return> -->  ')
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")


while(True):
    ret, img = camera
.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceface.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("./images/Users." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

 
    k = cv2.waitKey(100) & 0xff
    if k < 30:
        break
    elif count >= 30:
         break

print("\n [INFO] Exiting Program.")
camera.release()
cv2.destroyAllWindows()
