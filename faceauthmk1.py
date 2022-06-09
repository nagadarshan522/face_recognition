import cv2
import numpy as np
from PIL import Image

faces=()
import sys
cascPath = '/Users/nagadarshan/Downloads/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
while faces is ():

    ret, frame1 =  video_capture.read()
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))


    for (x1, y1, w1, h1) in faces:
        cv2.rectangle(frame1, (x1,y1), (x1+w1, y1+h1), (0,255,0), 2)

    cv2.imshow('Video', frame1)

    if cv2.waitKey(1500) & 0xFF== ord('q'):
        break


video_capture.release()
faces=()
cv2.destroyAllWindows()

print(frame1)

im = Image.fromarray(frame1)
im.save("me1.jpeg")



faceCascade1 = cv2.CascadeClassifier(cascPath)
faces = ()
video_capture = cv2.VideoCapture(0)
while faces is ():

    ret1, frame2 =  video_capture.read()
    gray1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))


    for (x2, y2, w2, h2) in faces:
        cv2.rectangle(frame2, (x2,y2), (x2+w2, y2+h2), (0,255,0), 2)

    cv2.imshow('Video', frame2)

    if cv2.waitKey(1500) & 0xFF== ord('q'):
        break


video_capture.release()
faces1=()
cv2.destroyAllWindows()


frame1 = frame1[:,:,0]
frame3 = frame1[y1:y1+h1 , x1:x1+w1]

frame2 = frame2[:,:,0]
frame4 = frame2[y2:y2+h2, x2:x2+w2]

im = Image.fromarray(frame3)
im.save('me2.jpeg')
im = Image.fromarray(frame4)
im.save("me3.jpeg")


registered_face = []
new_face = []

for v in np.nditer(frame3):
    registered_face.append(v)
for w in np.nditer(frame4):
    new_face.append(w)


if(len(registered_face)>=len(new_face)):
    leng = len(new_face)
else:
    leng = len(registered_face)


q=0
for i in range(leng):
    registered_face[i] = (registered_face[i]/10).astype(int)
    new_face[i] = (new_face[i]/10).astype(int)
    if registered_face[i]==new_face[i]:
        q=q+1

r=(q/leng)
print(r)

if (r>=0.1):
    print("hey, you are authorised")





