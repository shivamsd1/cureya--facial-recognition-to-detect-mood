import os
import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
from mtcnn import MTCNN

model = load_model("model.h5")


#face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = MTCNN()


cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    col =cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = detector.detect_faces(col)

    for face in faces_detected:
        x,y,w,h = face['box']
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(150,150,150,0),thickness=2)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = "Emotion detected : " + emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,255), 1)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows