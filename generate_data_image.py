import cv2
import imutils
import argparse

# attention, generate photos need this file "haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier("shape predictor/haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)
index = 0

while(True):
    rect, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=3, minNeighbors=5)
    img_item_base = "images/happy/"                   # configure before execution, create directory if necessary
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_color = frame[y:y + h, x:x + w]
        roi_color = imutils.resize(roi_color, width=300)
        img_item = img_item_base + str(index) + ".png"
    if cv2.waitKey(20) & 0xFF == ord('c'):                          # press c to save and capture photo
        cv2.imwrite(img_item, roi_color)
        index += 1
        print("image written {}".format(img_item))
    cv2.imshow('head', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):                          # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
