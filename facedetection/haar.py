import os
import cv2
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('15.jpg')
filename = 'haar.jpg'
directory = r'C:\Users\Admin\PycharmProjects\facedetection\detected_images'
faces = classifier.detectMultiScale(img)
for result in faces:
    x, y, w, h = result
    x1, y1 = x + w, y + h
    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    os.chdir(directory)
    cv2.imwrite(filename, img)
print("Image saved...")

