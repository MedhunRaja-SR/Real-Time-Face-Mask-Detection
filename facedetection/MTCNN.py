import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
img = cv2.imread('15.jpg')
faces = detector.detect_faces(img)
filename = 'Mtcnn.jpg'
for result in faces:
    x, y, w, h = result['box']
    x1, y1 = x + w, y + h
    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(filename, img)
print("Image saved...")