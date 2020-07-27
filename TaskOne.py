import cv2
# Image reading
imgOld = cv2.imread("Images/people.jpg")
# Image reshape for better detection according to sliding window
img = cv2.resize(imgOld, (512, 1024))
# Implementation of Histogram of Oriented Gradients as DALAL and Triggs 2205 paper
hog = cv2.HOGDescriptor()
# Used Support vector machines on pre trained data
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = img.copy()
# people detection in the image
(found, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
# drawing of  bounding boxes
for (x, y, w, h) in found:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
cv2.imshow("original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
