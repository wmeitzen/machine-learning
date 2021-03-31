
import cv2
cam = cv2.VideoCapture(0)
img = cam.read()

cv2.namedWindow("camera", cv2.WINDOW_AUTOSIZE)
cv2.imshow("camera", img)
cv2.waitKey(0)
cv2.destroywindow("camera")

