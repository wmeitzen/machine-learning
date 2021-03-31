
import cv2
import time

cap = cv2.VideoCapture(0)

DELAY_SECONDS = 5

ret, frame = cap.read()
cv2.imshow("Webcam", frame)

time.sleep(5)

cap.release()
cv2.destroyAllWindows()


