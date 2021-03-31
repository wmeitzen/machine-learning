
import cv2
import time

cap = cv2.VideoCapture(0)
frames = []
start_time = time.time()

LAG_SECONDS = 3

while True:
    ret, frame = cap.read()
    frames.append(frame)
    if time.time() - start_time > LAG_SECONDS:
        cv2.imshow("Webcam", frames.pop(0))
    if cv2.waitKey(1) & 0xFF == 27: # use ESC to quit
        break

cap.release()
cv2.destroyAllWindows()


