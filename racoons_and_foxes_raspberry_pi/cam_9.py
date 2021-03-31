import cv2

cam = None
try:
    cam = cv2.VideoCapture(0)
except Exception as inst:
    pass
if cam is None:
    print("Camera not found")
    exit
    
#cv2.namedWindow("test")
ret = False
frame = None
try:
    ret, frame = cam.read()
except Exception as inst:
    pass
if ret == False:
    print("failed to grab frame")
    exit
img_name = "pic.png"
cv2.imwrite(img_name, frame)
print("file written!")

cam.release()

cv2.destroyAllWindows()