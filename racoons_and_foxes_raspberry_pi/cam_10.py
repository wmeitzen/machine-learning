
import cv2
import time
import numpy as np
from datetime import datetime
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img


# load an image and predict the class
def run_example(filename, model, should_be):
	print(f"image filename: {filename}")
	# load the image
	img = load_image(filename)
	# load model
	result_array = model.predict(img)
	# predict the class
	result = np.array(result_array)[0][0]
	print(f"result (close to 0.0=false, close to 1.0=true): {str(round(result, 2))}")
	predicted_answer = round(result)
	print(f"Should be: {should_be}")
	if predicted_answer == 1:
		print("Predicted answer: true (racoon or fox)")
	else:
		print("Predicted answer: false (not a racoon or a fox)")
	print("")

# load model
model = load_model(f"final_model.h5")

while True:
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

    run_example(filename='pic.png', model=model, should_be='an image')
