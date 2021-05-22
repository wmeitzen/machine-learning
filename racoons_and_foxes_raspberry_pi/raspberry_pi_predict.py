
#import cv2
import time
import numpy as np
from datetime import datetime
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
#from matplotlib import pyplot as plt
import pygame
import pygame.camera
from pygame.locals import *

# load and prepare the image
def transform_frame_to_binary_image(frame):
    # load the image
    print("Converting frame to binary image")
    #img = Image.fromarray(frame, 'RGB') # fails
    #img = frame
    pil_string_image = pygame.image.tostring(frame, 'RGB', False)
    img = Image.frombytes('RGB', (224, 224), pil_string_image)
    #img.show()
    #plt.imshow(img, interpolation = 'nearest')
    #plt.show()
    img = img.resize((224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

# load an image and predict the class
def run_frame_example(model, frame):
	# load the image
	img = transform_frame_to_binary_image(frame)
	# load model
	print("Predicting")
	start_time = time.time()
	result_array = model.predict(img)
	# predict the class
	result = np.array(result_array)[0][0]
	finish_time = time.time()
	elapsed = finish_time - start_time
	print(f"Predicting took {round(elapsed, 1)} sec")
	print(f"result (close to 0.0=false, close to 1.0=true): {str(round(result, 2))}")
	predicted_answer = round(result)
	if predicted_answer == 1:
		print("Predicted answer: true (racoon or fox)")
	else:
		print("Predicted answer: false (not a racoon or a fox)")
	print("")

def load_image(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

def run_example(filename, should_be):
    img = load_image(filename)
    print("Predicting")
    start_time = time.time()
    result_array = model.predict(img)
    result = np.array(result_array)[0][0]
    finish_time = time.time()
    elapsed = finish_time - start_time
    print(f"Predicting took {round(elapsed, 1)} sec")
    print(f"result (close to 0.0 = false, close to 1.0 = true): {str(round(result, 2))}")
    predicted_answer = round(result)
    print(f"Should be: {should_be}")
    if predicted_answer == 1:
        print("Predicted answer: true (racoon or fox)")
    else:
        print("Predicted answer: false (not racoon or fox)")
    print("")

import pygame
import pygame.camera
from pygame.locals import *

pygame.init()
pygame.camera.init()
size = (640, 480)
#size = (224, 224)
display = pygame.display.set_mode(size, 0)
clist = pygame.camera.list_cameras()
#print(clist)
if not clist:
    print("No camera detected. Aborting.")
    exit()
cam = pygame.camera.Camera(clist[0], size)
#print(cam)
cam.start()
snapshot = pygame.surface.Surface(size, 0, display)

# load model
model = load_model(f"final_model.h5")

print("Predicting from image files.")
run_example(filename = "calendar_fox_1.jpg", should_be = True)
run_example(filename = "pig_1.jpg", should_be = False)
run_example(filename = "calendar_fox_2.jpg", should_be = True)
run_example(filename = "sheep_1.jpg", should_be = False)

exit()

#cap = Capture()
while True:
    events = pygame.event.get()
    for e in events:
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            # close the camera safely
            cam.stop()
            exit()
    #self.get_and_flip()
    if cam.query_image():
        snapshot = cam.get_image(snapshot)

    # blit it to the display surface.  simple!
    display.blit(snapshot, (0,0))
    pygame.display.flip()
    run_frame_example(model = model, frame = snapshot)


