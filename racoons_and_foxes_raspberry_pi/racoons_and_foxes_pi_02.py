
import time
import tensorflow as tf
#from matplotlib import pyplot
#from matplotlib.image import imread
from os import makedirs
#from os import listdir
#from shutil import copyfile
from datetime import datetime
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os
from shutil import copyfile
#from random import seed
import random
import sys
from matplotlib import pyplot
#from keras.utils import to_categorical
#from keras.models import Sequential
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
#from keras.layers import Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import shutil

from keras.applications.vgg16 import VGG16
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
	print(f"exact result (close to 0.0=false, close to 1.0=true): {str(round(result, 2))}")
	predicted_answer = round(result)
	#print(f"type(result): {type(result)}")
	#print(f"predicted_answer: {predicted_answer}")
	print(f"Should be: {should_be}")
	if predicted_answer == 1:
		print("Predicted answer: true (racoon or fox)")
	else:
		print("Predicted answer: false (not a racoon or a fox)")
	print("")
	#print(result[0])

# load model
model_type='vgg16_transfer'
model = load_model(f"final_model.h5")

# entry point, run the examples
start_time = time.time()
run_example(filename='calendar_fox_1.jpg', model=model, should_be='a fox')
run_example(filename='calendar_fox_2.jpg', model=model, should_be='a fox')
run_example(filename='calendar_fox_3.jpg', model=model, should_be='a fox, b/w')
run_example(filename='pig_1.jpg', model=model, should_be='pig_1')
run_example(filename='sheep_1.jpg', model=model, should_be='sheep_1')
run_example(filename='cow_1.jpg', model=model, should_be='cow_1')
finish_time = time.time()

#print('Took %f sec to predict all 6 examples' % (finish_time-start_time))
elapsed = finish_time - start_time
each = elapsed / 6
print(f"Took {round(elapsed, 1)} sec to predict all 6 examples, {round(each, 1)} sec each")
