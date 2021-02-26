
# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
from os import makedirs
from os import listdir
from shutil import copyfile
import datetime

# define location of dataset
folder = 'train/'

"""
# show dogs
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'dog.' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
"""

"""
# show cats
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'cat.' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
"""

# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

"""
# - we only need to do this once
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
	# determine class
	output = 0.0
	if file.startswith('cat'):
		output = 1.0
	# load image
	photo = load_img(folder + file, target_size=(200, 200))
	# convert to numpy array
	photo = img_to_array(photo)
	# store
	photos.append(photo)
	labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('dogs_vs_cats_photos.npy', photos)
save('dogs_vs_cats_labels.npy', labels)
"""

from numpy import load

"""
# load and confirm the shape
photos = load('dogs_vs_cats_photos.npy')
labels = load('dogs_vs_cats_labels.npy')
print(photos.shape, labels.shape) # - outputs "(25000, 200, 200, 3) (25000,)"
"""

"""
# create directories
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['dogs/', 'cats/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		makedirs(newdir, exist_ok=True)
"""

from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

"""
# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = 'train/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/'
	if random() < val_ratio:
		dst_dir = 'test/'
	if file.startswith('cat'):
		dst = dataset_home + dst_dir + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + dst_dir + 'dogs/'  + file
		copyfile(src, dst)
"""

# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

# plot diagnostic learning curves
def summarize_diagnostics(history, filenamepart):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_' + filenamepart + '_plot.png')
	pyplot.close()

# define cnn model
def define_model_vgg1():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

def define_model_vgg2():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

def define_model_vgg3():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

def define_model_vgg3_dropout():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

def define_model_vgg16_transfer():
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# run the test harness for evaluating a model
def run_test_harness(model_type):
	# define model
	#model = define_model()
	if (model_type not in ['vgg1', 'vgg2', 'vgg3', 'vgg3_dropout', 'vgg3_augmentation', 'vgg16_transfer']):
		print(f"Model type not recognized: {model_type}")
		return
	print(f"Model type: {model_type}")
	if (model_type == 'vgg1'):
		model = define_model_vgg1()
	if (model_type == 'vgg2'):
		model = define_model_vgg2()
	if (model_type in ['vgg3', 'vgg3_augmentation']):
		model = define_model_vgg3()
	if (model_type == 'vgg3_dropout'):
		model = define_model_vgg3_dropout()
	if (model_type == 'vgg16_transfer'):
		model = define_model_vgg16_transfer()

	# prepare iterators and data generators
	if (model_type == 'vgg3_augmentation'):
		train_datagen = ImageDataGenerator(rescale=1.0/255.0,
			width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
		test_datagen = ImageDataGenerator(rescale=1.0/255.0)
		train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
			class_mode='binary', batch_size=64, target_size=(200, 200))
		test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
			class_mode='binary', batch_size=64, target_size=(200, 200))
	elif (model_type == 'vgg16_transfer'):
		datagen = ImageDataGenerator(featurewise_center=True)
		# specify imagenet mean values for centering
		datagen.mean = [123.68, 116.779, 103.939]
		train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
											   class_mode='binary', batch_size=64, target_size=(224, 224))
		test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
											  class_mode='binary', batch_size=64, target_size=(224, 224))
	else:
		datagen = ImageDataGenerator(rescale=1.0/255.0)
		train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
			class_mode='binary', batch_size=64, target_size=(200, 200))
		test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
			class_mode='binary', batch_size=64, target_size=(200, 200))

	# fit model
	if (model_type in ['vgg1', 'vgg2']):
		history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
			validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	elif (model_type == 'vgg16_transfer'):
		history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
									  validation_data=test_it, validation_steps=len(test_it), epochs=1, verbose=0) # epochs=10
	else:
		history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
			validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)

	model.save(f"final_model_{model_type}.h5")

	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history, filenamepart=model_type)

"""
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M"))
# 10 epochs / 13 min / Model type: vgg16_transfer / > 97.874
# 1 epoch / 4 sec / Model type: vgg16_transfer / > 97.573
run_test_harness(model_type='vgg16_transfer')

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M"))
run_test_harness(model_type='vgg3_augmentation') # 1 hr 40 min / Model type: vgg3_augmentation / > 85.896

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M"))
run_test_harness(model_type='vgg3_dropout') # 22 min / Model type: vgg3_dropout / > 81.834

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M"))
run_test_harness(model_type='vgg3') # 19 min / Model type: vgg3 / > 80.739

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M"))
run_test_harness(model_type='vgg1') # 9 min / Model type: vgg1 / > 72.664

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M"))
run_test_harness(model_type='vgg2') # 10 min / Model type: vgg2 / > 76.075

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M"))
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

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
	print(f"model type filename: final_model_{model_type}.h5")
	# load the image
	img = load_image(filename)
	# load model
	# predict the class
	result_array = model.predict(img)
	result = np.array(result_array)[0][0]
	print(f"exact result (close to 0.0=cat, close to 1.0=dog): {result}")
	predicted_answer = round(result)
	#print(f"type(result): {type(result)}")
	#print(f"predicted_answer: {predicted_answer}")
	print(f"Should be: {should_be}")
	if (predicted_answer == 1):
		print("Predicted answer: a dog")
	else:
		print("Predicted answer: a cat")
	print("")
	#print(result[0])

# load model
model_type='vgg16_transfer'
print(f"model_type: {model_type}")
model = load_model(f"final_model_{model_type}.h5")

# entry point, run the examples
run_example(filename='unseen_sample_image_1.jpg', model=model, should_be='a dog')
run_example(filename='kitten-440379.jpg', model=model, should_be='a kitten')
run_example(filename='tiger.jpg', model=model, should_be='a tiger')
run_example(filename='wolf.png', model=model, should_be='wolf in art')
run_example(filename='wolf_2.jpg', model=model, should_be='wolf')
run_example(filename='harry.JPG', model=model, should_be='harry, our cat')
run_example(filename='card_with_puppy.JPG', model=model, should_be='puppy on a card')
run_example(filename='dog with cat ears.jpg', model=model, should_be='dog with cat ears')
run_example(filename='dog with bunny ears.jpg', model=model, should_be='dog with bunny ears')
run_example(filename='stuffed animal dog.jpg', model=model, should_be='stuffed animal dog')
