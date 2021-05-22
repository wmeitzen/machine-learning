
import tensorflow as tf
#from matplotlib import pyplot
#from matplotlib.image import imread
from os import makedirs
#from os import listdir
#from shutil import copyfile
from datetime import datetime
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
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
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import shutil
from timeit import default_timer as timer
from PIL import Image, ImageStat, ImageFilter
import glob
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

start = timer()

# load dogs vs cats dataset, reshape and save to a new file
#from os import listdir
#from numpy import asarray
#from numpy import save
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#from numpy import load

# plot diagnostic learning curves
def summarize_diagnostics(history):
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
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

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
def run_test_harness(epochs, fit_generator_verbose, evaluate_generator_verbose):
	# define model
	#fit_generator_verbose = 1
	#evaluate_generator_verbose = 1
	model = define_model_vgg16_transfer()
	#model = define_model_vgg16_transfer_out_shape()

	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	batch_size = 64 # - original
	#batch_size = 128
	train_it = datagen.flow_from_directory(processed_dataset_home + 'train/',
			class_mode='binary', batch_size=batch_size, target_size=(224, 224))
	test_it = datagen.flow_from_directory(processed_dataset_home + 'test/',
			class_mode='binary', batch_size=batch_size, target_size=(224, 224))

	# original epochs = 1
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=epochs, verbose=fit_generator_verbose) # epochs=10

	#model.save(f"final_model.h5") # not yet, I guess

	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=evaluate_generator_verbose)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

def run_finalize_harness(epochs, fit_generator_verbose):
	model = define_model_vgg16_transfer()
	datagen = ImageDataGenerator(featurewise_center=True)
	datagen.mean = [123.68, 116.779, 103.939]
	finalize_it = datagen.flow_from_directory(processed_dataset_home + 'finalize/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	# epochs = 10
	model.fit_generator(finalize_it, steps_per_epoch=len(finalize_it), epochs=epochs, verbose=fit_generator_verbose)
	model.save(f"final_model.h5") # 82 mb
	#model.save(f"final_model.tfsm") # ?? mb

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


#Function calculates the difference between the CURRENT image file RMS value and RMS values calculated at start
def average_diff(v1, v2, show_rms_info = False):
	duplicate = False
	diff = 0.01 # original
	#diff = 0.1
	if len(v1) >= 3 and len(v2) >= 3: # jpg
		calculated_rms_difference = [v1[0]-v2[0], v1[1]-v2[1], v1[2]-v2[2]]
		if calculated_rms_difference[0] < diff and calculated_rms_difference[0] > -diff and \
				calculated_rms_difference[1] < diff and calculated_rms_difference[1] > -diff and \
				calculated_rms_difference[2] < diff and calculated_rms_difference[2] > -diff:
			duplicate = True
	elif len(v1) >= 1 and len(v2) >= 1: # png
		calculated_rms_difference = [v1[0] - v2[0]]
		if calculated_rms_difference[0] < diff and calculated_rms_difference[0] > -diff:
			duplicate = True
	if show_rms_info == True:
		print(f'rms info: calculated_rms_difference = {calculated_rms_difference}')
	return duplicate

def delete_dupes_using_rms(images, rms_pixels):
	old_percentage = -100
	for image_file_count in range(len(images) - 1):
		new_percentage = round(image_file_count / (len(images) - 1) * 100)
		new_string = f'{image_file_count} of {len(images) - 1}, {new_percentage} % complete'
		if new_percentage >= old_percentage + 10:
			print(new_string)
			old_percentage = new_percentage
		image_file = images[image_file_count]
		if not(os.path.exists(image_file)):
			continue
		rms_file_1_binary = Image.open(fp=image_file)
		rms_file_1_properties = ImageStat.Stat(rms_file_1_binary).mean
		for rms_file_count in range(image_file_count + 1, len(images)):
			rms_file_2_properties = rms_pixels[rms_file_count]
			rms_file_2 = rms_file_2_properties[len(rms_file_2_properties) - 1]
			is_duplicate = average_diff(rms_file_1_properties, rms_file_2_properties)
			if is_duplicate:
				if os.path.exists(image_file) and os.path.exists(rms_file_2):
					print(f"Found dupe: {image_file} and {rms_file_2}")
					print(f"  Deleting dupe: {rms_file_2}")
					os.remove(rms_file_2)

def delete_duplicate_image_hashes(root_directory, subdirectories=None):
	print(f"Retrieving files from {root_directory}")
	if subdirectories is not None:
		print(f"then under {subdirectories}")
	image_files = []
	rms_pixels = []
	if subdirectories is None:
		for filename in glob.iglob(pathname=os.path.join(root_directory, '**'), recursive=True):
			if os.path.isfile(filename):
				image_files.append(filename)
	else:
		for subdir in subdirectories:
			for filename in glob.iglob(pathname=os.path.join(root_directory, subdir, '**'), recursive=True):
				if os.path.isfile(filename):
					image_files.append(filename)
	# Create a list with all the Image RMS values. These are used to compare to the CURRENT image file in list
	print(f"Retrieving stats for {len(image_files)} images")
	for x in image_files:
		# print(x)
		compare_image = Image.open(x)  # Image.open(os.path.join(image_folder, x))
		rms_pixel = ImageStat.Stat(compare_image).mean
		rms_pixel.append(x)
		rms_pixels.append(rms_pixel)
	# print(rms_pixel)
	# Driver code, runs the script
	print(f"Removing duplicates")
	delete_dupes_using_rms(images=image_files, rms_pixels=rms_pixels)

def delete_duplicate_images():
	shutil.rmtree(path=processing_deduplicated_images_dataset_home, ignore_errors=True)
	print(f"Copying files from unprocessed directory ({unprocessed_dataset_home}) to processing directory ({processing_deduplicated_images_dataset_home})")
	shutil.copytree(src=unprocessed_dataset_home, dst=processing_deduplicated_images_dataset_home)
	delete_duplicate_image_hashes(root_directory=processing_deduplicated_images_dataset_home)

# - assumes the "root" directory has a set of directories 1 level deep, with only files in
# each of those directories
def write_bw_blurred_images():
	src = processing_deduplicated_images_dataset_home
	gaussian_blur_radius = 5 # original: 3
	print(f"Generating additional B&W blurred images from originals in {src}")
	# delete bw_blurred directories, if any
	for negative_or_positive in os.listdir(path=src):
		if negative_or_positive.endswith('_bw_blurred'):
			shutil.rmtree(path=os.path.join(src, negative_or_positive))
	for negative_or_positive in os.listdir(path=src):
		makedirs(name=os.path.join(src, negative_or_positive + '_bw_blurred'), exist_ok=False)
		for file in os.listdir(path=os.path.join(src, negative_or_positive)):
			if file.endswith('.png'):
				img = Image.open(fp=os.path.join(src, negative_or_positive, file)).convert('LA').filter(ImageFilter.GaussianBlur(radius=gaussian_blur_radius))
				file = file + '.bw_g.png'
			else:
				img = Image.open(fp=os.path.join(src, negative_or_positive, file)).convert('L').filter(ImageFilter.GaussianBlur(radius=gaussian_blur_radius))
				file = file + '.bw_g.jpg'
			img.save(os.path.join(src, negative_or_positive + '_bw_blurred', file))

def recreate_train_test_validate_images_subdirs():
	# create directories
	src = processing_deduplicated_images_dataset_home
	print(f"Copying and distributing files from directory ({src}) to processed directory ({processed_dataset_home})")
	subdirs = ['train/', 'test/', 'validate/', 'finalize/']
	for subdir in subdirs:
		newdir = processed_dataset_home + subdir
		shutil.rmtree(path=newdir, ignore_errors=True)
	for subdir in subdirs:
		# create label subdirectories
		labeldirs = ['positive/', 'negative/']
		for labldir in labeldirs:
			newdir = processed_dataset_home + subdir + labldir
			makedirs(newdir, exist_ok=True)

	#from os import makedirs
	#from os import listdir

	# seed random number generator
	#seed(1)
	random.seed(datetime.now())
	# define ratio of pictures to use for training, testing, and validating
	# - add them up to 1.0
	validation_ratio = 0.02 # must be smaller than, and not equal to, test_ratio (makes my code simpler)
	test_ratio = 0.15 # must be a lot lower than train_ratio
	train_ratio = 0.83
	# copy training dataset images into subdirectories
	for directory_or_filename in os.listdir(path=src):
		check_directory_or_filename = src + directory_or_filename + '/'
		directory_alone = directory_or_filename
		if os.path.isdir(check_directory_or_filename):
			directory = check_directory_or_filename
			for file in os.listdir(path=directory):
				if directory_alone.startswith('positive_'):
					postive_or_negative_directory = labeldirs[0] # positive/
				if directory_alone.startswith('negative_'):
					postive_or_negative_directory = labeldirs[1] # negative/
				r = random.random()
				if r <= validation_ratio:
					test_train_validate = 'validate/'
				elif r <= test_ratio:
					test_train_validate = 'test/'
				else:
					test_train_validate = 'train/'
				fq_source_filename=directory + file
				# - copy to validate, test, and train directories
				fq_destination_filename = processed_dataset_home + test_train_validate + postive_or_negative_directory + file
				if not(os.path.exists(fq_destination_filename)):
					copyfile(fq_source_filename, fq_destination_filename)
				# - copy to finalize directory
				fq_destination_filename = processed_dataset_home + 'finalize/' + postive_or_negative_directory + file
				if not(os.path.exists(fq_destination_filename)):
					copyfile(fq_source_filename, fq_destination_filename)

def run_validation_tests(model):
	validation_directory = processed_dataset_home + 'validate/'
	ground_truth_matches_prediction = 0
	ground_truth_mismatches_prediction = 0
	file_count = 0
	negative_ground_truth_file_count = 0
	positive_ground_truth_file_count = 0
	ground_truth_matched_negative_prediction = 0
	ground_truth_matched_positive_prediction = 0
	ground_truth_mismatched_negative_prediction = 0
	ground_truth_mismatched_positive_prediction = 0
	for negative_or_positive in os.listdir(path=validation_directory):
		image_directory = validation_directory + negative_or_positive + '/'
		for file in os.listdir(image_directory):
			if negative_or_positive == 'negative':
				negative_ground_truth_file_count = negative_ground_truth_file_count + 1
				ground_truth = 0.0
			else:
				positive_ground_truth_file_count = positive_ground_truth_file_count + 1
				ground_truth = 1.0
			file_count = file_count + 1
			image_filename = image_directory + file
			# load the image
			binary_image_data = load_image(image_filename)
			result_array = model.predict(binary_image_data)
			# predict the class
			result = np.array(result_array)[0][0]
			#print(f"exact result (close to 0.0=false, close to 1.0=true): {str(round(result, 2))}")
			predicted_answer = round(result)
			if predicted_answer == ground_truth:
				ground_truth_matches_prediction = ground_truth_matches_prediction + 1
			if predicted_answer != ground_truth:
				ground_truth_mismatches_prediction = ground_truth_mismatches_prediction + 1
			if predicted_answer == ground_truth and ground_truth == 0.0: # it matched and was negative
				ground_truth_matched_negative_prediction = ground_truth_matched_negative_prediction + 1
			if predicted_answer == ground_truth and ground_truth == 1.0:  # it matched and was positive
				ground_truth_matched_positive_prediction = ground_truth_matched_positive_prediction + 1
			if predicted_answer != ground_truth and ground_truth == 0.0:  # it mismatched and was negative
				ground_truth_mismatched_negative_prediction = ground_truth_mismatched_negative_prediction + 1
				print(f"{image_filename} predicted positive, should be negative")
			if predicted_answer != ground_truth and ground_truth == 1.0:  # it mismatched and was positive
				ground_truth_mismatched_positive_prediction = ground_truth_mismatched_positive_prediction + 1
				print(f"{image_filename} predicted negative, should be positive")
	print(f"positive match::positive total = {ground_truth_matched_positive_prediction}::{positive_ground_truth_file_count} = {str(round(ground_truth_matched_positive_prediction / positive_ground_truth_file_count * 100, 2))} %")
	print(f"negative match::negative total = {ground_truth_matched_negative_prediction}::{negative_ground_truth_file_count} = {str(round(ground_truth_matched_negative_prediction / negative_ground_truth_file_count * 100, 2))} %")
	print(f"sum match::sum total = {ground_truth_matches_prediction}::{file_count} = {str(round(ground_truth_matches_prediction / file_count * 100, 2))} %")

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # recent update to pycharm shows errors until this code is added

unprocessed_dataset_home = 'unprocessed_images/'
processing_deduplicated_images_dataset_home = 'processing_deduplicated_images/'
processed_dataset_home = 'processed_images/'

# - only need to set "initialize = True" once (unless you want to use more / less / different images)
# delete and recreate processed_images directory
# separate files into directories
# generate the "final_model.h5"
# Once the .h5 file is created, use the run_example function to see if an image is a fox/racoon or not
# with a good PC and GPU, this process takes 10 min
# with a good PC and no GPU, this process takes hours

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # uncomment to avoid using a GPU

initialize = True
if initialize == True:
	now = datetime.now()
	print (now.strftime("%Y-%m-%d %H:%M"))
	delete_duplicate_images() # uncomment later
	write_bw_blurred_images()
	recreate_train_test_validate_images_subdirs()
	# - epochs 4: seems best
	run_test_harness(epochs=3, fit_generator_verbose = 1, evaluate_generator_verbose = 1)
	run_finalize_harness(epochs=3, fit_generator_verbose = 1)

# load model
model = load_model(f"final_model.h5")

# entry point, run the examples
run_validation_tests(model=model)
run_example(filename='calendar_fox_1.jpg', model=model, should_be='true, a fox')
run_example(filename='calendar_fox_2.jpg', model=model, should_be='true, a fox')
run_example(filename='calendar_fox_3.jpg', model=model, should_be='true, a fox, b/w')

run_example(filename='cow_1.jpg', model=model, should_be='false, a cow')
run_example(filename='pig_1.jpg', model=model, should_be='false, a pig')
run_example(filename='sheep_1.jpg', model=model, should_be='false, a sheep')

run_example(filename='fox_example_wilbur_1.jpg', model=model, should_be='true, a fox')

run_example(filename='wilbur_racoon_1.jpg', model=model, should_be='true, a racoon')
run_example(filename='wilbur_owl_chicken_1.jpg', model=model, should_be='false, chicken and owl')
run_example(filename='wilbur_armadillo_1.jpg', model=model, should_be='false, armadillo')
run_example(filename='wilbur_racoon_2.jpg', model=model, should_be='true, a racoon')

duration = timer() - start
print(f"{str(round(duration, 1))} sec")
