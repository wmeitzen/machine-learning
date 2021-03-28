
import math
import os
from PIL import Image, ImageStat, ImageChops
import shutil
import glob
import sys
import numpy as np

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
    old_percentage = -1
    for image_file_count in range(len(images) - 1):
        new_percentage = str(round(image_file_count / (len(images) - 1) * 100))
        new_string = f'{image_file_count} of {len(images) - 1}, {new_percentage} % complete'
        if old_percentage != new_percentage:
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
                #if image_file != rms_file_2_properties[3]:  # skip the exact same files
                #if image_file != rms_file_2_properties[len(rms_file_2_properties)-1]:  # skip the exact same files
                #print(f"Found dupe: {rms_file_2_properties[3]} and {image_file}")
                if os.path.exists(image_file) and os.path.exists(rms_file_2):
                    print(f"Found dupe: {image_file} and {rms_file_2}")
                    print(f"  Deleting dupe: {rms_file_2}")
                    os.remove(rms_file_2)

def delete_duplicate_images(root_directory, subdirectories=None):
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
    #Create a list with all the Image RMS values. These are used to compare to the CURRENT image file in list
    print(f"Retrieving stats for {len(image_files)} images")
    for x in image_files:
        #print(x)
        compare_image = Image.open(x) #Image.open(os.path.join(image_folder, x))
        rms_pixel = ImageStat.Stat(compare_image).mean
        rms_pixel.append(x)
        rms_pixels.append(rms_pixel)
        #print(rms_pixel)
    #Driver code, runs the script
    print(f"Removing duplicates")
    delete_dupes_using_rms(images=image_files, rms_pixels=rms_pixels)

root = 'processing_duplicate_images'
delete_duplicate_images(root_directory=root, subdirectories=None)

exit()

root = 'processed_images'
subdirs = ['train', 'test', 'validate']
delete_duplicate_images(root_directory=root, subdirectories=subdirs)

subdirs = ['finalize']
delete_duplicate_images(root_directory=root, subdirectories=subdirs)
