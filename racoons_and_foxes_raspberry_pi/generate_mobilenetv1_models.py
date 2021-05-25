'''
Generates a dummy mobilenetv1 model
'''

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
import time

class TFliteModel:
    def __init__(self, tflite_model_file):
        self.interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_file))
        self.input_index_quant = self.interpreter_quant.get_input_details()[0]["index"]
        self.output_index_quant = self.interpreter_quant.get_output_details()[0]["index"]
        #self.interpreter_quant.resize_tensor_input(self.input_index_quant, [batch_size, 112, 112, 3])
        self.interpreter_quant.allocate_tensors()
    def predict(self, test_image):
        self.interpreter_quant.set_tensor(self.input_index_quant, test_image)
        self.interpreter_quant.invoke()
        predictions = self.interpreter_quant.get_tensor(self.output_index_quant)
        return predictions

def transfer_learn_mbv1():
    '''
    Adds a classification head on top of
    pretrained mobilenet v1 model.
    Note: use tf.keras.applications.mobilenet.preprocess_input
    or perprocess_mobilenetv1 before passing the input tensor
    to model.predict.
    '''
    model = tf.keras.applications.MobileNet(input_shape=(224,224,3), 
                                            alpha=0.25,
                                            include_top=False,
                                            weights='imagenet')

    for layer in model.layers: # Do not train layers from base model
        layer.trainable = False
    
    # Classification head to classify features from mbv1 base
    x = Flatten()(model.layers[-1].output)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dense(1, activation='sigmoid')(x)
    

    model = Model(inputs=model.inputs, outputs=x)
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def perprocess_mobilenetv1(img):
    '''
    Takes in an image with pixel value [0, 255]
    and converts the image pixel for mbv1 model.
    Same as tf.keras.applications.mobilenet.preprocess_input
    see: https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/applications/imagenet_utils.py#L259
    '''
    img = img / 127.5
    img = img - 1.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def convert_to_tflite(keras_model, tflite_filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(tflite_filename,  "wb") as f:
        f.write(tflite_model)

model = transfer_learn_mbv1()
dummy_input = (np.random.uniform(low=0.0, high=1.0, size=(224,224,3)) * 255).astype(np.uint8)
output = model.predict(perprocess_mobilenetv1(dummy_input))

tflite_filename = "mbv1.tflite"
convert_to_tflite(keras_model=model, tflite_filename=tflite_filename)
tflite_model = TFliteModel(tflite_filename)
tflite_model.predict(perprocess_mobilenetv1(dummy_input))

start_time = time.time()
output = model.predict(perprocess_mobilenetv1(dummy_input))
finish_time = time.time()
elapsed = finish_time - start_time
print("Keras predicting took {}".format(elapsed))

start_time = time.time()
tflite_model.predict(perprocess_mobilenetv1(dummy_input))
finish_time = time.time()
elapsed = finish_time - start_time
print("TFlite predicting took {}".format(elapsed))
