from MainNode.MainNode import MainNode
from ctypes import *
import array
from PIL import Image
from io import BytesIO
import numpy as np
from keras.models import load_model
import h5py
from keras import __version__ as keras_version
import tensorflow as tf
from keras import backend as K
import cv2
import time
from data_buffer import DataBuffer
import queue
import threading
import argparse
import base64
import json

import numpy as np
#import socketio
#import eventlet
#import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
#from flask import Flask, render_template

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
#import tensorflow as tf
#import matplotlib.pyplot as plt



def customLoss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)



f = h5py.File("thunderhill.h5", mode='r')
model_version = f.attrs.get('keras_version')
keras_version = str(keras_version).encode('utf8')

if model_version != keras_version:
	print('You are using Keras version ', keras_version, ', but the model was built using ', model_version)

model = load_model("thunderhill.h5")
#
# with open('model.json', 'r') as jfile:
#     model = model_from_json(jfile.read())
#
# model.compile("adam", "mse")
# weights_file = 'weights.0011-16.389.hdf5'
# model.load_weights(weights_file)

# model = load_model("multiModel.h5", custom_objects={'customLoss':customLoss})
graph = tf.get_default_graph()

data_buffer = DataBuffer()
res_queue = queue.Queue(maxsize=1)
#
# idxs = [0, 1, 2]
# means = [-122.33790211, 39.53881540, 62.68238949]
# stds = [0.00099555, 0.00180817, 13.48539298]


# def normalize_vector(xVec):
#     for i, mean, std in zip(idxs, means, stds):
#         xVec[i] -= mean
#         xVec[i] /= std
#     return xVec


def copyImage(byte_array, imageSize):
    if imageSize > 8:
        resize(byte_array, imageSize)
        image = []
        for i in range(imageSize):
            image.append(byte_array[i])
        return array.array('B', image).tostring()
    return byte_array


def imageReceived(imageSize, rawImage, speed, lat, lon):
    # print("in image received")
    # print(speed, lat, lon)
    jpegImage = copyImage(rawImage, imageSize)
    data_buffer.add_item((jpegImage, speed, lat, lon))
    try:
        prediction = res_queue.get(block=False)
        Node.steerCommand(c_float(prediction[0]))
        Node.throttleCommand(c_float(prediction[1]))
        Node.brakeCommand(c_float(prediction[2]))
    except queue.Empty:
        pass

def make_prediction():
    global graph
    # print('make prediction')
    while True:
        with graph.as_default():
            item = data_buffer.get_item_for_processing()
            if item and len(item) == 4:
                import time
                start = time.time()
                jpeg_image = item[0]
                speed = item[1]
                lat = item[2]
                lon = item[3]
                # xVec = np.array([lon, lat, speed])
                # norm_xVec = normalize_vector(xVec)
                # print(jpeg_image)
                if jpeg_image:
                    img = np.array(Image.frombytes('RGB', [960,480], jpeg_image, 'raw'))
                    image_array = cv2.resize(img, (320, 160))[::-1,:,:]
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

                    output = model.predict(image_array[None, :, :, :], batch_size=1)
                    steering_angle = output[0][0]*(-1)
                    throttle = output[0][1]

                    min_speed = 8 * 0.44704
                    max_speed = 50 * 0.44704

                    if float(speed) > 25 * 0.44704:
                        throttle = 0

                    if steering_angle > 0.2 or steering_angle < -0.2:
                        throttle = -throttle


                    brake=0
                    if throttle<0:
                        throttle=0
                        brake=np.abs(throttle)

                    print('prediction:',steering_angle,throttle,brake, 'speed: ',speed/0.44704)

                    if res_queue.full(): # maintain a single most recent prediction in the queue
                        res_queue.get(False)

                    res_queue.put((steering_angle, throttle, brake))
                end = time.time()
                print('\n%.3f ms\n'%(end - start))


thread = threading.Thread(target=make_prediction, args=())
thread.daemon = True
thread.start()


Node = MainNode(imageReceived)
Node.connectPolySync()
