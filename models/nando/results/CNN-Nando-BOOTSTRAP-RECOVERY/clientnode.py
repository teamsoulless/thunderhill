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
import ctypes


f = h5py.File("model.h5", mode='r')
model_version = f.attrs.get('keras_version')
keras_version = str(keras_version).encode('utf8')

if model_version != keras_version:
	print('You are using Keras version ', keras_version, ', but the model was built using ', model_version)


model = load_model("model.h5")
graph = tf.get_default_graph()

data_buffer = DataBuffer()
res_queue = queue.Queue(maxsize=1)

MAX = 33.706074


def copyImage(byte_array, imageSize):
    new_array = np.ctypeslib.as_array(byte_array,shape=(imageSize,)).reshape((960, 480, 3))
    return new_array
	


def imageReceived(imageSize, rawImage, speed, lat, lon):
	jpegImage = copyImage(rawImage, imageSize)
	data_buffer.add_item((jpegImage, speed, lat, lon))
	
Node = MainNode(imageReceived)

def make_prediction():
	global graph
	while True:
		with graph.as_default():
			item = data_buffer.get_item_for_processing()
			if item and len(item) == 4:
				jpeg_image = item[0]
				speed = item[1]
				image = np.array(Image.frombytes('RGB', [960,480], jpeg_image, 'raw'))
				image_array = np.asarray(image)
				image_array = cv2.resize(image_array, (160, 80))
				prediction = model.predict([image_array[None, :, :, :],np.vstack([float(speed)/MAX])], batch_size=1)[0]
				steering_angle = float(prediction[0])
				throttle = float(prediction[1])
				brake = float(prediction[2])
				############################# Steering angles ##########################################################
				if brake > 0.5:
					throttle = 0.
				else:
					brake = 0.
				if speed > 23.0:
					throttle = 0.
				print("puts in the queue: ", steering_angle, throttle, brake)
				res_queue.put((steering_angle, throttle, brake))


def sendValues():
	steer = 0
	throttle = 0
	brake = 0
	while 1:
		try:
			prediction = res_queue.get(block=False)
			steer = c_float(prediction[0])
			throttle = c_float(prediction[1])
			brake = c_float(prediction[2])
		except queue.Empty:
			pass
		Node.steerCommand(steer)
		Node.throttleCommand(throttle)
		Node.brakeCommand(brake)
		time.sleep(0.01)

thread = threading.Thread(target=make_prediction, args=())
thread.daemon = True
thread.start()
thread2 = threading.Thread(target=sendValues, args=())
thread2.daemon = True
thread2.start()



Node.connectPolySync()