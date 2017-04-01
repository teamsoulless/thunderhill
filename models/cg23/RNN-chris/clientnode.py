from MainNode.MainNode import MainNode
from ctypes import *
import array
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import cv2
import time
from data_buffer import DataBuffer
import queue
import threading
from komanda_model import KomandaModel

model = KomandaModel(checkpoint_dir="checkpoint", metagraph_file="checkpoint-sdc-ch2.meta")
graph = tf.get_default_graph()

data_buffer = DataBuffer()
res_queue = queue.Queue(maxsize=1)


debug = False


def copyImage(byte_array, imageSize):
	if imageSize > 8:
		resize(byte_array, imageSize)
		image = []
		for i in range(imageSize):
			image.append(byte_array[i])
		return array.array('B', image).tostring()
	return byte_array


def imageReceived(imageSize, rawImage, speed, lat, lon):
	jpegImage = copyImage(rawImage, imageSize)
	data_buffer.add_item((jpegImage, speed, lat, lon))
	
Node = MainNode(imageReceived)

def make_prediction():
	global graph
	print('make prediction')
	while True:
		with graph.as_default():
			item = data_buffer.get_item_for_processing()
			if item and len(item) == 4:
				jpeg_image = item[0]
				if jpeg_image:
					if debug:
						image = Image.open(BytesIO(jpeg_image))
						image_array = np.asarray(image)
						img = cv2.resize(image_array, (320, 160))[::-1,::-1]
						print("this is debug mode, switch it to False")
					else:
						img = np.array(Image.frombytes('RGB', [960, 480], jpeg_image, 'raw'))
						img = cv2.resize(img, (320, 160))[::-1,::-1]
						transformed_image_array = img[:, :, :]
						transformed_image_array = (transformed_image_array.astype(np.float32))
					steering_angle, throttle, brake = model.predict(transformed_image_array)
					if res_queue.full(): # maintain a single most recent prediction in the queue
						res_queue.get(False)
					# the assumption is that throttle is always present unless brake is very high
					if brake > 0.6:
						throttle = 0
					else:
						brake = 0
					print("putting in the queue: ", steering_angle, throttle, brake)
					res_queue.put((steering_angle, throttle, brake))


def sendValues():
	steer = 0
	throttle = 0
	brake = 0
	while 1:
		try:
			prediction = res_queue.get(block=False)
			steer = c_float(4.*prediction[0])
			throttle = c_float(prediction[1])
			brake = c_float(prediction[2])
			print("got values: ", steer, throttle, brake)
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
