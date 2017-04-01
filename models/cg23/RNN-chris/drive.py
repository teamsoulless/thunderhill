import argparse
import base64
import json
import os

#import tempfile
dequefrom collections import
#import cv2
#from math import pi
#import time

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
#import matplotlib as mpl
#import matplotlib.image as mpimg
from scipy.misc import imread, imresize, imsave
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import tensorflow as tf
#tf.python.control_flow_ops = tf

#from keras.models import model_from_json
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')

def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    #image = mpimg.imread(BytesIO(base64.b64decode(imgString)))
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    
    image = np.asarray(image)
    transformed_image_array = image[:, :, :]
    transformed_image_array = (transformed_image_array.astype(np.float32))
    steering_angle, throttle = model.predict(transformed_image_array)

    if throttle < 0:
       throttle = 0
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

class KomandaModel(object):
    """Steering angle prediction model for Udacity challenge 2.
    """
    def __init__(self, checkpoint_dir, metagraph_file):
        self.graph =tf.Graph()
        self.LEFT_CONTEXT = 5 # TODO remove hardcode; store it in the graph
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(metagraph_file)
            ckpt = tf.train.latest_checkpoint('./')
        self.session = tf.Session(graph=self.graph)
        saver.restore(self.session, ckpt)
        #self.ops = self.session.graph.get_operations()
        #for op in self.ops:
        #    print(op.values())
        self.input_images = deque() # will be of size self.LEFT_CONTEXT + 1
        self.internal_state = [] # will hold controller_{final -> initial}_state_{0,1,2}

        # TODO controller state names should be stored in the graph
        self.input_tensors = list(map(self.graph.get_tensor_by_name, ["input_images:0", "controller_initial_state_0:0", "controller_initial_state_1:0", "controller_initial_state_2:0"]))
        self.output_tensors = list(map(self.graph.get_tensor_by_name, ["output_steering:0", "output_throttle:0","controller_final_state_0:0", "controller_final_state_1:0", "controller_final_state_2:0"]))

    def predict(self, img):
        if len(self.input_images) == 0:
            self.input_images += [img] * (self.LEFT_CONTEXT + 1)
        else:
            self.input_images.popleft()
            self.input_images.append(img)
        input_images_tensor = np.stack(self.input_images)
        if not self.internal_state:
            feed_dict = {self.input_tensors[0] : input_images_tensor}
        else:
            feed_dict = dict(zip(self.input_tensors, [input_images_tensor] + self.internal_state))
        steering, throttle, c0, c1, c2 = self.session.run(self.output_tensors, feed_dict=feed_dict)
        #self.internal_state = [c0, c1[0,:,:], c2[0,:,:]]
        self.internal_state = [c0, c1, c2]
        return steering[0][0], throttle[0][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('--metagraph_file', type=str, help='Path to the metagraph file')
    parser.add_argument('--checkpoint_dir', type=str, help='Path to the checkpoint dir')
    args = parser.parse_args()

    def make_predictor():
        model = KomandaModel(
            checkpoint_dir=args.checkpoint_dir,
            metagraph_file=args.metagraph_file)
        return model #lambda img: model.predict(img)

    def process(predictor, img):
        steering = predictor(img)
        print(steering)
        return steering
    
    model = make_predictor()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
