# ----------------------------------------------------------------------------------
# Self Racing Cars - Stateful RNN for Steering, Throttle and Brake - drive.py
# ----------------------------------------------------------------------------------
# By: Chris Gundling, chrisgundling@gmail.com

import argparse
import base64
import json
import os
from collections import deque
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from scipy.misc import imread, imresize, imsave
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import tensorflow as tf
tf.python.control_flow_ops = tf

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
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    
    image = np.asarray(image)
    transformed_image_array = image[:, :, :]
    transformed_image_array = (transformed_image_array.astype(np.float32))
    steering_angle, throttle, brake = model.predict(transformed_image_array)

    # If brake, set throttle to negative of brake
    if throttle < 0:
       throttle = 0
    if brake > 0:
        throttle = -brake

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


class SRCModel(object):
    """Steering/throttle/brake prediction model for Self Racing Cars.
    """
    def __init__(self, checkpoint_dir, metagraph_file):
        self.graph =tf.Graph()
        self.LEFT_CONTEXT = 5 
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(metagraph_file)
            ckpt = tf.train.latest_checkpoint('./')
        self.session = tf.Session(graph=self.graph)
        saver.restore(self.session, ckpt)
        self.input_images = deque() # of size self.LEFT_CONTEXT + 1
        self.internal_state = [] # holds controller_{final -> initial}_state_{0,1,2}

        # Controller state names stored in the graph
        self.input_tensors = map(self.graph.get_tensor_by_name, ["input_images:0", "controller_initial_state_0:0", "controller_initial_state_1:0", "controller_initial_state_2:0"])
        self.output_tensors = map(self.graph.get_tensor_by_name, ["output_steering:0", "output_throttle:0", "output_brake:0", "controller_final_state_0:0", "controller_final_state_1:0", "controller_final_state_2:0"])

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
        steering, throttle, brake, c0, c1, c2 = self.session.run(self.output_tensors, feed_dict=feed_dict)
        self.internal_state = [c0, c1, c2]
        return steering[0][0], throttle[0][0], brake[0][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('--metagraph_file', type=str, help='Path to the metagraph file')
    parser.add_argument('--checkpoint_dir', type=str, help='Path to the checkpoint dir')
    args = parser.parse_args()

    def make_predictor():
        model = SRCModel(
            checkpoint_dir=args.checkpoint_dir,
            metagraph_file=args.metagraph_file)
        return model

    model = make_predictor()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
