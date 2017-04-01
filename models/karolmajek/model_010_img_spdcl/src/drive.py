import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt



from transformations import *


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

# PI controller for throttle/brake
class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

controller = SimplePIController(0.1, 0.01)
set_speed = 70
# set_speed = 50
# set_speed = 35
controller.set_desired(set_speed)
count=0
@sio.on('telemetry')
def telemetry(sid, data):
    global controller, count
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR)
    image_array = Preproc(image_array)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # print(image_array.shape)
    speed_cl=np.array(speedToClass(speed))

    image_array=image_array.reshape(1,image_array.shape[0],image_array.shape[1],image_array.shape[2])
    speed_cl=speed_cl.reshape(1,speed_cl.shape[0])

    res=model.predict([image_array,speed_cl], batch_size=1)
    steering_angle,throttle,brake = res[0]
    print(res)
    if brake>0.7:
        throttle=-1

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str, help='Path to model definition json.')
    parser.add_argument('weights', type=str, help='Path to model weights.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.weights
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
