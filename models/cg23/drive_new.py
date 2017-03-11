# ----------------------------------------------------------------------------------
# SRC Simulator Driving - drive_new.py
# ----------------------------------------------------------------------------------
'''
This script when run will drive the vehicle autonomously in the simulator
using the trained model.
'''

import argparse
import base64
import json

import numpy as np
import cv2
import socketio
import eventlet
import eventlet.wsgi
import time
import matplotlib as mpl
import matplotlib.image as mpimg
from scipy.misc import imread, imresize, imsave
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

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
set_speed = 50
controller.set_desired(set_speed)


# Top down view transform
def birds_eye(img):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    # Four Source Points
    src = np.float32(
        [[100, 75],
         [0, 120],
         [200, 75],
         [320, 120]])

    # Four Destination Points
    dst = np.float32(
        [[80, 0],
         [100, 160],
         [215, 0],
         [215, 160]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


@sio.on('telemetry')

def telemetry(sid, data):
    global count
    if data:
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

        # Mods
        image = birds_eye(image)
        transformed_image_array = image[None, :, :, :]
        transformed_image_array = (transformed_image_array.astype(np.float32) - 128.) / 128.

        # This model currently assumes that the features of the model are just the images. Feel free to change this.
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))
        throttle = controller.update(float(speed))
        
        # Limit the vehicle speed
        if float(speed) > 50:
            throttle = 0
            
        # Tap the brakes if steering is large (big turns)
        if steering_angle > 0.65 or steering_angle < -0.65:
            count += 1
            if count < 3:
                throttle = -throttle
                count = 0
      
        print(steering_angle, throttle, float(speed))
        send_control(steering_angle, throttle)
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--resized-image-width', type=int, help='image resizing')
    parser.add_argument('--resized-image-height', type=int, help='image resizing')
    args = parser.parse_args()

    image_size = (args.resized_image_width, args.resized_image_height)

    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    count = 0

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)