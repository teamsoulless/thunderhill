import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
from mainMultiple import preprocessImage, customLoss, toGPS
from matplotlib import image as mpimg
import cv2
import functools
import time

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

img = mpimg.imread('./simulator/data/data/IMG/center_2016_12_01_13_30_48_287.jpg')
h, w = img.shape[:2]

def staticVar(**kwargs):
    """This function allows to define C-Like
    static variables for function in python using"""
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

#static Var to store the previous steering angles
@staticVar(angles = list(np.zeros(5)))
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        splitPos = data["position"].split(":")
        splitOri = data["rotation"].split(":")
        xVec = toGPS(float(splitPos[0]),float(splitPos[1]))
        xVec = list(xVec)
        xVec.append(float(data["speed"]))
        xVec = np.array(xVec)
        print(xVec)
        
        for i, mean, std in zip([0,1,2],
                            [-122.33790211, 39.53881540, 62.68238949],
                             [0.00099555, 0.00180817, 13.48539298]):
            xVec[i] -= mean
            xVec[i] /= std
        print(xVec)
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image).astype(np.uint8)
        t = time.time()
        steering_angle, throttle = model.predict([preprocessImage(image_array)[None,:,:,:], 
                                                            xVec[None,:]])#,np.array(telemetry.angles)[None,:]])
        print(time.time() - t, steering_angle, throttle)
        steering_angle = float(steering_angle)
        throttle = float(throttle)
        if float(speed) > 50:
            throttle -=.3
        telemetry.angles.pop(0)
        telemetry.angles.append(steering_angle)
        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
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
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model, custom_objects={'customLoss':customLoss})

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)