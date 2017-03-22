import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from flask import Flask, render_template
from io import BytesIO

# Fix error with Keras and TensorFlow
import matplotlib.pyplot as plt

import cv2
from robot import *
from geo import *
from drawing import *
from dataprovider import *
from model import *

# tf.python.control_flow_ops = tf

filename='data/000/driving_log.csv'
position_data = load_simulator_data(filename)
print(len(position_data))

keypoints = []
img = np.ones((1500,1800, 3), np.uint8) 

for idx, p in enumerate (position_data[280:]):
    pos = (int(p[1][4]), int(p[1][5]))
    cv2.circle(img,pos, 1, (0,0,255))
        
    if idx % 1 == 0 and idx<1855:
        pos = (int(float(p[1][4])), int(float(p[1][5])))
        keypoints.append(pos)

tpidx = 2
DEG25 = deg2rad(25) 


sio = socketio.Server()
app = Flask(__name__)

pidmodel = PidModel()

sequence = 0

@sio.on('telemetry')
def telemetry(sid, data):
    global sequence, model
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = float(data["throttle"])
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    position = data["position"]

    rotation = data["rotation"]

    _steering_angle, _throttle = pidmodel.predict(sequence, steering_angle, throttle, speed, imgString, position, rotation)
    sequence += 1
    send_control(_steering_angle, _throttle)




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
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)