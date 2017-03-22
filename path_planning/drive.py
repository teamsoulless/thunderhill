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

taup = 0.001
taud = 0.01
taui = 0.0
int_cte = 0.0

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def get_cte(keypoints, cp):
    prev=keypoints[0]
    tp = (keypoints[0], keypoints[1])
    tpidx = 1
    mind = 50000.
    for idx, kp in enumerate(keypoints[1:]):
        d = distance(prev, kp, cp)
        if mind>d:
            mind = d
            tp,tpidx = (prev, kp), idx
        prev = kp   
        
    return mind, tp, tpidx

def line_point_position(l1,l2, p):
    return ((l2[0] - l1[0]) * (p[1] - l1[1]) - (l2[1]-l1[1]) * (p[0]-l1[0]))

def get_nearest_point(keypoints, cp):
    mind = 1000
    min_idx = 0
    for idx, kp in enumerate(keypoints):
        d = line_dist(kp, cp)
        if mind > d:
            min_idx = idx
            mind = d

    return min_idx

prev_point = [0,0]
count = 0
target_speed = 15.0
prevcte = 0.0
@sio.on('telemetry')
def telemetry(sid, data):
    global tpidx, count, target_speed, myrobot, prevcte, int_cte, prev_point
    global taup, taud, taui
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = float(data["throttle"])
    # The current speed of the car
    speed = float(data["speed"])
    print("Speed: ", speed)
    # The current image from the center camera of the car
    imgString = data["image"]
    position = data["position"]
    px,py, _ = [float(x) for x in position.split(":")]
    cp = (int(px), int(py))

    rotation = data["rotation"]
    rx,ry,rz = [float(x) for x in rotation.split(":")]

    tpidx = (get_nearest_point(keypoints, (px,py)) + 25) % len(keypoints)

    cv2.circle(img,(int(px),int(py)), 2, (255,255,255))
    cv2.circle(img, keypoints[tpidx], 2, (255,255,0))
    cv2.line(img, cp, keypoints[tpidx], (255, 0, 0))
    cv2.imwrite("image.png", img)

    car_orientation = deg2rad(ry-175.0-84.92393566084966)

    target = keypoints[tpidx]

    dir = line_point_position(prev_point, cp, target)

    cte, _, _ = get_cte(keypoints, cp)
    dcte = cte - prevcte * taud

    steer = -cte * taup - dcte *taud - int_cte * taui

    if dir < 0.0:
            steer *= -1

    int_cte += cte
    print(prev_point, ",", cp, ",", target, "tpidx: ", tpidx, "car_orientation: ", rad2deg(car_orientation), "heading : ", rad2deg(steer))

    steering_angle = steer
    if count % 3 == 0:
        prev_point = cp

    if count % 50 == 0:
        count = 0
        int_cte = 0.0

    if speed < target_speed:
        throttle += 0.025
    elif speed > target_speed:
        throttle -= 0.025

    send_control(steering_angle, throttle)
    prevcte = cte
    count += 1



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


# (1242, 682) (1213, 669) tpidx:  188 dist:  40.63088121429323 car_orientation:  -77.06213566084968 heading :  -155.85445803957836 31.78049716414141 - CORRECT ANGLE
# (1243, 682) (1268, 655) tpidx:  188 dist:  40.30683719469939 car_orientation:  -78.75963566084967 heading :  -47.20259816176582 2.8284271247461903