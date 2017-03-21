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
import cv2
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

for idx, p in enumerate (position_data[:1860]):
    pos = (int(p[1][4]), int(p[1][5]))
    cv2.circle(img,pos, 1, (0,0,255))
        
    if idx % 1 == 0 and idx<1855:
        pos = (int(float(p[1][4])), int(float(p[1][5])))
        keypoints.append(pos)

tpidx = 2
DEG25 = deg2rad(25) 

taup = 0.25

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def can_move_next_tp(keypoints, tpidx, cp, change_at=10):
    car_d=line_dist(cp, keypoints[tpidx])
    print(cp, keypoints[tpidx], car_d)
    if car_d<change_at:
        return True
        
    return False

def get_cte(keypoints, cp):
    prev=keypoints[0]
    tp = (keypoints[0], keypoints[1])
    tpidx = 1
    mind = 1000.
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

throttle = 0.1

targets = [(1243.80, 704.22), (1245, 680) , (1268,655), (1213, 669)]
curidx = 0

prev_point = None
count = 0
target_speed = 15.0
@sio.on('telemetry')
def telemetry(sid, data):
    global tpidx, throttle, targets, curidx, prev_point, count, target_speed
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    car_throttle = float(data["throttle"])
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    #print('speed:',data["speed"])
    position = data["position"]
    px,py, _ = [float(x) for x in position.split(":")]
    cp = (int(px), int(py))
    #print('position: ', data["position"])

    #_,_,tpidx=get_cte(keypoints, (px,py))
    
    rotation = data["rotation"]
    rx,ry,rz = [float(x) for x in rotation.split(":")]

    tpidx = (get_nearest_point(keypoints, (px,py)) + 25) % len(keypoints)
    #ptdist = line_dist(keypoints[tpidx], (px,py))


    cv2.circle(img,(int(px),int(py)), 2, (255,255,255))
    cv2.circle(img, keypoints[tpidx], 2, (255,255,0))
    cv2.imwrite("image.png", img)

    car_orientation = deg2rad(ry-175.0-84.92393566084966)
    #print("ry ", ry)

    target = keypoints[tpidx]
    print((int(px),int(py)),", ", target)

    steer = 0.0
    if prev_point:
        dir = line_point_position(prev_point, cp, target)
        print("Direction: ", dir)

        if dir < 0.0:
            steer = 0.2
        else:
            steer = -0.2



    #steer = get_heading((int(px),int(py)), keypoints[tpidx])
    #steer = get_heading(cp, target)

    print(prev_point, ",", cp, ",", target, "tpidx: ", tpidx, "car_orientation: ", rad2deg(car_orientation), "heading : ", rad2deg(steer))

    #steer = (steer - car_orientation)
    #steer = steer * taup / DEG25

    """
    steer = angle_trunc(steer)
    if steer > DEG25:
        steer = DEG25
    if steer < -DEG25:
        steer = -DEG25
    """

    steering_angle = steer
    #print(steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.

    print("control: ", steering_angle, throttle, tpidx)
    #steering_angle = 0.32
    send_control(steering_angle, throttle)
    count += 1
    if count % 3 == 0:
        prev_point = cp

    if speed < target_speed:
        throttle += 0.025
    elif speed > target_speed:
        throttle -= 0.025



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