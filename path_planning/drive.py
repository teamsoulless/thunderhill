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
import tensorflow as tf
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

for idx, p in enumerate (position_data):
    pos = (int(p[1][4]), int(p[1][5]))
    cv2.circle(img,pos, 1, (0,0,255))
        
    if idx % 25 == 0 and idx<1900:
        pos = (int(p[1][4]), int(p[1][5]))
        keypoints.append(pos)

tpidx = 2
DEG25 = deg2rad(25) 

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def can_move_next_tp(keypoints, tpidx, cp, change_at = 5):            
    car_d=line_dist(cp, keypoints[tpidx])
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
        if(mind>d):
            mind = d
            tp,tpidx = (prev, kp), idx
        prev = kp   
        
    return mind, tp, tpidx


@sio.on('telemetry')
def telemetry(sid, data):
    global tpidx
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    #print('speed:',data["speed"])
    position = data["position"]
    px,py, _ = [float(x) for x in position.split(":")]
    #print('position: ', data["position"])

    _,_,tpidx=get_cte(keypoints, (px,py))
    
    rotation = data["rotation"]
    rx,ry,rz = [float(x) for x in rotation.split(":")]

    #print('rotation: ', rx,ry,rz)
    if can_move_next_tp(keypoints, tpidx, (int(position[0]),int(position[1]))):
        tpidx = (tpidx + 1) % (len(keypoints)-1)
        print("Moving to next key point")


    cv2.circle(img,(int(px),int(py)), 5, (255,0,255))
    cv2.circle(img, keypoints[tpidx], 5, (255,0,0))
    cv2.imwrite("image.png", img)
    
    car_orientation = deg2rad(ry-175.0-82.0)    

    print((int(px),int(py)), keypoints[tpidx])
    steer = get_heading((int(px),int(py)), keypoints[tpidx])
    #print("car_orientation: ", rad2deg(car_orientation), "heading : ", rad2deg(steer))
    steer = float(steer) % (2.0 * pi)
    #print(steer)
    steer = car_orientation - steer  
    steer = angle_trunc(steer)
    if steer > DEG25:
        steer = DEG25
    if steer < -DEG25:
        steer = -DEG25

    #print("Steer: ", steer)
    steering_angle = steer / DEG25    
    #print(steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.25
    print("control: ", steering_angle, throttle, tpidx)
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
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
