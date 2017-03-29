import argparse
import base64
import json
import cv2

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import h5py
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.models import load_model
from keras import __version__ as keras_version
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import keras.backend.tensorflow_backend as K

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

from geopy.distance import vincenty
from mpl_toolkits.basemap import Basemap

start = (39.53745, -122.33879)
mt = Basemap(llcrnrlon=-122.341041,llcrnrlat=39.532678,urcrnrlon=-122.337929,urcrnrlat=39.541455,
    projection='merc',lon_0=start[1],lat_0=start[0],resolution='h')

ch, img_rows, img_cols = 3, 160, 320

diffx = -989.70
diffy = -58.984

def toGPS(simx, simy):
    projx, projy = simx+diffx, simy + diffy
    lon, lat = mt(projx, projy,inverse=True)
    return [lon, lat]

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # print the data
    # print("data", data.keys())
    # The current steering angle of the car
    steering_angle = data["steering_angle"]

    # The current position of the car in the simulator
    position = [float(n) for n in data['position'].split(':')]

    # set the gps coordinates from the simulator position.
    print("GPS:", toGPS(position[0], position[1]))

    # The current throttle of the car
    throttle = data["throttle"]

    # The current speed of the car
    speed = data["speed"]

    # The current image from the center camera of the car
    imgString = data["image"]

    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    # verify sizing is correctly set.
    print("original size: ", image_array.shape)

    # resizing image here, you may or may not want to do this...
    image_array = cv2.resize(image_array, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
    print("resized size: ", image_array.shape)

    # set up the tensor for the model.
    transformed_image_array = image_array[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    #steering_angle, throttle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    #throttle = 0.8
    predictions = model.predict(transformed_image_array, batch_size=1)
    steering_angle = float(predictions[0][0])
    #steering_angle = float(predictions)
    throttle = float(predictions[0][1])

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

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

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

