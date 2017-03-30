import argparse
import base64
from datetime import datetime
import os
import h5py
import shutil

import matplotlib.pyplot as plt
import cv2
import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
from keras import __version__ as keras_version

import utils
from visualize import VisualizeActivations
from decorators import staticVars


utils.setGlobals()

sio = socketio.Server()
app = Flask(__name__)
model = None
visualizer = None


@staticVars(control_que=np.zeros(8))
@sio.on('telemetry')
def telemetry(sid, data):
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
        image_array = np.asarray(image)

        processed = utils.process_image(image_array)

        steering_angle, throttle = model.predict(
            [processed[np.newaxis, ...], telemetry.control_que.reshape(1, -1)],
            batch_size=1
          )[0]

        telemetry.control_que = telemetry.control_que[1:]
        telemetry.control_que = np.append(telemetry.control_que, float(speed)/100)

        send_control(steering_angle, throttle)
        print('S: %2.5f | T: %2.5f' % (steering_angle, throttle))

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)

            if args.visualize_activations != '':
                processed = visualizer.heat_map(
                    layer_name=args.visualize_activations,
                    im=image_array.astype(np.uint8),
                    c_space='RGB'
                  )
                plt.imsave('{}.jpg'.format(image_filename), processed)
            else:
                plt.imsave('{}.jpg'.format(image_filename), image_array)
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
    parser.add_argument(
        'visualize_activations',
        type=str,
        nargs='?',
        default='',
        help='Visualizes the activations of the given layer on the images stored in the `image_folder`.'
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

    visualizer = VisualizeActivations(model, utils.process_image, utils.rectify_image)

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
