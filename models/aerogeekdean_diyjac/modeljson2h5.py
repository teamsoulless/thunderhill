import os
from pathlib import Path
import argparse
import json
import h5py
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import __version__ as keras_version
from keras.optimizers import Adam

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'modeljson',
        type=str,
        help='Path to new model json file.'
    )
    parser.add_argument(
        'modelh5',
        type=str,
        nargs='?',
        default='',
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()

    fileModelJSON = args.modeljson
    fileWeights = fileModelJSON.replace("json", "h5")

    if Path(fileModelJSON).is_file():
        with open(fileModelJSON, 'r') as jfile:
            model = model_from_json(json.load(jfile))
        # load weights into new model
        # centered
        # adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
        # left
        # adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
        # right
        adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
        model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
        model.load_weights(fileWeights)
        print("Loaded model from disk:", args.modeljson)
        model.summary()
        print("Saving model to disk:", args.modelh5)
        model.save(args.modelh5)
