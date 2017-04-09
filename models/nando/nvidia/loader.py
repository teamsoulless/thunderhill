import glob
import pickle

import numpy as np
import pandas as pd
import cv2



# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformations import Preproc, RandomShift, RandomFlip, RandomBrightness, RandomRotation, RandomBlur
from config import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def ReadImg(path):
    img = np.array(cv2.cvtColor(cv2.imread(path.strip()), code=cv2.COLOR_BGR2RGB))
    if '320x160' in path:
        img = img[20:140, :, :]
    if 'thunderhill' in path:
        img = cv2.resize(img, dsize=(WIDTH, HEIGHT))
    return img

MAX = 33.706074

def generate_thunderhill_batches(df, args):

    while True:
        batch_x = []
        batch_y = []
        batch_speed = []

        for idx, row in df.iterrows():
            steering_angle = row['steering']
            speed = row['speed']
            brake = row['brake']
            throttle = row['throttle']
            img = ReadImg(row['center'])
            img, steering_angle = RandomShift(img, steering_angle, args.adjustement)
            img, steering_angle = RandomFlip(img, steering_angle)
            img, steering_angle = RandomBrightness(img, steering_angle)
            img, steering_angle = RandomRotation(img, steering_angle)
            img, steering_angle = RandomBlur(img, steering_angle)
            # Preproc is after ....
            img = Preproc(img)
            batch_x.append(np.reshape(img, (1, HEIGHT, WIDTH, DEPTH)))
            batch_speed.append(speed/MAX)
            batch_y.append([
                steering_angle,
                throttle,
                brake
            ])
            if len(batch_x) == args.batch:
                batch_x, batch_y = shuffle(batch_x, batch_y)
                yield [np.vstack(batch_x), np.vstack(batch_speed)], np.vstack(batch_y)
                batch_x = []
                batch_y = []
                batch_speed = []


def getDataFromThunderhill(folder, normalize=False, randomize=True, balance=True, split=True):
    data = pd.DataFrame(columns=COLUMNS)
    for csvpath in glob.glob('{}/**/**/output_processed.txt'.format(folder)):
        df = pd.read_csv(csvpath)
        df = df.dropna(axis=0)
        basename = os.path.dirname(csvpath)
        df['center'] = basename + '/' + df['path']
        data = data.append(df, ignore_index=True)

    if balance:
        hist, counts = np.histogram(data.steering, bins=50)
        upper_limit = 2000
        over = [(i, v) for i, v in enumerate(hist) if v > upper_limit ]
        over_ranges = [(counts[i], counts[i+1]) for i,_ in over]

        #loop through ranges and create a mask for each bin
        masks = ["data[(data.steering >= {0}) & (data.steering < {1})]".format(l, r) for l, r in over_ranges]

        for mask in masks:
            selected = eval(mask)
            selected_length = len(selected)
            frac_to_drop = abs((selected_length-upper_limit)/selected_length)
            samples_to_drop = selected.sample(frac=frac_to_drop)
            data = data.drop(samples_to_drop.index)

    if randomize:
        data = shuffle(data)

    if normalize:
        scaler = StandardScaler()
        data[COLUMNS_TO_NORMALIZE] = scaler.fit_transform(data[COLUMNS_TO_NORMALIZE])
    if split:
        return train_test_split(data, test_size=0.2, random_state=42)
    return data