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
        # img = img[-40:10:-1, :, :]
    return img

def generate_thunderhill_batches(df, args):

    while True:
        batch_x = []
        batch_y = []

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
            batch_y.append([steering_angle, throttle, brake])
            if len(batch_x) == args.batch:
                batch_x, batch_y = shuffle(batch_x, batch_y)
                yield np.vstack(batch_x), np.vstack(batch_y)
                batch_x = []
                batch_y = []

"""
Randomly split the dataset
"""
def __train_test_split(csvpath, balance=True):
    df = pd.read_csv(csvpath)
    if balance:
        zeros = df.where(df['steering'] == 0).dropna(axis=0).sample(1500)
        non_zeros = df.where(df['steering'] != 0).dropna(axis=0)
        df = zeros.append(non_zeros)
    df = shuffle(df)
    return train_test_split(df, test_size=0.2, random_state=42)


def getDataFromFolder(folder, output, normalize=False, randomize=True, balance=True, split=True):
    data = pd.DataFrame(columns=COLUMNS)
    for csvpath in glob.glob('{}/**/output.txt'.format(folder)):
        df = pd.read_csv(csvpath)
        skip = False
        for toSkip in SKIP:
            if toSkip in csvpath:
                skip = True
        if skip:
            continue
        basename = os.path.dirname(csvpath)
        df['center'] = basename + '/' + df['path']
        data = data.append(df, ignore_index=True)

    if balance:
        # data = data.where(data['steering'] != 0).dropna(axis=0)
        zeros = data.where(data['steering'] == 0).dropna(axis=0).sample(1000)
        non_zeros = data.where(data['steering'] != 0).dropna(axis=0)
        data = zeros.append(non_zeros)

    if randomize:
        data = shuffle(data)

    if normalize:
        scaler = StandardScaler()
        data[COLUMNS_TO_NORMALIZE] = scaler.fit_transform(data[COLUMNS_TO_NORMALIZE])
    if split:
        return train_test_split(data, test_size=0.2, random_state=42)
    return data


def getDataFromThunderhill(folder, output, normalize=False, randomize=True, balance=False, split=True):
    data = pd.DataFrame(columns=COLUMNS)
    for csvpath in glob.glob('{}/**/output_accel.txt'.format(folder)):
        df = pd.read_csv(csvpath)
        df = df.dropna(axis=0)
        skip = False
        for toSkip in SKIP:
            if toSkip in csvpath:
                skip = True
        if skip:
            continue
        basename = os.path.dirname(csvpath)
        df['center'] = basename + '/' + df['path']
        df['speed'] = np.power(np.power(df['vel0'], 2) + np.power(df['vel0'], 2) + np.power(df['vel0'], 2), 0.5)
        mask = df['accel']>-0.2
        df.loc[mask, 'throttle'] = df[mask]['accel'] + 0.2
        data = data.append(df, ignore_index=True)

    if balance:
        # data = data.where(data['steering'] != 0).dropna(axis=0)
        zeros = data.where(data['steering'] == 0).dropna(axis=0).sample(1000)
        non_zeros = data.where(data['steering'] != 0).dropna(axis=0)
        data = zeros.append(non_zeros)

    if randomize:
        data = shuffle(data)

    if normalize:
        scaler = StandardScaler()
        data[COLUMNS_TO_NORMALIZE] = scaler.fit_transform(data[COLUMNS_TO_NORMALIZE])
    if split:
        return train_test_split(data, test_size=0.2, random_state=42)
    return data