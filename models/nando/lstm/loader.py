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
    return img

def getDataFromFolder(folder, output, normalize=False, randomize=True, balance=True, split=True):
    data = pd.DataFrame(columns=COLUMNS)
    print('<<<<<<<<<<<<<<<<<<<< ', folder, ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    for csvpath in glob.glob('{}/**/driving_log.csv'.format(folder)):
        print('-- ', csvpath)
        df = pd.read_csv(csvpath)
        df.columns = COLUMNS

        skip = False
        for toSkip in SKIP:
            if toSkip in csvpath:
                skip = True
        if skip:
            continue
        basename = os.path.dirname(csvpath)
        df['center'] = basename + '/' + df['center']
        df['positionX'], df['positionY'], df['positionZ'] = df['position'].str.split(':', 2).str
        df['rotationX'], df['rotationY'], df['rotationZ'] = df['rotation'].str.split(':', 2).str
        df[COLUMNS_TO_NORMALIZE] = df[COLUMNS_TO_NORMALIZE].astype(float)
        data = data.append(df)

    data = data.drop(['right', 'left'], 1)

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
        pickle.dump(scaler, open(os.path.join(output, SCALER), 'wb'))

    if split:
        return train_test_split(data, test_size=0.2, random_state=42)
    return data