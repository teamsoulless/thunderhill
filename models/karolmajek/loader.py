import numpy as np
import pandas as pd
import cv2
import glob
import os
import csv

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformations import Preproc, RandomShift, RandomFlip, RandomBrightness
from config import *

def ReadImg(path):
    return np.array(cv2.cvtColor(cv2.imread(path.strip()), code=cv2.COLOR_BGR2RGB))

def generate_validation(df, basename='data'):
    batch_x, batch_y = [], []
    # for idx, row in df.iterrows():
    for idx, row in enumerate(df):
        basename = '{}/{}'.format(basename, row['center'].strip())
        steering_angle = row['steering']
        img = ReadImg(basename)
        img = Preproc(img)
        batch_x.append(np.reshape(img, (1, HEIGHT, WIDTH, DEPTH)))
        batch_y.append([steering_angle])
    return batch_x, batch_y


def generate_thunderhill_batches(df, batch_size):
    batch_x = []
    batch_y = []
    while True:
        for idx, row in enumerate(df):
            steering_angle = row[1]
            img = cv2.imread(row[0])
            img, steering_angle = RandomShift(img, steering_angle)
            img, steering_angle = RandomFlip(img, steering_angle)
            img, steering_angle = RandomBrightness(img, steering_angle)
            img = Preproc(img)
            batch_x.append(np.reshape(img, (1, HEIGHT, WIDTH, DEPTH)))
            batch_y.append([steering_angle])
            if len(batch_x) == batch_size:
                batch_x, batch_y = shuffle(batch_x, batch_y)
                yield np.vstack(batch_x), np.vstack(batch_y)
                batch_x = []
                batch_y = []

def getSession5(repo):
    csvfname=repo+'/dataset_session_5/output.csv'
    data=None
    with open(csvfname, 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        dd='/'.join(csvfname.split('/')[:-1])
        data = [(dd+'/'+x[0],float(x[3]),float(x[4]),float(x[5]),float(x[6])) for x in data]
    return data

sim320csvs=['dataset_sim_000_km_few_laps/driving_log.csv','dataset_sim_001_km_320x160/driving_log.csv','dataset_sim_002_km_320x160_recovery/driving_log.csv']

def getSim320(repo,rel_csv):
    csvfname=repo+rel_csv
    data=[]
    with open(csvfname, 'r') as csvfile:
        data_tmp = list(csv.reader(csvfile, delimiter=','))
        for row in data_tmp:
            x7=[float(x) for x in row[7].split(':')]
            x8=[float(x) for x in row[8].split(':')]
            dd='/'.join(csvfname.split('/')[:-1])
            data.append((dd+'/'+row[0],float(row[3]),float(row[4]),float(row[5]),float(row[6])))
    return data

def getDataFromFolder(folder):
    data5=getSession5(folder)
    data000=getSim320(folder,sim320csvs[0])
    data001=getSim320(folder,sim320csvs[1])
    data002=getSim320(folder,sim320csvs[2])
    print('Dataset Session5 size:',len(data5))
    print('Dataset 000 size:',len(data000))
    print('Dataset 001 size:',len(data001))
    print('Dataset 002 size:',len(data002))
    data=data5 + data001 + data002
    df = shuffle(data)
    return train_test_split(df, test_size=0.2, random_state=42)
