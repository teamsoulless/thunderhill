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

show_images=True
show_images=False


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


def generate_thunderhill_batches(gen, batch_size):
    batch_x = []
    batch_y = []
    while True:
        for img, steering_angle in gen:
            for img1, steering_angle1 in RandomFlip(img, steering_angle):
                img1, steering_angle1 = RandomBrightness(img1, steering_angle1)
                img1, steering_angle1 = RandomShift(img1, steering_angle1)
                img1 = Preproc(img1)
                if steering_angle1>1:
                    steering_angle1=1
                if steering_angle1<-1:
                    steering_angle1=-1

                font = cv2.FONT_HERSHEY_SIMPLEX
                if show_images:
                    imgp = Preproc(img.copy())
                    imgshow=(np.concatenate((imgp,img1),axis=1)+0.5)
                    cv2.putText(imgshow,'%.3f'%(steering_angle),(10,50), font, 1,(0,0,255),1,cv2.LINE_AA)
                    cv2.putText(imgshow,'%.3f'%(steering_angle1),(330,50), font, 1,(0,0,255),1,cv2.LINE_AA)
                    cv2.imshow('img',imgshow)
                    cv2.waitKey(0)
                batch_x.append(np.reshape(img1, (1, HEIGHT, WIDTH, DEPTH)))
                batch_y.append([steering_angle1])
                if len(batch_x) == batch_size:
                    batch_x, batch_y = shuffle(batch_x, batch_y)
                    yield np.vstack(batch_x), np.vstack(batch_y)
                    batch_x = []
                    batch_y = []


def generate_thunderhill_batches_time(gen, batch_size,time=TIME):
    batch_x = []
    batch_y = []
    time_x = []
    time_y = []
    while True:
        for img1, steering_angle1 in gen:
            img1 = Preproc(img1)
            if steering_angle1>1:
                steering_angle1=1
            if steering_angle1<-1:
                steering_angle1=-1

            font = cv2.FONT_HERSHEY_SIMPLEX
            time_x.append(np.reshape(img1, (HEIGHT, WIDTH, DEPTH)))

            if len(time_x) == time:
                if show_images:
                    print(steering_angle1)
                    for t in range(time):
                        imgshow=(time_x[t]+0.5)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(imgshow,'%d'%(t),(10,50), font, 1,(0,0,255),1,cv2.LINE_AA)
                        cv2.imshow('img',imgshow)
                        cv2.waitKey(100)
                batch_x.append(time_x)
                batch_y.append(steering_angle1)
                time_x = []
                time_y = []
                if len(batch_x) == batch_size:
                    batch_x, batch_y = shuffle(batch_x, batch_y)
                    batch_x=np.array(batch_x)
                    batch_y=np.array(batch_y)
                    yield batch_x, batch_y
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

sim320csvs=['dataset_sim_000_km_few_laps/driving_log.csv','dataset_sim_001_km_320x160/driving_log.csv','dataset_sim_002_km_320x160_recovery/driving_log.csv','dataset_sim_003_km_320x160/driving_log.csv']

polysynccsvs=['dataset_polysync_1464466368552019/output.txt','dataset_polysync_1464470620356308/output.txt','dataset_polysync_1464552951979919/output.txt']

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

# def getDataFromFolder(folder):
#     data5=getSession5(folder)
#     data000=getSim320(folder,sim320csvs[0])
#     data001=getSim320(folder,sim320csvs[1])
#     data002=getSim320(folder,sim320csvs[2])
#     print('Dataset Session5 size:',len(data5))
#     print('Dataset 000 size:',len(data000))
#     print('Dataset 001 size:',len(data001))
#     print('Dataset 002 size:',len(data002))
#     # data=data5 + data001 + data002
#     # data=data5[::5]
#     # df = data)
#     # return train_test_split(df, test_size=0.2, random_state=42)
#     return shuffle(data001),data002)
#     # return data5[::10]),data001)

def genSession5(folder):
    data=getSession5(folder)
    while True:
        for row in data:
            img = cv2.imread(row[0])[:200,200:-200,:]
            steering_angle = row[1]
            yield img,steering_angle/2.0

def genSim001(folder):
    data=getSim320(folder,sim320csvs[1])
    while True:
        for row in data:
            img = cv2.imread(row[0])[20:140,:,:]
            steering_angle = row[1]
            yield img,steering_angle

def genSim002(folder):
    data=getSim320(folder,sim320csvs[2])
    while True:
        for row in data:
            img = cv2.imread(row[0])[20:140,:,:]
            steering_angle = row[1]
            yield img,steering_angle

def genSim003(folder):
    data=getSim320(folder,sim320csvs[2])
    while True:
        for row in data:
            img = cv2.imread(row[0])[20:140,:,:]
            steering_angle = row[1]
            yield img,steering_angle

def getPolysync(repo, rel_csv):
    csvfname=repo+rel_csv
    data=[]
    with open(csvfname, 'r') as csvfile:
        data_tmp = list(csv.reader(csvfile, delimiter=','))[1:]
        dd='/'.join(csvfname.split('/')[:-1])
        for row in data_tmp:
            data.append((dd+'/'+row[0],float(row[-3]),float(row[-2]),float(row[-1]),float(row[-6])))
    return data

def genPolysync0(folder):
    data=getPolysync(folder,polysynccsvs[0])[500:-500]
    while True:
        for row in data:
            img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
            steering_angle = row[1]
            yield img,steering_angle/5.0

def genPolysync2(folder):
    data=getPolysync(folder,polysynccsvs[2])
    data=data[500:2000] + data[3100:-500]
    while True:
        for row in data:
            img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
            steering_angle = row[1]
            yield img,steering_angle/5.0

def genAll(folder):
    datasets=[]
    datasets.append(getPolysync(folder,polysynccsvs[0])[500:-500])
    data=getPolysync(folder,polysynccsvs[2])
    data=data[500:2000] + data[3100:-500]
    datasets.append(data)
    datasets.append(getSim320(folder,sim320csvs[2]))
    datasets.append(getSim320(folder,sim320csvs[3]))
    datasets.append(getSession5(folder))

    while True:
        nr=0
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
        steering_angle = row[1]
        yield img, steering_angle/5.0
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
        steering_angle = row[1]
        yield img, steering_angle/5.0
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[:200,200:-200,:]
        steering_angle = row[1]
        yield img,steering_angle/2.0
