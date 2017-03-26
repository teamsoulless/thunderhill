import numpy as np
import pandas as pd
import cv2
import glob
import os
import csv

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformations import Preproc, RandomShift, RandomFlip, RandomBrightness, speedToClass, RandomNoise
from config import *

show_images=True
show_images=False

def generate_thunderhill_batches(gen, batch_size):
    batch_x = []
    batch_x2 = []
    batch_y = []
    while True:
        for img, steering_angle, throttle, brake, speed in gen:
            for img1, steering_angle1 in RandomFlip(img, steering_angle):
                img1, steering_angle1 = RandomBrightness(img1, steering_angle1)
                img1, steering_angle1, throttle1 = RandomShift(img1, steering_angle1,throttle)
                img1 = Preproc(img1)
                img1=RandomNoise(img1)
                if steering_angle1>1:
                    steering_angle1=1
                if steering_angle1<-1:
                    steering_angle1=-1

                brake1=brake

                if steering_angle1>0.5:
                    brake1=1
                    throttle1=0
                if steering_angle1<-0.5:
                    brake1=1
                    throttle1=0

                if np.random.uniform() < 0.5:
                    factor=float(np.random.uniform())
                    speed1=speed*factor
                else:
                    speed1=speed

                if speed1<50:
                    throttle1=1
                if speed1<60 and steering_angle1==0:
                    throttle1=1

                speed_cl=np.array(speedToClass(speed1))

                # if speed_cl[-2]==1 and steering_angle1!=0:
                #     throttle1=0
                if speed_cl[-1]==1 and steering_angle1!=0:
                    throttle1=0


                font = cv2.FONT_HERSHEY_SIMPLEX
                if show_images:
                    imgp = Preproc(img.copy())
                    imgshow=(np.concatenate((imgp,img1),axis=0)+0.5)*0.5
                    cv2.putText(imgshow,'%s'%(str(list(speed_cl[:8]))),(10,100), font, 0.4,(0,0,255),1,cv2.LINE_AA)
                    cv2.putText(imgshow,'%s'%(str(list(speed_cl[8:]))),(10,120), font, 0.4,(0,0,255),1,cv2.LINE_AA)

                    cv2.putText(imgshow,'%.3f %.1f %.1f %.1f'%(steering_angle,throttle, brake,speed),(10,70), font, 0.4,(0,0,255),1,cv2.LINE_AA)
                    cv2.putText(imgshow,'%.3f %.1f %.1f %.1f'%(steering_angle1,throttle1, brake1,speed1),(10,150), font, 0.4,(0,0,255),1,cv2.LINE_AA)
                    cv2.imshow('img',imgshow)
                    cv2.waitKey(0)
                batch_x.append(np.reshape(img1, (1, HEIGHT, WIDTH, DEPTH)))
                batch_x2.append(speed_cl)
                batch_y.append([steering_angle1,throttle1, brake1])
                if len(batch_x) == batch_size:
                    batch_x, batch_x2, batch_y = shuffle(batch_x, batch_x2, batch_y)
                    yield [np.vstack(batch_x), np.vstack(batch_x2)], np.vstack(batch_y)
                    batch_x = []
                    batch_x2 = []
                    batch_y = []
def getSession5(repo):
    csvfname=repo+'/dataset_session_5/output.csv'
    data=None
    with open(csvfname, 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        dd='/'.join(csvfname.split('/')[:-1])
        data = [(dd+'/'+x[0],float(x[3]),float(x[4]),float(x[5]),float(x[6])) for x in data]
    return data

sim320csvs=['dataset_sim_000_km_few_laps/driving_log.csv','dataset_sim_001_km_320x160/driving_log.csv','dataset_sim_002_km_320x160_recovery/driving_log.csv','dataset_sim_003_km_320x160/driving_log.csv','dataset_sim_004_km_320x160_cones_brakes/driving_log.csv']

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

def getDataFromFolder(folder):
    data5=getSession5(folder)
    data000=getSim320(folder,sim320csvs[0])
    data001=getSim320(folder,sim320csvs[1])
    data002=getSim320(folder,sim320csvs[2])
    print('Dataset Session5 size:',len(data5))
    print('Dataset 000 size:',len(data000))
    print('Dataset 001 size:',len(data001))
    print('Dataset 002 size:',len(data002))
    # data=data5 + data001 + data002
    # data=data5[::5]
    # df = shuffle(data)
    # return train_test_split(df, test_size=0.2, random_state=42)
    return shuffle(data001),shuffle(data002)
    # return shuffle(data5[::10]),shuffle(data001)

def genSession5(folder):
    data=shuffle(getSession5(folder))
    while True:
        for row in data:
            img = cv2.imread(row[0])[:200,200:-200,:]
            steering_angle = row[1]
            yield img,steering_angle/2.0,row[2],row[3],row[4]

def genSim001(folder):
    data=shuffle(getSim320(folder,sim320csvs[1]))
    while True:
        for row in data:
            img = cv2.imread(row[0])[20:140,:,:]
            steering_angle = row[1]
            yield img,steering_angle,row[2],row[3],row[4]

def genSim002(folder):
    data=shuffle(getSim320(folder,sim320csvs[2]))
    while True:
        for row in data:
            img = cv2.imread(row[0])[20:140,:,:]
            steering_angle = row[1]
            yield img,steering_angle,row[2],row[3],row[4]

def genSim003(folder):
    data=shuffle(getSim320(folder,sim320csvs[2]))
    while True:
        for row in data:
            img = cv2.imread(row[0])[20:140,:,:]
            steering_angle = row[1]
            yield img,steering_angle,row[2],row[3],row[4]

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
    data=shuffle(getPolysync(folder,polysynccsvs[0])[500:-500])
    while True:
        for row in data:
            img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
            steering_angle = row[1]
            yield img,steering_angle/5.0,0.8,row[3],row[4]

def genPolysync2(folder):
    data=getPolysync(folder,polysynccsvs[2])
    data=shuffle(data[500:2000] + data[3100:-500])
    while True:
        for row in data:
            img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
            steering_angle = row[1]
            yield img,steering_angle/5.0,0.8,row[3],row[4]

def genAll(folder):
    datasets=[]
    # datasets.append(shuffle(getPolysync(folder,polysynccsvs[0])[500:-500]))
    # data=getPolysync(folder,polysynccsvs[2])
    # data=shuffle(data[500:2000] + data[3100:-500])
    # datasets.append(data)
    # datasets.append(shuffle(getSim320(folder,sim320csvs[2])))
    # datasets.append(shuffle(getSim320(folder,sim320csvs[3])))
    datasets.append(shuffle(getSim320(folder,sim320csvs[4])))
    datasets.append(shuffle(getSession5(folder)))

    while True:
        nr=0
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle,row[2],row[3],row[4]
        # nr=nr+1
        # ex=int(np.random.uniform() * len(datasets[nr]))
        # row=datasets[nr][ex]
        # img = cv2.imread(row[0])[20:140,:,:]
        # steering_angle = row[1]
        # yield img,steering_angle,row[2],row[3],row[4]
        # nr=nr+1
        # ex=int(np.random.uniform() * len(datasets[nr]))
        # row=datasets[nr][ex]
        # img = cv2.imread(row[0])[20:140,:,:]
        # steering_angle = row[1]
        # yield img,steering_angle,row[2],row[3],row[4]
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[:200,200:-200,:]
        steering_angle = row[1]
        yield img,steering_angle/2.0,row[2],row[3],row[4]
