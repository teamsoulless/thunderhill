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

import multiprocessing

show_images=True
show_images=False

def dispatch_tasks(function, tasks, numthreads):
	pool = multiprocessing.Pool( numthreads )
	results = [pool.apply_async( function, t ) for t in tasks]
	pool.close()
	pool.join()
	return results

def worker(img1, steering_angle1, throttle, brake, speed):
        img1, steering_angle1 = RandomBrightness(img1, steering_angle1)
        img1, steering_angle1, throttle1 = RandomShift(img1, steering_angle1,throttle)
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

        speed1=np.abs(speed1)

        if speed1<50:
            throttle1=1
        if speed1<60 and steering_angle1==0:
            throttle1=1

        if speed1>70:
            throttle1=0

        if brake1>0.7:
            throttle1=0

        speed_cl=np.array(speedToClass(speed1))

        img1=img1.astype(np.float32)
        img1=-0.5+img1/255

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # if show_images:
        #     imgp = img1.copy()
        #     imgp=imgp+0.5
        #     cv2.putText(imgp,'%s'%(str(list(speed_cl[:8]))),(10,20), font, 0.4,(0,0,255),1,cv2.LINE_AA)
        #     cv2.putText(imgp,'%s'%(str(list(speed_cl[8:]))),(10,40), font, 0.4,(0,0,255),1,cv2.LINE_AA)
        #
        #     # cv2.putText(imgp,'%.3f %.1f %.1f %.1f'%(steering_angle,throttle, brake,speed),(10,70), font, 0.4,(0,0,255),1,cv2.LINE_AA)
        #     cv2.putText(imgp,'%.3f %.1f %.1f %.1f'%(steering_angle1,throttle1, brake1, speed1),(10,70), font, 0.4,(0,0,255),1,cv2.LINE_AA)
        #     cv2.imshow('imgp',imgp)
        #     cv2.waitKey(0)

        return [np.reshape(img1, (1, HEIGHT, WIDTH, DEPTH)), speed_cl], [steering_angle1,throttle1, brake1]

def generate_thunderhill_batches(gen, batch_size):
    batch_x = []
    batch_x2 = []
    batch_y = []
    tasks=[]
    while True:
        for img, steering_angle, throttle, brake, speed in gen:
            for img1, steering_angle1 in RandomFlip(img, steering_angle):

                img1 = Preproc(img1)

                tasks.append((img1, steering_angle1, throttle, brake, speed))

                if len(tasks)==BATCH_SIZE:
                    results = dispatch_tasks(worker, tasks, 1)#multiprocessing.cpu_count())
                    tasks=[]
                    for i,result in enumerate(results):
                        res_list = result.get()

                        batch_x.append(res_list[0][0])
                        batch_x2.append(res_list[0][1])
                        batch_y.append(res_list[1])
                    print(len(batch_x))
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

polysynccsvs=['dataset_polysync_1464466368552019/outputNew.csv','dataset_polysync_1464470620356308/outputNew.csv','dataset_polysync_1464552951979919/outputNew.csv']

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
            data.append((dd+'/'+row[0],float(row[-4]),float(row[-3]),float(row[-2]),float(row[-7]),float(row[-1])))
    return data

def genPolysync0(folder):
    data=shuffle(getPolysync(folder,polysynccsvs[0])[500:-500])
    while True:
        for row in data:
            img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
            steering_angle = row[1]
            yield img,steering_angle/5.0,0.8,row[3],row[4],row[-1]*2

def genPolysync2(folder):
    data=getPolysync(folder,polysynccsvs[2])
    data=shuffle(data[500:2000] + data[3100:-500])
    while True:
        for row in data:
            img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
            steering_angle = row[1]
            yield img,steering_angle/5.0,0.8,row[3],row[4],row[-1]*2

def genAll(folder):
    datasets=[]
    datasets.append(shuffle(getPolysync(folder,polysynccsvs[0])[500:-500]))
    data=getPolysync(folder,polysynccsvs[2])
    data=shuffle(data[500:2000] + data[3100:-500])
    datasets.append(data)
    datasets.append(shuffle(getSim320(folder,sim320csvs[2])))
    datasets.append(shuffle(getSim320(folder,sim320csvs[3])))
    datasets.append(shuffle(getSim320(folder,sim320csvs[4])))
    datasets.append(shuffle(getSession5(folder)))

    while True:
        nr=0
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
        steering_angle = row[1]
        if row[3]==0:
            throttle=1
        else:
            throttle=0
        yield img, steering_angle/5.0,throttle,row[3],row[4]
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[-250:1:-1,300:-300,:]
        steering_angle = row[1]
        if row[3]==0:
            throttle=1
        else:
            throttle=0
        yield img, steering_angle/5.0,throttle,row[3],row[4]
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle,row[2],row[3],row[4]
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle,row[2],row[3],row[4]
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle,row[2],row[3],row[4]
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = cv2.imread(row[0])[:200,200:-200,:]
        steering_angle = row[1]
        yield img,steering_angle/2.0,row[2],row[3],row[4]
