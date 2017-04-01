import numpy as np
import pandas as pd
import cv2
from glob import glob
import os
import csv

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformations import *
from config import *

show_images=True
show_images=False


def generate_thunderhill_batches(gen, batch_size):
    batch_x = []
    batch_x2 = []
    batch_y = []
    while True:
        for img, steering_angle, throttle, brake, speed, longitude, latitude in gen:
            for img1, steering_angle1 in RandomFlip(img, steering_angle):
                img1, steering_angle1, throttle1 = RandomBrightness(img1, steering_angle1,throttle)
                img1 = Preproc(img1)
                img1, steering_angle1, throttle1 = RandomShift(img1, steering_angle1,throttle1)
                img1, steering_angle1, throttle1 = RandomRotate(img1, steering_angle1,throttle1)

                brake1=brake

                if steering_angle1>1.5:
                    throttle1=0
                if steering_angle1<-1.5:
                    throttle1=0

                if np.random.uniform() < 0.5:
                    speed1 = speed * np.random.uniform()
                else:
                    speed1 = speed

                # speed_cl=np.array(speedToClass(speed))
                # speed_cl1=np.array(speedToClass(speed1))

                if speed1 < 20*0.44704:
                    throttle1=1
                if speed1 > 50*0.44704:
                    throttle1=0
                if speed1 > 70*0.44704:
                    throttle1=0
                    brake=1

                if brake>0.7:
                    throttle=0

                # if speed_cl[-2]==1 and steering_angle1!=0:
                #     throttle1=0
                # if speed_cl[-1]==1 and steering_angle1!=0:
                #     brake1=1
                #     throttle1=0

                batch_x.append(np.reshape(img1, (1, HEIGHT, WIDTH, DEPTH)))

                spd1 = speed1/40 - 0.5
                batch_x2.append(spd1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if show_images:
                    imgp = Preproc(img.copy()[::-1,:,:])
                    imgshow=(np.concatenate((imgp,img1[::-1,:,:]),axis=0)+0.5)*0.5

                    imgshow = cv2.resize(imgshow,(0,0),fx=2,fy=2)
                    # cv2.putText(imgshow,'%s'%(str(list(speed_cl))),(10,30), font, 0.8,(0,0,255),2,cv2.LINE_AA)
                    # cv2.putText(imgshow,'%s'%(str(list(speed_cl1))),(10,190), font, 0.8,(0,0,255),2,cv2.LINE_AA)
                    cv2.putText(imgshow,'%.3f %.1f %.1f %.1f'%(steering_angle,throttle, brake,speed),(10,140), font, 0.8,(0,0,255),2,cv2.LINE_AA)
                    cv2.putText(imgshow,'%.3f %.1f %.1f %.1f'%(steering_angle1,throttle1, brake1,speed1),(10,300), font, 0.8,(0,0,255),2,cv2.LINE_AA)

                    cv2.putText(imgshow,'%.5f %.5f'%(longitude, latitude),(10,110), font, 0.8,(0,0,255),2,cv2.LINE_AA)
                    cv2.imshow('img',imgshow)
                    cv2.waitKey(0)


                batch_y.append([steering_angle1,throttle1, brake1])
                if len(batch_x) == batch_size:
                    batch_x, batch_x2, batch_y = shuffle(batch_x, batch_x2, batch_y)
                    yield [np.vstack(batch_x), np.vstack(batch_x2)], np.vstack(batch_y)
                    batch_x = []
                    batch_x2 = []
                    batch_y = []

thunderhill_data_dir_1='/home/karol/projects/udacity/Racing/thunderhill-day1-data/*'
thunderhill_datasets_1=glob(thunderhill_data_dir_1)
thunderhill_data_dir_2='/home/karol/projects/udacity/Racing/thunderhill-day2-data/*'
thunderhill_datasets_2=glob(thunderhill_data_dir_2)

def genThDay1():
    while True:
        for dataset in thunderhill_datasets_1:
            with open(dataset + '/output_processed.txt', 'r') as csvfile:
                data_tmp = shuffle(list(csv.reader(csvfile, delimiter=','))[1:])
                dd='/'.join(dataset.split('/'))
                for row in data_tmp:
                    if row[-1] == '':
                        continue
                    # row=row[1:]
                    img = cv2.imread(dataset + '/' + row[0]) # image
                    steering_angle = float(row[14])
                    throttle = float(row[-1])
                    brake = float(row[16])
                    speed = float(row[17])
                    longtitude = float(row[2])
                    latitude = float(row[3])

                    if throttle<-0.1:
                        throttle=-0.1
                    throttle=throttle+0.1
                    throttle=throttle/0.3
                    if throttle>1:
                        throttle=1

                    yield img,steering_angle, throttle, brake, speed, longtitude, latitude
                    break
def genThDay2():
    while True:
        for dataset in thunderhill_datasets_2:
            with open(dataset + '/output_processed.txt', 'r') as csvfile:
                data_tmp = shuffle(list(csv.reader(csvfile, delimiter=','))[1:])
                dd='/'.join(dataset.split('/'))
                for row in data_tmp:
                    if row[-1] == '':
                        continue
                    # row=row[1:]
                    img = cv2.imread(dataset + '/' + row[0]) # image
                    steering_angle = float(row[14])
                    throttle = float(row[-1])
                    brake = float(row[16])
                    speed = float(row[17])
                    longtitude = float(row[2])
                    latitude = float(row[3])

                    if throttle<-0.1:
                        throttle=-0.1
                    throttle=throttle+0.1
                    throttle=throttle/0.3
                    if throttle>1:
                        throttle=1

                    yield img,steering_angle, throttle, brake, speed, longtitude, latitude
                    break
def genThDay1_2():
    while True:
        for dataset in thunderhill_datasets_1 + thunderhill_datasets_2:
            with open(dataset + '/output_processed.txt', 'r') as csvfile:
                data_tmp = shuffle(list(csv.reader(csvfile, delimiter=','))[1:])
                dd='/'.join(dataset.split('/'))
                for row in data_tmp:
                    if row[-1] == '':
                        continue
                    # row=row[1:]
                    img = cv2.imread(dataset + '/' + row[0]) # image
                    steering_angle = float(row[14])
                    throttle = float(row[15])
                    brake = float(row[16])
                    speed = float(row[17])
                    longtitude = float(row[2])
                    latitude = float(row[3])

                    if throttle<-0.1:
                        throttle=-0.1
                    throttle=throttle+0.1
                    throttle=throttle/0.3
                    if throttle>1:
                        throttle=1

                    yield img,steering_angle, throttle, brake, speed, longtitude, latitude
                    break
