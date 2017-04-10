import pandas as pd
import numpy as np
from PIL import Image
import pickle
import csv
import os
import cv2
import sklearn
import random
import utm

# read in from different folders
#path = "/Volumes/TOSHIBA EXT/"
path = "/Volumes/TOSHIBA EXT/thunderhill-day1-data/1538/"
slow = 'thunderhill-day1-data/1543/'
# medium = ['1538','1610','1620','1645','1702','1708']
samples = []
with open(path+'output_processed.txt','r') as log_1:
  reader_1 = csv.reader(log_1)
  #reader_2 = csv.reader(log_2)
  next(reader_1, None)
  #next(reader_2, None)
  for line in reader_1:
    samples.append(line)
  #for row in reader_2:
  #  samples.append(row)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=16):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    sklearn.utils.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      input_all = []
      output = []
      imageArr = np.zeros((batch_size*2,480,960,3))
      speedArr = np.zeros((batch_size*2,1))
      outputArr = np.zeros((batch_size*2,2))
      for i,batch_sample in enumerate(batch_samples):
        center_angle = float(batch_sample[-4])
        # check steering distribution
        if center_angle == 0:
          continue
        throttle = float(batch_sample[-3])
        brake = float(batch_sample[-2])
        speed = float(batch_sample[-1])
        longitude = float(batch_sample[2])
        latitude = float(batch_sample[3])

        if brake > 0:
          throttle = -brake

        img = cv2.imread(path+batch_sample[0])
        center_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)[::-1,:,:]
        imageArr[i*2] = center_image
        speedArr[i*2] = speed
        outputArr[i*2] = np.array([center_angle,throttle])
        #input_all.append([center_image,speed])
        center_image_T = adjust_brightness(center_image)
        imageArr[i*2+1] = center_image_T
        speedArr[i*2+1] = speed
        outputArr[i*2+1] = np.array([center_angle,throttle])
        # input_all.append(center_image_T,speed])

        #single_output = [center_angle,throttle]
        #output.extend([single_output, single_output])

      #X_train = np.array(input_all)
      #y_train = np.array(output).reshape([-1,2])
      #print(X_train.shape)
      # add flipped images and angles to avoid bias
      # print(input_all)
      #flipped = np.array([np.array([np.fliplr(image),speed]) for [image,speed] in input_all])
      #print(images_flipped.shape)
      #angles_flipped = np.array([[-angle,throttle] for [angle,throttle] in output]).reshape([-1,2])
      #X_train = np.concatenate((X_train, flipped), axis=0)
      #X_train = np.array(X_train).reshape(-1,2)
      #y_train = np.concatenate((y_train, angles_flipped), axis=0)

      yield ({'inputImg': imageArr, 'input_speed': speedArr}, {'output': outputArr})

def adjust_brightness(image):
    rand = random.uniform(0.4, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * rand
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Flatten, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, merge, Input
from keras import backend as K

row, col, ch = 480,960,3

input_image = Input(shape=(row,col,ch),name='inputImg')
branch_image = Cropping2D(cropping=((50,180), (0,0)), input_shape=(row, col, ch))(input_image)
branch_image = Lambda(lambda x: (x / 255.0) - 0.5)(branch_image)
branch_image = Convolution2D(24, 5, 5, subsample=(2,2), activation='relu')(branch_image)
branch_image = Convolution2D(36, 5, 5, subsample=(2,2), activation='relu')(branch_image)
branch_image = Convolution2D(48, 5, 5, subsample=(2,2), activation='relu')(branch_image)
branch_image = Convolution2D(64, 3, 3, activation='relu')(branch_image)
branch_image = Convolution2D(64, 3, 3, activation='relu')(branch_image)
branch_image = Flatten()(branch_image)

input_speed = Input(shape=(1,),name='input_speed')

output = Lambda(lambda x: K.concatenate(x, axis=1))([branch_image, input_speed])
output = Dense(2,name='output')(output)

model =  Model((input_image, input_speed), output)

model.compile(optimizer='adam',loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=20)

model.save('thunderhill_med.h5')