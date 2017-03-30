import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import math
from sklearn.utils import shuffle
from keras.models import Sequential, Model, load_model
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import ELU

import matplotlib.pyplot as plt

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            #steering_angles = []
            #throttles = []
            actions = []
            for batch_sample in batch_samples:               
                name = './IMG/'+batch_sample[0].split('/')[-1]
                image = cv2.imread(name)
                steering = float(batch_sample[3])
                throttle = float(batch_sample[4])
                
                images.append(image)
                #steering_angles.append(steering)
                #throttles.append(throttle)
                actions.append((steering, throttle))

                image_flipped = np.fliplr(image)
                steering_flipped = -steering
                images.append(image_flipped)
                #steering_angles.append(steering_flipped)
                #throttles.append(throttle)
                actions.append((steering_flipped, throttle))

            X_train = np.array(images)
            #y_train = np.vstack((np.array(steering_angles),np.array(throttles)))
            y_train = np.array(actions)
            yield sklearn.utils.shuffle(X_train, y_train)

row, col, ch = 160, 320, 3

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5,border_mode='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5,border_mode='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48,3,3,border_mode='valid'))
#model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='valid'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(Dense(1164))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(Dense(100))  
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(Dense(50))
#model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(10))
#model.add(Dropout(0.2))
model.add(Activation('linear'))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam')
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=30)

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

model.save('model.h5')
