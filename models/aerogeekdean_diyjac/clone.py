import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load data
# data_path = './data/' # default sim data from Udacity
# data_path = '../Simulator/' # my sim data
data_path = './training_data/' # AWS sim data
csv_file = data_path + 'driving_log.csv'

steer_corr = 0.05 # [normalized +/-1.0] (+) right

images=[]
measurements = []

with open(csv_file) as f:
  reader = csv.reader(f)
  next(reader) # skip 1st line
  for line in reader:
    steering_angle = float(line[3])

    # center channel
#    if(np.abs(steering_angle)>0.1): 
    if(1): 
      source_path = line[0]
      filename = source_path.split('/')[-1] #remove intermediate path
      image_file = data_path + 'IMG/' + filename
      image = cv2.imread(image_file, cv2.COLOR_BGR2RGB)
      images.append(image)

      measurements.append(steering_angle)

    # left channel
    if(np.random.random()<.2):
#    if(1):
      source_path = line[1]
      filename = source_path.split('/')[-1] #remove intermediate path
      image_file = data_path + 'IMG/' + filename
      image = cv2.imread(image_file, cv2.COLOR_BGR2RGB)
      images.append(image)

      measurement = steering_angle + steer_corr #steering angle
      measurement = min(measurement, 1.0)
      measurements.append(measurement)

    # right channel
    if(np.random.random()<.2):
#    if(1):
      source_path = line[2]
      filename = source_path.split('/')[-1] #remove intermediate path
      image_file = data_path + 'IMG/' + filename
      image = cv2.imread(image_file, cv2.COLOR_BGR2RGB)
      images.append(image)

      measurement = steering_angle - steer_corr #steering angle
      measurement = max(measurement, -1.0)
      measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

# generate flipped image data
X_train_flip = np.array(np.fliplr(images))
y_train_flip = -np.array(measurements)

X_train = np.concatenate((X_train, X_train_flip), axis=0)
y_train = np.concatenate((y_train, y_train_flip), axis=0)

# train model  
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D

imshape = X_train[0].shape

model = Sequential()
model.add(Lambda(lambda x: ((x/255.0)-0.5),
                 input_shape=imshape,
                 name='lambda'))
model.add(Cropping2D(cropping=((50,20),(0,0)),
                     name='crop'))
model.add(Conv2D(nb_filter=24,
                 nb_row=5,
                 nb_col=5,
                 subsample=(2,2),
                 border_mode='valid',
                 activation='elu',
                 name='conv1'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(nb_filter=36,
                 nb_row=5,
                 nb_col=5,
                 subsample=(2,2),
                 border_mode='valid',
                 activation='elu',
                 name='conv2'))
model.add(Conv2D(nb_filter=48,
                 nb_row=5,
                 nb_col=5,
		 subsample=(2,2),
                 border_mode='valid',
                 activation='elu',
                 name='conv3'))
model.add(Conv2D(nb_filter=64,
                 nb_row=3,
                 nb_col=3,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='elu',
                 name='conv4'))
model.add(Conv2D(nb_filter=64,
                 nb_row=3,
                 nb_col=3,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='elu',
                 name='conv5'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten(name='flat'))
model.add(Dropout(0.5, name='dropout'))
model.add(Dense(100, activation='elu', name='dense1'))
model.add(Dense(50, activation='elu', name='dense2'))
model.add(Dense(10, activation='elu', name='dense3'))
model.add(Dense(1, name='output'))

from functools import reduce
def prod(input):
  return reduce(lambda x, y: x*y, input, 1)

print('Layer input shapes:')
for layer in model.layers:
  shape = layer.get_input_shape_at(0)
  print(layer.name, shape, ' = '+str(prod(shape[1:]))+' neurons', sep='\t')



model.compile(loss='mse',
              optimizer='adam')
history_obj = model.fit(X_train, y_train,
                        validation_split=0.2,
                        shuffle=True,
                        nb_epoch=3,
                        batch_size=64)

model.save('model.h5')

### print the keys contained in the history object
print('History object keys:')
print(history_obj.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


exit()
