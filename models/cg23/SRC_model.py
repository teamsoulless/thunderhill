# ----------------------------------------------------------------------------------
# Self Racing Cars - SRC_model.py
# ----------------------------------------------------------------------------------
'''
Part 1: Build model using keras

Part 2: Setup the data for the generator, perform image augmentation,
and generate images/steerings to train, validate and test model

Part 3: Main Program for running train, validation and test

By: Chris Gundling, chrisgundling@gmail.com
'''

from __future__ import print_function

import numpy as np
import pandas as pd
import csv
import cv2
import random
import matplotlib as mpl
import matplotlib.image as mpimg
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import time

from os import path
from collections import defaultdict
from numpy import sin, cos
from scipy.misc import imread, imresize, imsave
from scipy import ndimage
from scipy import misc

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras import backend as K

from PIL import Image
from PIL import ImageOps


###########################################################################################
# Model Structure
###########################################################################################

'''
NVIDIA model layers have been modified for 320X160 images. Images should not be resized 
as it will alter the top-down view settings. First layer now uses 8X8 convs with stride 4 
and the third layer used 2X2 convs with stride 2.
'''
def build_cnn(image_size=None,weights_path=None):
    image_size = image_size or (128, 128)
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3, )

    img_input = Input(input_shape)

    # Layer 1
    #x = Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2), border_mode='valid', init='he_normal')(img_input)
    x = Convolution2D(24, 8, 8, activation='elu', subsample=(4, 4), border_mode='valid', init='he_normal')(img_input)

    # Layer 2
    x = Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2), border_mode='valid', init='he_normal')(x)

    # Layer 3
    #x = Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2), border_mode='valid', init='he_normal')(x)
    x = Convolution2D(48, 2, 2, activation='elu', subsample=(2, 2), border_mode='valid', init='he_normal')(x)

    # Layer 4
    x = Convolution2D(64, 3, 3, activation='elu', subsample=(1, 1), border_mode='valid', init='he_normal')(x)

    # Layer 5
    x = Convolution2D(64, 3, 3, activation='elu', subsample=(1, 1), border_mode='valid', init='he_normal')(x)
    
    # Flatten
    y = Flatten()(x)

    # FC 1
    y = Dense(100, activation='elu', init='he_normal')(y)

    # FC 2
    y = Dense(50, activation='elu', init='he_normal')(y)

    # FC 3
    y = Dense(10, activation='elu', init='he_normal')(y)

    # FC 4
    y = Dense(1, init='he_normal')(y)

    model = Model(input=img_input, output=y)
    model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')
    return model

##########################################################################################
# Data Processing and Augmentations
##########################################################################################
'''
Generator assumes the dataset folder has the following structure:

driving_log.csv  IMG/
'''

# The following are all the functions used for data processing/augmentation
# ----------------------------------------------------------------------------------

def data_setup(steering_log):
    # Read in .csv and store all information numpy arrays and pandas dataframes
    df_steer = pd.read_csv(steering_log,usecols=['center','left','right','steering','speed'],index_col = False)
    df_steer['t1'] = df_steer['center'].str.split('.').str[0]
    df_steer['timestamp'] = df_steer['t1'].str.split('_').str[4:8]
    x = np.zeros((df_steer.shape[0]))
    for i in range(df_steer.shape[0]):
        x[i] = np.int(''.join(df_steer['timestamp'].iloc[i]))
    df_steer['timestamp'] = x
    
    angle = np.zeros((df_steer.shape[0],1))
    time = np.zeros((df_steer.shape[0],1))
    speed = np.zeros((df_steer.shape[0],1))
    
    angle[:,0] = df_steer['steering'].values
    time[:,0] = df_steer['timestamp'].values.astype(int)
    speed[:,0] = df_steer['speed'].values
    
    data = np.append(time,angle,axis=1)
    data = np.append(data,speed,axis=1)

    image_paths = pd.read_csv(steering_log,usecols=['center','left','right'],index_col = False)
    return data, image_paths


def read_data(data, image_paths):
    # Convert the .csv info to dictionaries for data generator
    steerings = defaultdict(list)
    speeds = defaultdict(list)
    timestamps = defaultdict(list)
    image_path = defaultdict(list)
    image_path_l = defaultdict(list)
    image_path_r = defaultdict(list)
    for i in range(data.shape[0]):
        time, angle, speed = int(data[i,0]), float(data[i,1]), float(data[i,2])
        steerings[time].append(angle)
        speeds[time].append(speed)
        timestamps[time].append(time)
        image_path[time].append(image_paths['center'].iloc[i])
        image_path_l[time].append(image_paths['left'].iloc[i].strip(' '))
        image_path_r[time].append(image_paths['right'].iloc[i].strip(' '))
    return steerings, speeds, timestamps, image_path, image_path_l, image_path_r


def camera_adjust(angle,speed,camera):
    # Steering angle adjustment for left, right camera
    angle_adj = 0.25
    if camera == 'right':
        angle_adj = -angle_adj
    angle = angle_adj + angle
    return angle


def image_rotate(img):
    # Rotate image randomly to simulate camera jitter
    rotate = random.uniform(-1, 1)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img


def image_shift(img):
    # Shift image and transform steering angle
    trans_range = 80
    shift_x = trans_range*np.random.uniform()-trans_range/2 # random.randint(-40, 40)
    shift_y = 0 
    M = np.float32([[1,0,shift_x],[0,1,shift_y]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    ang_adj = shift_x/trans_range*2*.2 #shift_x / 200.
    return img, ang_adj


def image_blur(img):
    # Blur image with random kernel
    kernel_size = random.randint(1, 5)
    if kernel_size % 2 != 1:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def image_HSV(img):
    # HSV brightness transform
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    brightness = np.random.uniform(0.5,1.1)
    img[:,:,2] = img[:,:,2]*brightness
    return cv2.cvtColor(img,cv2.COLOR_HSV2RGB)


def image_viewpoint_transform(im, isdeg=True):
    # Viewpoint transform for recovery
    theta = random.randint(-80,80)
    if isdeg:
        theta = np.deg2rad(theta)

    f = 2.
    h, w, _ = im.shape
    cx = cz = cos(0)
    sx = sz = sin(0)
    cy = cos(theta)
    sy = sin(theta)

    R = np.array([[cz * cy, cz * sy * sx - sz * cx],[sz * cy, sz * sy * sx + cz * cx],[ -sy, cy * sx]], np.float32)

    pts1 = [[-w/2, -h/2],[w/2, -h/2],[w/2, h/2],[-w/2, h/2]]

    pts2 = []
    mx, my = 0, 0
    for i in range(4):
        pz = pts1[i][0] * R[2][0] + pts1[i][1] * R[2][1];
        px = w / 2 + (pts1[i][0] * R[0][0] + pts1[i][1] * R[0][1]) * f * h / (f * h + pz);
        py = h / 2 + (pts1[i][0] * R[1][0] + pts1[i][1] * R[1][1]) * f * h / (f * h + pz);
        pts2.append([px, py])

    pts2 = np.array(pts2, np.float32)
    pts1 = np.array([[0, 0],[w, 0],[w, h],[0, h]], np.float32)

    x1, x2 = int(min(pts2[0][0], pts2[3][0])), int(max(pts2[1][0], pts2[2][0]))
    y1, y2 = int(max(pts2[0][1], pts2[1][1])),  int(min(pts2[2][1], pts2[3][1]))

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(im, M, (w, h), cv2.INTER_NEAREST | cv2.INTER_NEAREST)

    x1 = np.clip(x1, 0, w)
    x2 = np.clip(x2, 0, w)
    y1 = np.clip(y1, 0, h)
    y2 = np.clip(y2, 0, h)
    z = dst[y1:y2, x1:x2]
    x, y, _ = z.shape
    if x == 0 or y == 0:
        return
    return cv2.resize(z, (w, h), interpolation = cv2.INTER_AREA), -np.rad2deg(theta) / 200.


def birds_eye(img):      
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    
    # Four Source Points
    src = np.float32(
        [[100, 75],
         [0, 120],
         [200, 75],
         [320, 120]])
    
    # Four Destination Points
    dst = np.float32(
        [[80, 0],
         [100, 160],
         [215, 0],
         [215, 160]])
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped
    

def read_images(image_log, image_path, ids, image_size):
    # Baseline function for reading images and perspective transform
    imgs = []
    for id in image_path:
        prefix = path.join(image_log,id)
        img = mpimg.imread(prefix)
        
        # Birds Eye
        img = birds_eye(img)        
        imgs.append(img)
        
    if len(imgs) < 1:
        print('Error no image at timestamp')
        print(ids)
        
    img_block = np.stack(imgs, axis=0)
    
    if K.image_dim_ordering() == 'th':
        img_block = np.transpose(img_block, axes = (0, 3, 1, 2)) 
    return img_block


def image_augment(image_log, image_path, ids, image_size, option):
    imgs = []
    for id in image_path:
        prefix = path.join(image_log,id)
        img = mpimg.imread(prefix)

        if option == 1:
            # Flip Image
            img = np.fliplr(img)

            # Rotate randomly by small amount (not a viewpoint transform)
            img = image_rotate(img)

            # Blur Image
            img = image_blur(img)

            # Image Brightness
            img = image_HSV(img)
            #imsave('aug1.jpg', img)
            
            # Birds Eye
            img = birds_eye(img)
            
            # Steering angle not adjusted
            ang_adj = 0.

        if option == 2:
            # Birds Eye
            img = birds_eye(img)

            # Shift Image
            img, ang_adj = image_shift(img)

        if option == 3:
            # Viewpoint Transform
            img, ang_adj = image_viewpoint_transform(img, isdeg=True)
            
            # Birds Eye
            img = birds_eye(img)

        imgs.append(img)
        
    if len(imgs) < 1:
        print('Error no image at timestamp')
        print(ids)
        
    img_block = np.stack(imgs, axis=0)
    
    if K.image_dim_ordering() == 'th':
        img_block = np.transpose(img_block, axes = (0, 3, 1, 2))
    return img_block, ang_adj


def normalize_input(x):
    return (x - 128.) / 128.  #255.


def exact_output(y):
    return y


# Data generator (output steering angles and images)
# ---------------------------------------------------------------------------------
def data_generator(steering_log, image_log, image_folder, gen_type='train',
                   camera='center', batch_size=64, time_factor=10, image_size=0.5,
                   timestamp_start=None, timestamp_end=None, shuffle=True,
                   preprocess_input=normalize_input,
                   preprocess_output=exact_output):
    
    # Constants
    # -----------------------------------------------------------------------------
    minmax = lambda xs: (min(xs), max(xs))

    # Read all steering angles, speeds and timestamps
    # -----------------------------------------------------------------------------
    data, image_paths = data_setup(steering_log)
    steerings, speeds, image_stamps, image_path, image_path_l, image_path_r = read_data(data, image_paths)

    # More data exploration stats
    # -----------------------------------------------------------------------------
    print('timestamp range for all steerings: %d, %d' % minmax(steerings.keys()))
    print('timestamp range for all images: %d, %d' % minmax(image_stamps.keys()))
    print('min and max # of steerings per time unit: %d, %d' % minmax(map(len, steerings.values())))
    print('min and max # of images per time unit: %d, %d' % minmax(map(len, image_stamps.values())))
    
    # Generate images and steerings within one time unit
    # (Mean steering angles used for multiple steering angles within a single unit)
    # (Doesn't apply to simulator data, but does for real world data from car)
    # -----------------------------------------------------------------------------
    start = max(min(steerings.keys()), min(image_stamps.keys()))
    if timestamp_start:
        start = max(start, timestamp_start)
    end = min(max(steerings.keys()), max(image_stamps.keys()))
    if timestamp_end:
        end = min(end, timestamp_end)
    print("sampling data from timestamp %d to %d" % (start, end))
    
    # While loop for data generator
    # -----------------------------------------------------------------------------
    i = start
    if gen_type == 'train':
        thresh = 0.25
    else:
        thresh = 0
    print(thresh)
    epochs = 0
    epoch_num = 0
    bias = 0.
    x_buffer, y_buffer, buffer_size = [], [], 0
    j = 0
    while True:
        if i > end:
            i = start

        if gen_type =='train':
            camera_select = 'center' #random.choice(camera) not using left/right
        else:
            camera_select = camera

        coin = random.randint(1, 2)
        if steerings[i] and image_stamps[i]:
            if camera_select == 'right':
                if coin == 1:
                    images = read_images(image_log, image_path_r[i], image_stamps[i], image_size)
                elif coin == 2:
                    option = random.randint(1, 3)
                    images, ang_adj = image_augment(image_log, image_path_r[i], image_stamps[i], image_size, option)
            elif camera_select == 'left':
                if coin == 1:
                    images = read_images(image_log, image_path_l[i], image_stamps[i], image_size)
                elif coin == 2:
                    option = random.randint(1, 3)
                    images, ang_adj = image_augment(image_log, image_path_l[i], image_stamps[i], image_size, option)
            elif camera_select == 'center':
                if gen_type == 'train':
                    if coin == 1:
                        images = read_images(image_log, image_path[i], image_stamps[i], image_size)
                    elif coin == 2:
                        option = random.randint(1, 3)
                        images, ang_adj = image_augment(image_log, image_path[i], image_stamps[i], image_size, option)
                else:
                    images = read_images(image_log, image_path[i], image_stamps[i], image_size)

            # Mean angle/speed with a timestamp
            angle = np.repeat([np.mean(steerings[i])], images.shape[0])
            speed = np.repeat([np.mean(speeds[i])], images.shape[0])

            # Adjust the steerings of the offcenter cameras
            if camera_select != 'center':
                angle = camera_adjust(angle[0],speed[0],camera_select)
                angle = np.repeat([angle], images.shape[0])

            # Adjust steering angle for flips and shifts
            if gen_type == 'train':
                if coin == 2 and option == 1:
                    angle = -angle
                if coin == 2 and option > 1:
                    angle = angle + ang_adj

            if (abs(angle) + bias) >= thresh:       
                # Pass images and steerings to yield
                x_buffer.append(images)
                y_buffer.append(angle)
                buffer_size += images.shape[0]
                if buffer_size >= batch_size:
                    epoch_num += batch_size
                    if epoch_num == 8000:
                        epochs += 1
                        epoch_num = 0
                        bias = 0.0 + (epochs*0.05)
                        print()
                        print(bias)
                if buffer_size >= batch_size:
                    indx = range(buffer_size)
                    if gen_type == 'train':
                        np.random.shuffle(indx)
                    x = np.concatenate(x_buffer, axis=0)[indx[:batch_size], ...]
                    y = np.concatenate(y_buffer, axis=0)[indx[:batch_size], ...] 
                    x_buffer, y_buffer, buffer_size = [], [], 0
                    yield preprocess_input(x.astype(np.float32)), preprocess_output(y)

        if shuffle:
            i = int(random.choice(image_stamps.keys()))
            while i not in image_stamps.keys():
                i = int(random.choice(range(start,end)))
        else:
            i += 1
            while i not in image_stamps.keys():
                i += 1
                if i > end:
                    i = start

##########################################################################################
# Main Program
##########################################################################################

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Testing Udacity SRC Models")
    parser.add_argument('--dataset', type=str, help='dataset folder with csv and image folders')
    parser.add_argument('--model', type=str, help='model to evaluate, current list: {cnn}')
    parser.add_argument('--resized-image-height', type=int, help='image resizing')
    parser.add_argument('--resized-image-width', type=int, help='image resizing')
    parser.add_argument('--nb-epoch', type=int, help='# of training epoch')
    parser.add_argument('--camera', type=str, default='center', help='camera to use, default is center')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    args = parser.parse_args()

    # Model and data gen args
    # ---------------------------------------------------------------------------------
    dataset_path = args.dataset
    model_name = args.model
    image_size = (args.resized_image_height, args.resized_image_width)
    camera = args.camera
    camera_train = 'center','right','left'
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    weights_path = None

    # Data paths
    # ---------------------------------------------------------------------------------
    # build model and train it
    steering_log = path.join(dataset_path, 'driving_log_11.csv')
    image_log = dataset_path
    camera_images = dataset_path
    epoch = 0

    # Model build
    # ---------------------------------------------------------------------------------
    model_builders = {'cnn': (build_cnn, normalize_input, exact_output)}

    if model_name not in model_builders:
        raise ValueError("unsupported model %s" % model_name)
    model_builder, input_processor, output_processor = model_builders[model_name]
    model = model_builder(image_size,weights_path)
    print('model %s built...' % model_name)
        
    # Use data_sim.py to generate training data
    # ---------------------------------------------------------------------------------
    if not weights_path:
        train_generator = data_generator(steering_log=steering_log,
                            image_log=image_log,
                            image_folder=camera_images,
                            gen_type='train',
                            camera=camera_train,
                            batch_size=batch_size,
                            image_size=image_size,
                            timestamp_start=204030536,
                            timestamp_end=210120660,
                            shuffle=True,
                            preprocess_input=input_processor,
                            preprocess_output=output_processor)
 
        # Use data_sim.py to generate validation data
        # -----------------------------------------------------------------------------
        val_generator = data_generator(steering_log=steering_log,
                            image_log=image_log,
                            image_folder=camera_images,
                            gen_type='val',
                            camera=camera,
                            batch_size=240, 
                            image_size=image_size,
                            timestamp_start=203954079,
                            timestamp_end=204030464,
                            shuffle=False,
                            preprocess_input=input_processor,
                            preprocess_output=output_processor)
    
        # Training the model - with EarlyStopping, ModelCheckpoint
        # ------------------------------------------------------------------------------
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0), 
                    ModelCheckpoint(filepath=os.path.join('model_sim_{epoch:02d}.hdf5'), 
                    monitor='val_loss', verbose=0, save_best_only=False)]
        model.fit_generator(train_generator, samples_per_epoch=8000, nb_epoch=nb_epoch,verbose=1,
                    callbacks=callbacks, validation_data=val_generator,nb_val_samples=480)

        print('Model was successfully trained and saved...')
                

    # Use data generator to generate test data
    # Test set here is same as validation data - seperate script was used for final tests
    # -----------------------------------------------------------------------------------
    test_generator = data_generator(steering_log=steering_log,
                        image_log=image_log, 
                        image_folder=camera_images,
                        gen_type='test',
                        camera=camera,
                        batch_size=480,
                        image_size=image_size,
                        timestamp_start=203954079,
                        timestamp_end=204030464,
                        shuffle=False,
                        preprocess_input=input_processor,
                        preprocess_output=output_processor)
       
    print('Testing...')

    test_x, test_y = test_generator.next()
    print('test data shape:', test_x.shape, test_y.shape)

    # Store test predictions
    yhat = model.predict(test_x)
        
    # Use dataframe to write results, calculate RMSE and plot actual vs. predicted steerings
    # ------------------------------------------------------------------------------------ 
    df_test = pd.read_csv('output1.csv',usecols=['frame_id','steering_angle','pred'],index_col = None)
    df_test['steering_angle'] = test_y
    df_test['pred'] = yhat # test_res
    df_test.to_csv('output2.csv')
        
    # Calculate RMSE
    # ------------------------------------------------------------------------------------
    sq = 0
    mse = 0
    for j in range(test_y.shape[0]):
        sqd = ((yhat[j]-test_y[j])**2)
        sq = sq + sqd
    print(sq)
    mse = sq/480
    rmse = np.sqrt(mse)
    print("model evaluated RMSE:", rmse)

    # Plot the results
    # ------------------------------------------------------------------------------------
    plt.figure(figsize = (32, 8))
    plt.plot(test_y, 'r.-', label='target')
    plt.plot(yhat, 'b.-', label='predict')
    plt.legend(loc='best')
    plt.title("RMSE Evaluated on 480 TimeStamps: %.4f" % rmse)
    plt.show()
    model_fullname = "%s_%d.png" % (model_name, int(time.time()))
    plt.savefig(model_fullname)


if __name__ == '__main__':
    main()
