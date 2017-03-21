# Load pickled data
import pandas as pd
import tensorflow as tf 
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from docutils.nodes import image
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import functools
from Preprocess import *
import numpy as np
from keras.optimizers import Adam
import multiprocessing as mp
import threading
from keras.layers.wrappers import TimeDistributed
import time

logger = mp.log_to_stderr()

LOADMODEL = False
ANGLESFED = 1


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
#flags.DEFINE_string('trainingCSV', '../simulator/simulator-linux/driving_log.csv', "training data")
#flags.DEFINE_string('trainingCSV', '../simulator/data/data/driving_log.csv', "training data")


def generateImagesFromPaths(data, batchSize, timeLength, inputShape, outputShape, transform, angles, images, train = False):
    """
        The generator function for the training data for the fit_generator
        Input:
        data - an pandas dataframe containing the paths to the images, the steering angle,...
        batchSize, the number of values, which shall be returned per call
    """
    start = 0
    while 1:
        returnArr = np.zeros((batchSize, timeLength, inputShape[0], inputShape[1], inputShape[2]))
        vecArr = np.zeros((batchSize, timeLength, 6))
        labels = np.zeros((batchSize, 1))
        weights = np.zeros((batchSize))
        for j in range(batchSize):
            start += timeLength
            start %=(len(data)-timeLength)
            flip = np.random.rand()
            indices = np.arange(start, start+timeLength, 1)
            for i,index in zip(range(len(indices)),indices):
                row = data.iloc[index]
                image = images[int(row['angleIndex'])]
                label = np.array([row['steering'], row['throttle'], row['brake']])
                xVector = row[['positionX', 'positionY', 'positionZ', 'orientationX', 'orientationY', 'orientationZ']].values
                #if(image.shape[0] != 320): image = cv2.resize(image, (320, 160))
                
                #if flip>.5:
                #    image = mirrorImage(image)
                #    label[0] *= -1
                #if(train):
                #    image, label[0] = augmentImage(image, label[0])
                labels[j,0] = label[0]
                returnArr[j,i] = image            
                weights[j] = row['norm']
                xVector = np.array([val + np.random.rand()*0.02 - 0.01 for val in xVector])
                vecArr[j,i] = xVector
                #plt.imshow(image)
                #plt.title(label[0])
                #plt.show()
                #if(train):
                #    print('Train: ', label[0])#
                #else:
                #    print('Val: ', label[0])
            #print(labels[:,0])
            yield(returnArr,labels, weights)
                

def customLoss(y_true, y_pred):
    """
        This loss function adds some constraints on the angle to 
        keep it small, if possible
    """
    return K.mean(K.square(y_pred - y_true), axis=-1) #+.01* K.mean(K.square(y_pred), axis = -1)


def showSamplesCompared(img1, func, storeName, title1='', title2=''):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=((16,6)))
    ax[0].imshow(img1)
    ax[0].set_title(title1)
    ax[1].imshow(func(img1))
    ax[1].set_title(title2)
    #plt.savefig('./output_images/'+storeName)
    plt.show()

def getNormFactor(angle, hist, edges):
    for i, edge in enumerate(edges[:-1]):
        if(angle>edge and angle< edges[i+1]):
            return hist[i]
    return hist[-1]

def retrieveVectors(vecString):
    split = vecString.split(":")
    return pd.Series([float(split[0]), float(split[1]), float(split[2])])
        

def main():
    img = mpimg.imread('/home/jendrik/git/thunderhill_data/dataset_sim_001_km_320x160/IMG/center_2017_03_07_07_21_54_311.jpg')
    h, w = img.shape[:2]
    src = np.float32([[w/2 - 57, h/2], [w/2 + 57, h/2], [w+140,h], [-140,h]])
    dst = np.float32([[w/4,0], [w*3/4,0], [w*3/4,h], [w/4,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    transform = functools.partial(perspectiveTransform, M = M.copy())
    #plt.imshow(preprocessImage(img, transform))
    #plt.show()
    
    #showSamplesCompared(img, transform, '', '', '')
    plt.xkcd()
    np.random.seed(0)
    #data = pd.read_csv('/home/jendrik/git/thunderhill_data/dataset_sim_000_km_few_laps/driving_log.csv', 
    #                   header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    #data['positionX'], data['positionY'], data['positionZ'] = data['position'].apply(retrieveVectors)
    #data['orientationX'], data['orientationY'], data['orientationZ'] = data['orientation'].apply(retrieveVectors)
    #data['center'] = '/home/jendrik/git/thunderhill_data/dataset_sim_000_km_few_laps/'+data['center'].apply(lambda x: x.strip())
    data1 = pd.read_csv('/home/jendrik/git/thunderhill_data/dataset_sim_001_km_320x160/driving_log.csv', 
                       header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    data1['center'] = '/home/jendrik/git/thunderhill_data/dataset_sim_001_km_320x160/'+data1['center'].apply(lambda x: x.strip())
    data1[['positionX','positionY','positionZ']] = data1['position'].apply(retrieveVectors)
    data1[['orientationX','orientationY','orientationZ']] = data1['orientation'].apply(retrieveVectors)
    data2 = pd.read_csv('/home/jendrik/git/thunderhill_data/dataset_sim_002_km_320x160_recovery/driving_log.csv', 
                       header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    data2['center'] = '/home/jendrik/git/thunderhill_data/dataset_sim_002_km_320x160_recovery/'+data2['center'].apply(lambda x: x.strip())
    data2[['positionX','positionY','positionZ']] = data2['position'].apply(retrieveVectors)
    data2[['orientationX','orientationY','orientationZ']] = data2['orientation'].apply(retrieveVectors)
    #data['right'] = '../simulator/data/data/'+data['right'].apply(lambda x: x.strip())
    #data['left'] = '../simulator/data/data/'+data['left'].apply(lambda x: x.strip())
    angles = []
    images = []
    """data2 = pd.read_csv('../simulator/simulator-linux/driving_log.csv', header = None, names=['center','left', 'right', 'steering',
                                                               'throttle', 'break', 'speed'])
    data = data.append(data2)"""
    dataNew = pd.DataFrame()
    offset = 0
    
    print(data1['positionX'])
    for dat in [data1, data2]:
        angles.extend(dat['steering'].values)
        for row in dat.iterrows():
            dat.loc[row[0], 'angleIndex'] = row[0]+ offset
            images.append(preprocessImage(mpimg.imread(row[1]['center'].strip())))
            #images.append(transform(mpimg.imread(row[1]['center'].strip())))
        offset+=100
        dataNew = dataNew.append(dat.ix[100:])
    # TODO: Normalisation of position and orientation
    print(len(dataNew), dataNew.columns)
    hist, edges = np.histogram(dataNew['steering'], bins = 31)
    hist = 1./np.array([val if val > len(dataNew)/20. else len(dataNew)/20. for val in hist])
    hist*=len(dataNew)/20.
    print(hist, len(dataNew))
    dataNew['norm'] = dataNew['steering'].apply(lambda x: getNormFactor(x, hist, edges))
    print(dataNew['norm'].unique())
    del data1, data2
    
    for col in ['positionX', 'positionY', 'positionZ', 'orientationX', 'orientationY', 'orientationZ']:
        vals = dataNew[col].values
        mean = np.mean(vals)
        std = np.std(vals)
        dataNew[col] -= mean
        dataNew[col] /= std
        print('%s Mean:%.3f Std:%.3f' %(col, mean, std))
    
    dataNew = shuffle(dataNew, random_state=0)
    plt.figure(1, figsize=(8,4))
    plt.hist(dataNew['steering'], bins =31)
    
    #plt.show()
    
    dataTrain, dataTest= train_test_split(dataNew, test_size = .2)
    dataTrain, dataVal= train_test_split(dataTrain, test_size = .2)
    
    imShape = preprocessImage(mpimg.imread(dataTrain['center'].iloc[0])).shape
    print(imShape)
    
    timeLength = 16
    batchSize = 1
    epochBatchSize = 1024
    
    trainGenerator = generateImagesFromPaths(dataTrain, batchSize, timeLength, imShape, [3], transform, angles, images, True)
    t = time.time()
    batch = trainGenerator.__next__()
    for val in batch:
        print(val.shape)
    print("Time to build train batch: ", time.time()-t)
    valGenerator = generateImagesFromPaths(dataVal, batchSize, timeLength, imShape, [3], transform, angles, images)
    t = time.time()
    valGenerator.__next__()
    print("Time to build validation batch: ", time.time()-t)
    model = load_model('initModel.h5', custom_objects={'customLoss':customLoss})
    prevWeights = []
    for layer in model.layers:
        prevWeights.append(layer.get_weights())
    print(len(prevWeights))
    inpC = Input(batch_shape=(batchSize, timeLength, imShape[0], imShape[1], imShape[2]), name='input_1')
    print(inpC.get_shape())
    xC = TimeDistributed(Convolution2D(24, 8, 8,border_mode='valid', subsample=(2,2), weights=prevWeights[1],
                       trainable=False))(inpC)
    xC = TimeDistributed(BatchNormalization())(xC)
    xC = Activation('elu')(xC)
    print(xC.get_shape())
    xC = TimeDistributed(Convolution2D(36, 5, 5, border_mode='valid',subsample=(2,2), weights=prevWeights[4],
                       trainable=False))(xC)
    xC = TimeDistributed(BatchNormalization())(xC)
    xC = Activation('elu')(xC)
    print(xC.get_shape())
    xC = TimeDistributed(Convolution2D(48, 5, 5, border_mode='valid',subsample=(2,2), weights=prevWeights[7],
                       trainable=False))(xC)
    xC = TimeDistributed(BatchNormalization())(xC)
    xC = Activation('elu')(xC)
    print(xC.get_shape())
    xC = TimeDistributed(Convolution2D(64, 5, 5, border_mode='valid', weights=prevWeights[10],
                       trainable=False))(xC)
    xC = TimeDistributed(BatchNormalization())(xC)
    xC = Activation('elu')(xC)
    print(xC.get_shape())
    xC = TimeDistributed(Convolution2D(64, 5, 5, border_mode='valid', weights=prevWeights[13],
                       trainable=False))(xC)
    xC = TimeDistributed(BatchNormalization())(xC)
    xC = Activation('elu')(xC)
    print(xC.get_shape())
    xOut = TimeDistributed(Flatten())(xC)
    del prevWeights

    xOut = LSTM(1024)(xOut)
        
    xOut = Dense(100)(xOut)
    xOut = BatchNormalization()(xOut)
    xOut = Activation('elu')(xOut)
    xOut = Dense(50)(xOut)
    xOut = BatchNormalization()(xOut)
    xOut = Activation('elu')(xOut)
    xOut = Dropout(.3)(xOut)
    xOut = Dense(10)(xOut)
    xOut = BatchNormalization()(xOut)
    xOut = Activation('elu')(xOut)
    xOut = Dense(1, activation='sigmoid')(xOut)
    xOut = Lambda(lambda x: x*2-1, name = 'output')(xOut)
    
    print(xOut.get_shape())
    
    endModel = Model((inpC), xOut)
    endModel.compile(optimizer=Adam(lr=1e-4), loss=customLoss, metrics=['mse', 'accuracy'])
    valErrorComp = 2
    counter = 0
    for i in range(100):
        print("Epoch: ", i)
        trainError = 0
        for j in range(epochBatchSize//2):
            x, y, weight = trainGenerator.__next__()
            trainError += endModel.train_on_batch(x, y, weight)[0]
            if j%256==0: print(j)
        endModel.reset_states()
        for j in range(epochBatchSize//2):
            x, y, weight = trainGenerator.__next__()
            trainError += endModel.train_on_batch(mirrorImage(x), -1.*y, weight)[0]
            if j%256==0: print(j)
        print("Train Error", trainError/epochBatchSize)
        endModel.reset_states()
        valError = 0
        for j in range(len(dataVal)//5):
            x, y, weight = valGenerator.__next__()
            valError += endModel.test_on_batch(x, y, weight)[0]
            if j%256==0: print(j)
        endModel.reset_states()
        for j in range(len(dataVal)//5):
            x, y, weight = valGenerator.__next__()
            valError += endModel.test_on_batch(mirrorImage(x), -1.*y, weight)[0]
            if j%256==0: print(j)
        valError/=len(dataVal)//5
        print("Validation Error", valError)
        endModel.reset_states()
        counter += 1
        if valError < valErrorComp:
            counter = 0
            valErrorComp = valError
            endModel.save_weights('model.ckpt')
        if counter == 6:
            break
    endModel.load_weights('model.ckpt')
    endModel.save('model.h5')
        
    endModel = load_model('model.h5', custom_objects={'customLoss':customLoss})
    print(endModel.evaluate_generator(valGenerator, val_samples=len(dataVal)))
    endModel.reset_states()
    print(endModel.evaluate_generator(generateImagesFromPaths(dataTest, batchSize, timeLength, imShape, [3], transform, angles, images), 
                                      val_samples=len(dataTest)))

if __name__ == '__main__':
    main()
















