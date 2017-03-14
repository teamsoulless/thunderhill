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

logger = mp.log_to_stderr()

LOADMODEL = False
ANGLESFED = 10


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
#flags.DEFINE_string('trainingCSV', '../simulator/simulator-linux/driving_log.csv', "training data")
#flags.DEFINE_string('trainingCSV', '../simulator/data/data/driving_log.csv', "training data")

def produceExamples(dataShuffled, dataNew):
    plt.figure(1, figsize=(16,9))
    plt.hist(dataShuffled['steering'], bins=np.arange(-1.2, 1.3, .1), log=True)
    plt.xlabel('steering angle')
    plt.ylabel('log(# of occurence)')
    plt.savefig('../distributionBeforeFiltering.png')
    indices = np.random.randint(0,len(dataNew),20)
    fig, ax = plt.subplots(nrows=5, ncols=4, figsize=((16,9)))
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
    for i, index in zip(range(20), indices):
        ax[i//4, i%4].set_title('angle: %.3f' % dataNew.iloc[index]['steering'])
        ax[i//4, i%4].imshow(mpimg.imread(dataNew.iloc[index]['center']))
    plt.savefig('../samplesImages.png')
    dataNew = shuffle(dataNew, random_state = 0)
    plt.figure(3, figsize=(16,9))
    plt.hist(dataNew['steering'], bins=np.arange(-1.2, 1.3, .1), log=True)
    plt.xlabel('steering angle')
    plt.ylabel('log(# of occurence)')
    plt.savefig('../distributionAfterFiltering.png')
    
    index = np.random.randint(0,len(dataNew))
    plt.figure(4, figsize=(8,4))
    plt.title('Original, Steering: %.3f' % dataNew.iloc[index]['steering'])
    plt.imshow(mpimg.imread(dataNew.iloc[index]['center']))
    plt.savefig('../image.png')
    plt.figure(5, figsize=(8,4))
    steer = dataNew.iloc[index]['steering']
    plt.title('Mirrored, Steering: %.3f' % (-1.*steer))
    plt.imshow(mirrorImage(mpimg.imread(dataNew.iloc[index]['center'])))
    plt.savefig('../flippedImage.png')
    
    plt.figure(6, figsize=(8,4))
    image = mirrorImage(mpimg.imread(dataNew.iloc[index]['center']))
    steer = dataNew.iloc[index]['steering']
    shiftHor = np.random.randint(-20,21)
    shiftVer = np.random.randint(-10,11)
    steer *= (1-shiftVer/100)
    steer += .1*shiftHor/(20)
    image = shiftImg(image, shiftHor, shiftVer)
    rot = np.random.randint(-10,11)
    steer += .5*rot/(25)
    steer = min(max(steer,-1),1)
    image = rotateImage(image, rot)
    plt.title('Augmented, Steering: %.3f' % steer)
    plt.imshow(image)
    plt.savefig('../augmentedImage.png')
    plt.show()


def generateTrainImagesFromPaths(data, batchSize, inputShape, outputShape, transform, angles, images):
    """
        The generator function for the training data for the fit_generator
        Input:
        data - an pandas dataframe containing the paths to the images, the steering angle,...
        batchSize, the number of values, which shall be returned per call
    """
    while 1:
        returnArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
        angleArr = np.zeros((batchSize, ANGLESFED))
        vecArr = np.zeros((batchSize, 6))
        labels = np.zeros((batchSize, outputShape[0]))
        weights = np.zeros(batchSize)
        indices = np.random.randint(0, len(data), batchSize)
        for i,index in zip(range(len(indices)),indices):
            row = data.iloc[index]
            imSelect = .5#np.random.random()
            if(imSelect <.1):
                image = np.array(mpimg.imread(row['right'].strip()))
                label = np.array([min(row['steering']-.15,-1), row['throttle'], row['brake']])
            elif(imSelect >.9):
                image = np.array(mpimg.imread(row['left'].strip()))
                label = np.array([min(row['steering']+.15,1), row['throttle'], row['brake']])
            else:
                image = images[int(row['angleIndex'])]
                label = np.array([row['steering'], row['throttle'], row['brake']])
            xVector = row[['positionX', 'positionY', 'positionZ', 'orientationX', 'orientationY', 'orientationZ']].values
            if(image.shape[0] != 320): image = cv2.resize(image, (320, 160))
            flip = np.random.random()
            if flip>.5:
                image = mirrorImage(image)
                label[0] *= -1
            
            
            rot = np.random.randint(-10,11)
            image = rotateImage(image, rot)
            # Add a part of the rotated angle, as it is counted counter-clockwise.
            # If you turn counter-clockwise, this looks like the car would be more left
            # and needs to drive to the right -> add some angle 
            # divide it by the maximum of the steering angle in deg ->25
            label[0] += .2*rot/(10)
            
            shiftHor = np.random.randint(-20,21)
            shiftVer = np.random.randint(-10,11)
            image = shiftImg(image, shiftHor, shiftVer)
            label[0] *= (1-shiftVer/100)
            label[0] += .2*shiftHor/(20)
            
            label[0] = min(max(label[0],-1),1)
            returnArr[i] = preprocessImage(image, transform)
            labels[i] = label
            if flip>.5:
                angleArr[i] = -1.*np.array(angles[int(row['angleIndex']-ANGLESFED):
                                                  int(row['angleIndex'])])
            else:
                angleArr[i] = np.array(angles[int(row['angleIndex']-ANGLESFED):int(row['angleIndex'])])
            weights[i] = row['norm']
            xVector = np.array([val + np.random.rand()*0.02 - 0.01 for val in xVector])
            vecArr[i] = xVector
        yield({'input_1': returnArr, 'input_2': angleArr, 'input_3': vecArr},
              {'output': labels[:,0]}, weights)
        del returnArr, angleArr, vecArr, labels, weights
                
def generateTestImagesFromPaths(data, batchSize, inputShape, outputShape, transform, angles, images):
    """
        The generator function for the validation and test data for the fit_generator
        Input:
        data - an pandas dataframe containing the paths to the images, the steering angle,...
        batchSize, the number of values, which shall be returned per call
    """
    size=0
    while 1:
        returnArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
        angleArr = np.zeros((batchSize, ANGLESFED))
        vecArr = np.zeros((batchSize, 6))
        labels = np.zeros((batchSize, outputShape[0]))
        weights = np.zeros(batchSize)
        row = data.iloc[size%len(data)]
        image = images[int(row['angleIndex'])]
        label = np.array([row['steering'], row['throttle'], row['brake']])
        xVector = row[['positionX', 'positionY', 'positionZ', 'orientationX', 'orientationY', 'orientationZ']].values
        xVector = [val + .01*np.random.randn() for val in xVector]
        if(image.shape[0] != 320): image = cv2.resize(image, (320, 160))
        image = preprocessImage(image, transform)
        flip = np.random.random()
        if flip>.5:
            image = mirrorImage(image)
            label[0] *= -1
        returnArr[size%batchSize] = image
        labels[size%batchSize] = label
        if flip>.5:
            angleArr[size%batchSize] = -1.*np.array(angles[int(row['angleIndex']-ANGLESFED):int(row['angleIndex'])])
        else:
            angleArr[size%batchSize] = np.array(angles[int(row['angleIndex']-ANGLESFED):int(row['angleIndex'])])
        weights[size%batchSize] = row['norm']
        vecArr[size%batchSize] = xVector 
        if(size%batchSize==0):
            yield({'input_1': returnArr ,'input_2': angleArr, 'input_3': vecArr},
                  {'output': labels[:,0]}, weights)
        del returnArr, angleArr, vecArr, labels, weights
            
    
        size+=1

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
    img = mpimg.imread('../simulator/data/data/IMG/center_2016_12_01_13_30_48_287.jpg')
    h, w = img.shape[:2]
    src = np.float32([[w/2 - 57, h/2], [w/2 + 57, h/2], [w+140,h], [-140,h]])
    dst = np.float32([[w/4,0], [w*3/4,0], [w*3/4,h], [w/4,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    transform = functools.partial(perspectiveTransform, M = M.copy())
    #plt.imshow(addGradientLayer(img, 7, (100, 255))[:,:,3])
    #plt.show()
    
    #showSamplesCompared(img, transform, '', '', '')
    plt.xkcd()
    np.random.seed(0)
    #data = pd.read_csv('/home/jjordening/git/thunderhill_data/dataset_sim_000_km_few_laps/driving_log.csv', 
    #                   header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    #data['positionX'], data['positionY'], data['positionZ'] = data['position'].apply(retrieveVectors)
    #data['orientationX'], data['orientationY'], data['orientationZ'] = data['orientation'].apply(retrieveVectors)
    #data['center'] = '/home/jjordening/git/thunderhill_data/dataset_sim_000_km_few_laps/'+data['center'].apply(lambda x: x.strip())
    data1 = pd.read_csv('/home/jjordening/git/thunderhill_data/dataset_sim_001_km_320x160/driving_log.csv', 
                       header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    data1['center'] = '/home/jjordening/git/thunderhill_data/dataset_sim_001_km_320x160/'+data1['center'].apply(lambda x: x.strip())
    data1[['positionX','positionY','positionZ']] = data1['position'].apply(retrieveVectors)
    data1[['orientationX','orientationY','orientationZ']] = data1['orientation'].apply(retrieveVectors)
    data2 = pd.read_csv('/home/jjordening/git/thunderhill_data/dataset_sim_002_km_320x160_recovery/driving_log.csv', 
                       header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    data2['center'] = '/home/jjordening/git/thunderhill_data/dataset_sim_002_km_320x160_recovery/'+data2['center'].apply(lambda x: x.strip())
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
            images.append(mpimg.imread(row[1]['center'].strip()))
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
    
    imShape = preprocessImage(mpimg.imread(dataTrain['center'].iloc[0]), transform).shape
    print(imShape)
    
    
    batchSize = 64
    epochBatchSize = 4096
    
    trainGenerator = generateTrainImagesFromPaths(dataTrain, batchSize, imShape, [3], transform, angles, images)
    valGenerator = generateTestImagesFromPaths(dataVal, batchSize, imShape, [3], transform, angles, images)
    stopCallback = EarlyStopping(monitor='val_loss', patience = 10, min_delta = 0.)
    checkCallback = ModelCheckpoint('model.ckpt', monitor='val_loss', save_best_only=True)
    visCallback = TensorBoard(log_dir = './logs')
    if LOADMODEL:
        endModel = load_model('initModel.h5', custom_objects={'customLoss':customLoss})
        endModel.fit_generator(trainGenerator, callbacks=[stopCallback, checkCallback, visCallback], nb_epoch=20, samples_per_epoch=epochBatchSize,
                               max_q_size=8, validation_data = valGenerator, nb_val_samples=len(dataVal),
                               nb_worker=8, pickle_safe=True)
        endModel.load_weights('model.ckpt')
        endModel.save('model.h5')
        
        
    else:
        inpC = Input(shape=(imShape[0], imShape[1], imShape[2]), name='input_1')
        xC = Convolution2D(24, 8, 8,border_mode='valid', subsample=(2,2))(inpC)
        xC = BatchNormalization()(xC)
        xC = Activation('elu')(xC)
        print(xC.get_shape())
        xC = Convolution2D(36, 5, 5, border_mode='valid',subsample=(2,2))(xC)
        xC = BatchNormalization()(xC)
        xC = Activation('elu')(xC)
        print(xC.get_shape())
        xC = Convolution2D(48, 5, 5, border_mode='valid',subsample=(2,2))(xC)
        xC = BatchNormalization()(xC)
        xC = Activation('elu')(xC)
        print(xC.get_shape())
        xC = Convolution2D(64, 3, 3, border_mode='valid')(xC)
        xC = BatchNormalization()(xC)
        xC = Activation('elu')(xC)
        print(xC.get_shape())
        xC = Convolution2D(64, 3, 3, border_mode='valid')(xC)
        xC = BatchNormalization()(xC)
        xC = Activation('elu')(xC)
        print(xC.get_shape())
        xC = Convolution2D(64, 3, 3, border_mode='valid')(xC)
        xC = BatchNormalization()(xC)
        xC = Activation('elu')(xC)
        print(xC.get_shape())
        xOut = Flatten()(xC)
        
        xVectorInp = Input(shape = (6,), name='input_3')
        xVector = Dense(100)(xVectorInp)
        xVector = BatchNormalization()(xVector)
        xVector = Activation('elu')(xVector)
        xVector = Dense(100)(xVector)
        xVector = BatchNormalization()(xVector)
        xVector = Activation('elu')(xVector)
        xVector = Dense(100)(xVector)
        xVector = BatchNormalization()(xVector)
        xVector = Activation('elu')(xVector)
        xVector = Dense(100)(xVector)
        xVector = BatchNormalization()(xVector)
        xVector = Activation('elu')(xVector)
        
        
        inpAngles = Input(shape=(ANGLESFED,), name='input_2')
        
        xOut = Lambda(lambda x : K.concatenate(x, axis=1))([xOut, inpAngles, xVector])
        xOut = Dense(1164)(xOut)
        xOut = BatchNormalization()(xOut)
        xOut = Activation('elu')(xOut)
        xOut = Dense(100)(xOut)
        xOut = BatchNormalization()(xOut)
        xOut = Activation('elu')(xOut)
        xOut = Dropout(.4)(xOut)
        xOut = Dense(50)(xOut)
        xOut = BatchNormalization()(xOut)
        xOut = Activation('elu')(xOut)
        xOut = Dropout(.4)(xOut)
        xOut = Dense(10)(xOut)
        xOut = BatchNormalization()(xOut)
        xOut = Activation('elu')(xOut)
        xOut = Dense(1, name = 'output')(xOut)
        
        endModel = Model((inpC, inpAngles, xVectorInp), xOut)
        endModel.compile(optimizer=Adam(lr=1e-4), loss=customLoss, metrics=['mse', 'accuracy'])
        endModel.fit_generator(trainGenerator, callbacks = [stopCallback, checkCallback,visCallback], 
                               nb_epoch=100, samples_per_epoch=epochBatchSize, 
                               max_q_size=8, validation_data = valGenerator, 
                               nb_val_samples=len(dataVal),
                               nb_worker=8, pickle_safe=True)
        endModel.load_weights('model.ckpt')
        endModel.save('initModel.h5')
        endModel.save('model.h5')
        
    endModel = load_model('model.h5', custom_objects={'customLoss':customLoss})
    print(endModel.evaluate_generator(valGenerator, val_samples=len(dataVal)))
    print(endModel.evaluate_generator(generateTestImagesFromPaths(dataTest, batchSize, imShape, [3], transform, angles, images), val_samples=len(dataTest)))

if __name__ == '__main__':
    main()
















