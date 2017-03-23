# Load pickled data
import pandas as pd
import tensorflow as tf 
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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
from mpl_toolkits.basemap import Basemap

logger = mp.log_to_stderr()

LOADMODEL = False
ANGLESFED = 5


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
#flags.DEFINE_string('trainingCSV', '../simulator/simulator-linux/driving_log.csv', "training data")
#flags.DEFINE_string('trainingCSV', '../simulator/data/data/driving_log.csv', "training data")

start = (39.53745, -122.33879)


mt = Basemap(llcrnrlon=-122.341041,llcrnrlat=39.532678,urcrnrlon=-122.337929,urcrnrlat=39.541455,
    projection='merc',lon_0=start[1],lat_0=start[0],resolution='h')

diffx = -989.70
diffy = -58.984

def toGPS(simx, simy):
    projx, projy = simx+diffx, simy + diffy
    lon, lat = mt(projx, projy,inverse=True)
    return np.array([lon, lat])


def generateImagesFromPaths(data, batchSize, inputShape, outputShape, angles, train = False):
    """
        The generator function for the training data for the fit_generator
        Input:
        data - an pandas dataframe containing the paths to the images, the steering angle,...
        batchSize, the number of values, which shall be returned per call
    """
    while 1:
        returnArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
        angleArr = np.zeros((batchSize, ANGLESFED))
        vecArr = np.zeros((batchSize, 2))
        labels = np.zeros((batchSize, outputShape[0]))
        weights = np.zeros(batchSize)
        indices = np.random.randint(0, len(data), batchSize)
        for i,index in zip(range(len(indices)),indices):
            row = data.iloc[index]
            image = preprocessImage(mpimg.imread(row['center'].strip()))
            #image = images[int(row['angleIndex'])]
            label = np.array([row['steering'], row['throttle'], row['brake']])
            xVector = row[['long', 'lat']].values
            #if(image.shape[0] != 320): image = cv2.resize(image, (320, 160))
            flip = np.random.rand()
            if flip>.5:
                image = mirrorImage(image)
                label[0] *= -1
            if(train):
                image, label = augmentImage(image, label)
            labels[i] = label
            if flip>.5:
                angleArr[i] = -1.*np.array(angles[int(row['angleIndex']-ANGLESFED):
                                                  int(row['angleIndex'])])
            else:
                angleArr[i] = np.array(angles[int(row['angleIndex']-ANGLESFED):int(row['angleIndex'])])
            returnArr[i] = image            
            weights[i] = row['norm']
            if(train):
                vecArr[i] = [val + 0.002*np.random.rand() -0.001 for val in xVector]
            else:
                vecArr[i] = xVector
        yield({'input_1': returnArr,'input_2': angleArr,'input_3': np.array(vecArr)},
              {'outputSteer': labels[:,0],'outputThr': labels[:,1],'outputBre': labels[:,2]}, [weights,weights,weights])
                

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

def retrieveGPS(vecString):
    split = vecString.split(":")
    return pd.Series(toGPS(float(split[0]), float(split[1])))
        

def main():
    img = mpimg.imread('/home/jjordening/git/thunderhill_data/dataset_sim_001_km_320x160/IMG/center_2017_03_07_07_21_54_311.jpg')
    h, w = img.shape[:2]
    src = np.float32([[w/2 - 57, h/2], [w/2 + 57, h/2], [w+140,h], [-140,h]])
    dst = np.float32([[w/4,0], [w*3/4,0], [w*3/4,h], [w/4,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    transform = functools.partial(perspectiveTransform, M = M.copy())
    #plt.imshow(preprocessImage(img, transform))
    #plt.show()
    
    #showSamplesCompared(img, transform, '', '', '')
    #plt.xkcd()
    np.random.seed(0)
    #data = pd.read_csv('/home/jjordening/git/thunderhill_data/dataset_sim_000_km_few_laps/driving_log.csv', 
    #                   header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    #data['positionX'], data['positionY'], data['positionZ'] = data['position'].apply(retrieveVectors)
    #data['orientationX'], data['orientationY'], data['orientationZ'] = data['orientation'].apply(retrieveVectors)
    #data['center'] = '/home/jjordening/git/thunderhill_data/dataset_sim_000_km_few_laps/'+data['center'].apply(lambda x: x.strip())
    data1 = pd.read_csv('/home/jjordening/git/thunderhill_data/dataset_sim_001_km_320x160/driving_log.csv', 
                       header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    data1['center'] = '/home/jjordening/git/thunderhill_data/dataset_sim_001_km_320x160/'+data1['center'].apply(lambda x: x.strip())
    data1[['long','lat']] = data1['position'].apply(retrieveGPS)
    
    data2 = pd.read_csv('/home/jjordening/git/thunderhill_data/dataset_sim_002_km_320x160_recovery/driving_log.csv', 
                       header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    data2['center'] = '/home/jjordening/git/thunderhill_data/dataset_sim_002_km_320x160_recovery/'+data2['center'].apply(lambda x: x.strip())
    data2[['long','lat']] = data2['position'].apply(retrieveGPS)
    
    data3 = pd.read_csv('/home/jjordening/git/thunderhill_data/dataset_sim_003_km_320x160/driving_log.csv', 
                       header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    data3['center'] = '/home/jjordening/git/thunderhill_data/dataset_sim_003_km_320x160/'+data3['center'].apply(lambda x: x.strip())
    data3[['long','lat']] = data3['position'].apply(retrieveGPS)
    
    data4 = pd.read_csv('/home/jjordening/git/thunderhill_data/dataset_sim_004_km_320x160_cones_brakes/driving_log.csv', 
                       header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    data4['center'] = '/home/jjordening/git/thunderhill_data/dataset_sim_004_km_320x160_cones_brakes/'+data4['center'].apply(lambda x: x.strip())
    data4[['long','lat']] = data4['position'].apply(retrieveGPS)
    #data['right'] = '../simulator/data/data/'+data['right'].apply(lambda x: x.strip())
    #data['left'] = '../simulator/data/data/'+data['left'].apply(lambda x: x.strip())
    angles = []
    dataNew = pd.DataFrame()
    offset = 0
    
    print(data1['long'])
    for dat in [data1, data2, data3, data4]:
        angles.extend(dat['steering'].values)
        for row in dat.iterrows():
            dat.loc[row[0], 'angleIndex'] = row[0]+ offset
            #images.append(preprocessImage(mpimg.imread(row[1]['center'].strip())))
            #images.append(transform(mpimg.imread(row[1]['center'].strip())))
        offset+=100
        dataNew = dataNew.append(dat.ix[100:])
    # TODO: Normalisation of position and orientation
    print(len(dataNew), dataNew.columns)
    hist, edges = np.histogram(dataNew['steering'], bins = 31)
    hist = 1./np.array([val if val > len(dataNew)/30. else len(dataNew)/30. for val in hist])
    hist*=len(dataNew)/30.
    print(hist, len(dataNew))
    dataNew['norm'] = dataNew['steering'].apply(lambda x: getNormFactor(x, hist, edges))
    print(dataNew['norm'].unique())
    del data1, data2,data3, data4
    
    for col in ['long', 'lat']:
        vals = dataNew[col].values
        mean = np.mean(vals)
        std = np.std(vals)
        dataNew[col] -= mean
        dataNew[col] /= std
        print('%s Mean:%.8f Std:%.8f' %(col, mean, std))
    
    dataNew = shuffle(dataNew, random_state=0)
    #plt.figure(1, figsize=(8,4))
    #plt.hist(dataNew['steering'], bins =31)
    
    #plt.show()
    
    dataTrain, dataTest= train_test_split(dataNew, test_size = .2)
    dataTrain, dataVal= train_test_split(dataTrain, test_size = .2)
    
    imShape = preprocessImage(mpimg.imread(dataTrain['center'].iloc[0])).shape
    print(imShape)
    
    
    batchSize = 128
    epochBatchSize = 4096
    trainGenerator = generateImagesFromPaths(dataTrain, batchSize, imShape, [3], angles, True)
    t = time.time()
    trainGenerator.__next__()
    print("Time to build train batch: ", time.time()-t)
    valGenerator = generateImagesFromPaths(dataVal, batchSize, imShape, [3], angles)
    t = time.time()
    valGenerator.__next__()
    print("Time to build validation batch: ", time.time()-t)
    stopCallback = EarlyStopping(monitor='val_loss', patience = 20, min_delta = 0.)
    checkCallback = ModelCheckpoint('multiModel.ckpt', monitor='val_loss', save_best_only=True)
    visCallback = TensorBoard(log_dir = './logs')
    if LOADMODEL:
        endModel = load_model('multiModel.h5', custom_objects={'customLoss':customLoss})
        inp = valGenerator.__next__()
        print(inp)
        print(inp[0])
        vals = endModel.predict([inp[0]['input_1'][0][None,:,:,:],np.reshape(inp[0]['input_2'][0],[1,5])])
        print(vals)
        endModel.fit_generator(trainGenerator, callbacks=[stopCallback, checkCallback, visCallback], nb_epoch=20, samples_per_epoch=epochBatchSize,
                               max_q_size=8, validation_data = valGenerator, nb_val_samples=len(dataVal),
                               nb_worker=8, pickle_safe=True)
        endModel.load_weights('multiModel.ckpt')
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
        xC = Convolution2D(64, 5, 5, border_mode='valid')(xC)
        xC = BatchNormalization()(xC)
        xC = Activation('elu')(xC)
        print(xC.get_shape())
        xC = Convolution2D(64, 5, 5, border_mode='valid')(xC)
        xC = BatchNormalization()(xC)
        xC = Activation('elu')(xC)
        print(xC.get_shape())
        xOut = Flatten()(xC)
        
        xVectorInp = Input(shape = (2,), name='input_3')
        xVector = Dropout(.1)(xVectorInp)
        xVector = Dense(50)(xVectorInp)
        xVector = BatchNormalization()(xVector)
        xVector = Activation('elu')(xVector)
        xVector = Dense(50)(xVector)
        xVector = BatchNormalization()(xVector)
        xVector = Activation('elu')(xVector)
        
        
        #inpAngles = Input(shape=(ANGLESFED,), name='input_2')
        
        xOut = Lambda(lambda x : K.concatenate(x, axis=1))([xOut, xVector])
        xOut = Dense(200)(xOut)
        xOut = BatchNormalization()(xOut)
        xOut = Activation('elu')(xOut)
        xOut = Dense(100)(xOut)
        xOut = BatchNormalization()(xOut)
        xEnd = Activation('elu')(xOut)
        
        
        xOutSteer = Dense(50)(xEnd)
        xOutSteer = BatchNormalization()(xOutSteer)
        xOutSteer = Activation('elu')(xOutSteer)
        xOutSteer = Dropout(.3)(xOutSteer)
        xOutSteer = Dense(10)(xOutSteer)
        xOutSteer = BatchNormalization()(xOutSteer)
        xOutSteer = Activation('elu')(xOutSteer)
        xOutSteer = Dense(1, activation='sigmoid')(xOutSteer)
        xOutSteer = Lambda(lambda x: x*2-1, name = 'outputSteer')(xOutSteer)
        
        xOutThr = Dense(50)(xEnd)
        xOutThr = BatchNormalization()(xOutThr)
        xOutThr = Activation('elu')(xOutThr)
        xOutThr = Dropout(.3)(xOutThr)
        xOutThr = Dense(10)(xOutThr)
        xOutThr = BatchNormalization()(xOutThr)
        xOutThr = Activation('elu')(xOutThr)
        xOutThr = Dense(1, activation='sigmoid')(xOutThr)
        xOutThr = Lambda(lambda x: x*2-1, name = 'outputThr')(xOutThr)
        
        xOutBre = Dense(50)(xEnd)
        xOutBre = BatchNormalization()(xOutBre)
        xOutBre = Activation('elu')(xOutBre)
        xOutBre = Dropout(.3)(xOutBre)
        xOutBre = Dense(10)(xOutBre)
        xOutBre = BatchNormalization()(xOutBre)
        xOutBre = Activation('elu')(xOutBre)
        xOutBre = Dense(1, activation='sigmoid')(xOutBre)
        xOutBre = Lambda(lambda x: x*2-1, name = 'outputBre')(xOutBre)
        #xRec = LSTM(10)(xOut)
        
        endModel = Model((inpC, xVectorInp), (xOutSteer, xOutThr, xOutBre))
        endModel.compile(optimizer=Adam(lr=1e-4), loss=customLoss, metrics=['mse', 'accuracy'])
        endModel.fit_generator(trainGenerator, callbacks = [visCallback], 
                               nb_epoch=100, samples_per_epoch=epochBatchSize, 
                               max_q_size=8, nb_worker=8, pickle_safe=True)
        endModel.fit_generator(trainGenerator, callbacks = [stopCallback, checkCallback,visCallback], 
                               nb_epoch=300, samples_per_epoch=epochBatchSize, 
                               max_q_size=8, validation_data = valGenerator, 
                               nb_val_samples=len(dataVal),
                               nb_worker=8, pickle_safe=True)
        endModel.load_weights('multiModel.ckpt')
        endModel.save('multiModel.h5')
        
    endModel = load_model('multiModel.h5', custom_objects={'customLoss':customLoss})
    
    print(endModel.evaluate_generator(valGenerator, val_samples=len(dataVal)))
    print(endModel.evaluate_generator(generateImagesFromPaths(dataTest, batchSize, imShape, [3], angles), 
                                      val_samples=len(dataTest)))

if __name__ == '__main__':
    main()
















