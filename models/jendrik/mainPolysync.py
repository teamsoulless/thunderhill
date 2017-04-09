# Load pickled data
import pandas as pd
import tensorflow as tf 
import matplotlib.image as mpimg
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
from PreprocessPsync import *
import numpy as np
from keras.optimizers import Adam
import multiprocessing as mp
from mpl_toolkits.basemap import Basemap
logger = mp.log_to_stderr()
from PIL import Image

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


def generateImagesFromPaths(data, batchSize, inputShape, outputShape, train = False):
    """
        The generator function for the training data for the fit_generator
        Input:
        data - an pandas dataframe containing the paths to the images, the steering angle,...
        batchSize, the number of values, which shall be returned per call
    """
    while 1:
        returnArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
        vecArr = np.zeros((batchSize, 3))
        labels = np.zeros((batchSize, outputShape[0]))
        weights = np.zeros(batchSize)
        indices = np.random.randint(0, len(data), batchSize)
        for i,index in zip(range(len(indices)),indices):
            row = data.iloc[index]
            file = open(row['path'].strip(), 'rb')
            # Use the PIL raw decoder to read the data.
            #   - the 'F;16' informs the raw decoder that we are reading a little endian, unsigned integer 16 bit data.
            img = np.array(Image.frombytes('RGB', [960,480], file.read(), 'raw'))
            file.close()
            image = preprocessImage(img)
            #image = images[int(row['angleIndex'])]
            label = np.array([row['steering'], row['throttle'], row['brake']])
            xVector = row[['longitude', 'latitude', 'speed']].values
            flip = np.random.rand()
            if flip>.5:
                image = mirrorImage(image)
                label[0] *= -1
            if(train):
                image, label = augmentImage(image, label)
            labels[i] = label
            """if flip>.5:
                angleArr[i] = -1.*np.array(angles[int(row['angleIndex']-ANGLESFED):
                                                  int(row['angleIndex'])])
            else:
                angleArr[i] = np.array(angles[int(row['angleIndex']-ANGLESFED):int(row['angleIndex'])])"""
            returnArr[i] = image            
            weights[i] = row['norm']
            if(train):
                vecArr[i] = [val + 0.002*np.random.rand() -0.001 for val in xVector]
            else:
                vecArr[i] = xVector
        yield({'input_1': returnArr,'input_3': np.array(vecArr)},
              {'outputSteer': labels[:,0],'outputThr': labels[:,1]-labels[:,2]}, [weights,weights])
                

def customLoss(y_true, y_pred):
    """
        This loss function adds some constraints on the angle to 
        keep it small, if possible
    """
    return 10*K.mean(K.sqrt(K.square(y_pred - y_true)), axis=-1) #+.01* K.mean(K.square(y_pred), axis = -1)

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
    #data = pd.read_csv('/home/jjordening/data/dataset_sim_000_km_few_laps/driving_log.csv', 
    #                   header = None, names=['center','left', 'right', 'steering','throttle', 'brake', 'speed', 'position', 'orientation'])
    #data['positionX'], data['positionY'], data['positionZ'] = data['position'].apply(retrieveVectors)
    #data['orientationX'], data['orientationY'], data['orientationZ'] = data['orientation'].apply(retrieveVectors)
    #data['center'] = '/home/jjordening/data/dataset_sim_000_km_few_laps/'+data['center'].apply(lambda x: x.strip())
    
    #data1 = pd.read_csv('/home/jjordening/data/udacity-day-01-exported-1102/output_processed.txt')
    #data1['path'] = '/home/jjordening/data/udacity-day-01-exported-1102/'+data1['path'].apply(lambda x: x.strip())
    
    #data2 = pd.read_csv('/home/jjordening/data/udacity-day-01-exported-1109/output_processed.txt')
    #data2['path'] = '/home/jjordening/data/udacity-day-01-exported-1109/'+data2['path'].apply(lambda x: x.strip())

    """data3 = pd.read_csv('/home/jjordening/data/1538/output_processed.txt')
    data3['path'] = '/home/jjordening/data/1538/'+data3['path'].apply(lambda x: x.strip())
    print('data3', np.max(data3['steering']), np.min(data3['steering']))
    
    data4 = pd.read_csv('/home/jjordening/data/1543/output_processed.txt')
    data4['path'] = '/home/jjordening/data/1543/'+data4['path'].apply(lambda x: x.strip())
    print('data4', np.max(data4['steering']), np.min(data4['steering']))
    
    data5 = pd.read_csv('/home/jjordening/data/1610/output_processed.txt')
    data5['path'] = '/home/jjordening/data/1610/'+data5['path'].apply(lambda x: x.strip())
    print('data5', np.max(data5['steering']), np.min(data5['steering']))
    
    data6 = pd.read_csv('/home/jjordening/data/1645/output_processed.txt')
    data6['path'] = '/home/jjordening/data/1645/'+data6['path'].apply(lambda x: x.strip())
    print('data6', np.max(data6['steering']), np.min(data6['steering']))
    
    data7 = pd.read_csv('/home/jjordening/data/1702/output_processed.txt')
    data7['path'] = '/home/jjordening/data/1702/'+data7['path'].apply(lambda x: x.strip())
    print('data7', np.max(data7['steering']), np.min(data7['steering']))"""
    
    data8 = pd.read_csv('/home/jjordening/data/1045/output_processed.txt')
    data8['path'] = '/home/jjordening/data/1045/'+data8['path'].apply(lambda x: x.strip())
    print('data8', np.max(data8['steering']), np.min(data8['steering']))
    
    data9 = pd.read_csv('/home/jjordening/data/1050/output_processed.txt')
    data9['path'] = '/home/jjordening/data/1050/'+data9['path'].apply(lambda x: x.strip())
    print('data9', np.max(data9['steering']), np.min(data9['steering']))
    
    data10 = pd.read_csv('/home/jjordening/data/1426/outputNew.txt')
    data10['path'] = '/home/jjordening/data/1426/'+data10['path'].apply(lambda x: x.strip())
    print('data9', np.max(data10['steering']), np.min(data10['steering']))
    
    data11 = pd.read_csv('/home/jjordening/data/1516/outputNew.txt')
    data11['path'] = '/home/jjordening/data/1516/'+data11['path'].apply(lambda x: x.strip())
    print('data9', np.max(data11['steering']), np.min(data11['steering']))
    
    print(data9['brake'].unique())
       
    """data3 = pd.read_csv('/home/jjordening/data/dataset_polysync_1464552951979919/output_processed.txt', header = None, 
                        names = ['path','heading','longitude','latitude','quarternion0','quarternion1','quarternion2','quarternion3','vel0','vel1',
                                'vel2','steering','throttle','brake','speed'], skiprows = 500)
    data3 = data3.ix[0:1500].append(data3.ix[2600:])
    data3 = data3.ix[-500:]
    data3['path'] = '/home/jjordening/data/dataset_polysync_1464552951979919/'+data3['path'].apply(lambda x: x.strip())
    data3['throttle'] = 0"""
    
    #data['right'] = '../simulator/data/data/'+data['right'].apply(lambda x: x.strip())
    #data['left'] = '../simulator/data/data/'+data['left'].apply(lambda x: x.strip())
    angles = []
    dataNew = pd.DataFrame()
    offset = 0
    #print(data3['steering'])
    #print(data1['longitude'])
    """for dat in [data3,data4,data5,data6,data7]:
        angles.extend(dat['steering'].values)
        for row in dat.iterrows():
            dat.loc[row[0], 'angleIndex'] = row[0]+ offset
            #images.append(preprocessImage(mpimg.imread(row[1]['center'].strip())))
            #images.append(transform(mpimg.imread(row[1]['center'].strip())))
        offset+=100
        dataNew = dataNew.append(dat.ix[100:])"""
    #dataNew['throttle'] = dataNew['accel'].apply(lambda x: max(x,0)/np.max(dataNew['accel']))
    for dat in [data8,data9,data10, data11]:
        angles.extend(dat['steering'].values)
        for row in dat.iterrows():
            dat.loc[row[0], 'angleIndex'] = row[0]+ offset
            #images.append(preprocessImage(mpimg.imread(row[1]['center'].strip())))
            #images.append(transform(mpimg.imread(row[1]['center'].strip())))
        offset+=100
        dataNew = dataNew.append(dat.ix[100:])
    dataNew = dataNew.loc[pd.notnull(dataNew['throttle'])]
    dataNew = dataNew.loc[pd.notnull(dataNew['brake'])]
    dataNew = dataNew.loc[pd.notnull(dataNew['steering'])]
    print(np.max(dataNew['throttle']), np.min(dataNew['throttle']))
    # TODO: Normalisation of position and orientation<
    #del data3,data4,data5,data6,data7
    print(len(dataNew), dataNew.columns)
    print(np.histogram(dataNew['throttle'], bins = 31))
    hist, edges = np.histogram(dataNew['steering'], bins = 31)
    print(hist,edges, len(dataNew))
    hist = 1./np.array([val if val > len(dataNew)/30. else len(dataNew)/30. for val in hist])
    hist*=len(dataNew)/30.
    print(hist,edges, len(dataNew))
    dataNew['norm'] = dataNew['steering'].apply(lambda x: getNormFactor(x, hist, edges))
    print(dataNew['norm'].unique())
    print(np.min(dataNew['steering']), np.max(dataNew['steering']))
    print(np.min(dataNew['throttle']), np.max(dataNew['throttle']))
    print(np.min(dataNew['brake']), np.max(dataNew['brake']))
    
    for col in ['longitude', 'latitude']:
        vals = dataNew[col].values
        mean = np.mean(vals)
        std = np.std(vals)
        dataNew[col] -= mean
        dataNew[col] /= std
        print('%s Mean:%.12f Std:%.12f' %(col, mean, std))
        
    dataNew['speed'] = dataNew['speed'].apply(lambda x: x/40. - 1)
    
    dataNew = shuffle(dataNew, random_state=0)
    #plt.figure(1, figsize=(8,4))
    #plt.hist(dataNew['steering'], bins =31)
    
    #plt.show()
    
    dataTrain, dataTest= train_test_split(dataNew, test_size = .1)
    dataTrain, dataVal= train_test_split(dataTrain, test_size = .1)
    
    file = open(dataTrain['path'].iloc[0], 'rb')
    # Use the PIL raw decoder to read the data.
    #   - the 'F;16' informs the raw decoder that we are reading a little endian, unsigned integer 16 bit data.
    img = np.array(Image.frombytes('RGB', [960,480], file.read(), 'raw'))
    file.close()
    
    imShape = preprocessImage(img).shape
    print(imShape)
    
    
    batchSize = 128
    epochBatchSize = 8192
    trainGenerator = generateImagesFromPaths(dataTrain, batchSize, imShape, [3], angles, True)
    t = time.time()
    trainGenerator.__next__()
    print("Time to build train batch: ", time.time()-t)
    valGenerator = generateImagesFromPaths(dataVal, batchSize, imShape, [3], angles) 
    t = time.time()
    valGenerator.__next__()
    print("Time to build validation batch: ", time.time()-t)
    stopCallback = EarlyStopping(monitor='val_loss', patience = 10, min_delta = 0.05)
    checkCallback = ModelCheckpoint('psyncModel.ckpt', monitor='val_loss', save_best_only=True)
    visCallback = TensorBoard(log_dir = './logs')
    if LOADMODEL:
        endModel = load_model('psyncModelBase.h5', custom_objects={'customLoss':customLoss})
        endModel.fit_generator(trainGenerator, callbacks=[stopCallback, checkCallback, visCallback], nb_epoch=100, samples_per_epoch=epochBatchSize,
                               max_q_size=8, validation_data = valGenerator, nb_val_samples=len(dataVal),
                               nb_worker=8, pickle_safe=True)
        endModel.load_weights('psyncModel.ckpt')
        endModel.save('psyncModel.h5')
        
        
    else:
        inpC = Input(shape=(imShape[0], imShape[1], imShape[2]), name='inputImg')
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
        
        xVectorInp = Input(shape = (3,), name='input_3')
        xVector = Dropout(.1)(xVectorInp)
        
        
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
        xOutSteer = Lambda(lambda x: x*10-5, name = 'outputSteer')(xOutSteer)
        
        xOutThr = Dense(50, name='thr1')(xEnd)
        xOutThr = BatchNormalization(name='thr2')(xOutThr)
        xOutThr = Activation('elu')(xOutThr)
        xOutThr = Dropout(.3)(xOutThr)
        xOutThr = Dense(10, name='thr3')(xOutThr)
        xOutThr = BatchNormalization(name='thr4')(xOutThr)
        xOutThr = Activation('elu')(xOutThr)
        xOutThr = Dense(1, activation='sigmoid',name='thr5')(xOutThr)
        xOutThr = Lambda(lambda x: x*2-1, name = 'outputThr')(xOutThr)
        
        endModel = Model((inpC, xVectorInp), (xOutSteer, xOutThr))
        endModel.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
        #endModel.fit_generator(trainGenerator, callbacks = [visCallback], 
        #                       nb_epoch=50, samples_per_epoch=epochBatchSize, 
        #                       max_q_size=24, nb_worker=8, pickle_safe=True)
        endModel.fit_generator(trainGenerator, 
                               nb_epoch=30, samples_per_epoch=epochBatchSize, 
                               max_q_size=24, nb_worker=8, pickle_safe=True)
        endModel.save('psyncModelBase.h5')
        endModel.fit_generator(trainGenerator, callbacks = [stopCallback, checkCallback,visCallback], 
                               nb_epoch=100, samples_per_epoch=epochBatchSize, 
                               max_q_size=24, validation_data = valGenerator, 
                               nb_val_samples=len(dataVal),
                               nb_worker=8, pickle_safe=True)
        endModel.load_weights('psyncModel.ckpt')
        endModel.save('psyncModel.h5')
        endModel.save('psyncModelBase.h5')
        
    endModel = load_model('psyncModelBase.h5', custom_objects={'customLoss':customLoss})
    
    print(endModel.evaluate_generator(valGenerator, val_samples=len(dataVal)))
    print(endModel.evaluate_generator(generateImagesFromPaths(dataTest, batchSize, imShape, [3], angles), 
                                      val_samples=len(dataTest)))

if __name__ == '__main__':
    main()
















