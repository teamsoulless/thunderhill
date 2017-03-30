# Import basic
import logging

# from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten,Lambda,ELU
# from keras.layers.merge import Concatenate
import keras
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
import argparse
import os
from loader import generate_thunderhill_batches, getDataFromFolder, genSim001, genSim002, genSession5,genPolysync0,genPolysync2,genAll
from config import *
import time

""" Usefeful link
		ImageDataGenerator 		- https://keras.io/preprocessing/image/
		Saving / Loading model  - http://machinelearningmastery.com/save-load-keras-deep-learning-models/
		NVIDIA					- https://arxiv.org/pdf/1604.07316v1.pdf
		Features Extraction     - https://keras.io/applications/
		ewma					- http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html
		Callbacks				- https://keras.io/callbacks/

		Dropout 5x5
"""

def NvidiaModel(learning_rate, dropout):
    input_model = Input(shape=(HEIGHT, WIDTH, DEPTH))
    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(input_model)
    x = ELU()(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Flatten()(x)
    x = Dense(100)(x)
    x = ELU()(x)
    x = Dropout(dropout)(x)
    x = Dense(50)(x)
    x = ELU()(x)
    x = Dropout(dropout)(x)
    x = Dense(10)(x)
    x = ELU()(x)
    predictions = Dense(1)(x)
    model = Model(input=input_model, output=predictions)
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    return model

def createModel(learning_rate, dropout):
    '''
    Function creates a model of a the network
    '''
    img_input = Input(shape=(HEIGHT, WIDTH, DEPTH))
    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(img_input)
    x = ELU()(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Flatten()(x)


    speed_input = Input(shape=(8,), name='speed_input')
    xx = Dense(50)(speed_input)
    xx = ELU()(xx)
    xx = Dense(100)(speed_input)
    xx = ELU()(xx)
    x = keras.layers.merge([x, xx], mode='concat', concat_axis=-1)


    x = Dense(100)(x)
    x = ELU()(x)
    x = Dropout(dropout)(x)
    x = Dense(50)(x)
    x = ELU()(x)
    x = Dropout(dropout)(x)
    x = Dense(10)(x)
    x = ELU()(x)
    predictions = Dense(3)(x)

    model = Model(input=[img_input,speed_input], output=predictions)
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=EPOCHS, help='Number of epochs.')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=DROPOUT, help='Dropout rate')
    parser.add_argument('--weights', type=str, help='Load weights')
    parser.add_argument('--dataset', type=str, required=True, help='Get dataset here')
    parser.add_argument('--output', type=str, required=True, help='Save model here')
    args = parser.parse_args()

    print('-------------')
    print('BATCH: {}'.format(args.batch))
    print('EPOCH: {}'.format(args.epoch))
    print('ALPA: {}'.format(args.alpha))
    print('DROPOUT: {}'.format(args.dropout))
    print('Load Weights?: {}'.format(args.weights))
    print('Dataset: {}'.format(args.dataset))
    print('Model: {}'.format(args.output))
    print('-------------')

    args.output='logs/'+args.output+'_%d'%int(time.time())

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))

    # model = NvidiaModel(args.alpha, args.dropout)
    model = createModel(args.alpha, args.dropout)

    # Saves the model...
    with open(os.path.join(args.output, 'model.json'), 'w') as f:
        f.write(model.to_json())

    try:
        if args.weights:
            print('Loading weights from file ...')
            model.load_weights(args.weights)
    except IOError:
        print("No model found")


    checkpointer = ModelCheckpoint(os.path.join(args.output, 'weights.{epoch:04d}-{val_loss:.3f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    # early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
    logger = CSVLogger(filename=os.path.join(args.output, 'history.csv'))

    board=TensorBoard(log_dir=args.output, histogram_freq=0, write_graph=True, write_images=True)

    history = model.fit_generator(
        generate_thunderhill_batches(genAll(args.dataset), args.batch),
        nb_epoch=args.epoch,
        samples_per_epoch=50*args.batch,
        validation_data=generate_thunderhill_batches(genSim001(args.dataset), args.batch),
        nb_val_samples=5*args.batch,
        callbacks=[checkpointer, logger, board]#, early_stop]
    )
