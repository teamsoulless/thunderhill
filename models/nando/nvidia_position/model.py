# Import basic

# from keras.models import Sequential
import argparse

from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from keras.layers import Input, Dense, Flatten, ELU, merge
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from loader import generate_thunderhill_batches, getDataFromThunderhill
from config import *

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.3
set_session(tf.Session(config=config))

def NvidiaModel(dropout):

    # Images
    image = Input(shape=(HEIGHT, WIDTH, DEPTH), name='img')
    speed = Input(shape=(1,), name='speed')

    x = BatchNormalization()(image)
    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu')(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2), activation='elu')(x)
    x = BatchNormalization(axis=1)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2), activation='elu')(x)
    x = Flatten()(x)

    # Speed
    x = merge([x, speed], mode='concat', concat_axis=-1, name="merged_layer")
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(50, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(10, activation='elu')(x)
    x = BatchNormalization()(x)

    prediction = Dense(2)(x)

    model = Model(input=[image, speed], output=prediction)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=EPOCHS, help='Number of epochs.')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=DROPOUT, help='Dropout rate')
    parser.add_argument('--width', type=int, default=WIDTH, help='width')
    parser.add_argument('--height', type=int, default=HEIGHT, help='height')
    parser.add_argument('--depth', type=int, default=DEPTH, help='depth')
    parser.add_argument('--adjustement', type=float, default=ADJUSTMENT, help='x per pixel')
    parser.add_argument('--weights', type=str, help='Load weights')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path')
    parser.add_argument('--output', type=str, required=True, help='output path')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print('-------------')
    print('BATCH        : {}'.format(args.batch))
    print('EPOCH        : {}'.format(args.epoch))
    print('ALPA         : {}'.format(args.alpha))
    print('DROPOUT      : {}'.format(args.dropout))
    print('Load Weights?: {}'.format(args.weights))
    print('Dataset      : {}'.format(args.dataset))
    print('OUTPUT       : {}'.format(args.output))
    print('-------------')

    # TODO: abstract method to normalize speed.
    df_train, df_val = getDataFromThunderhill(args.dataset, split=True, randomize=True, balance=True)
    print('TRAIN:', len(df_train))
    print('VALIDATION:', len(df_val))
    print(df_train[['speed', 'throttle', 'brake', 'accel']].describe())
    model = NvidiaModel(args.dropout)

    print(model.summary())

    # Saves the model...
    with open(os.path.join(args.output, 'model.json'), 'w') as f:
        f.write(model.to_json())

    try:
        if args.weights:
            print('Loading weights from file ...')
            model.load_weights(args.weights)
    except IOError:
        print("No model found")
    checkpointer = ModelCheckpoint(os.path.join(args.output, 'weights.{epoch:02d}-{val_loss:.3f}.hdf5'))
    early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    logger = CSVLogger(filename=os.path.join(args.output, 'history.csv'))
    board = TensorBoard(log_dir=args.output, histogram_freq=0, write_graph=True, write_images=True)

    history = model.fit_generator(
        generate_thunderhill_batches(df_train, args),
        nb_epoch=args.epoch,
        samples_per_epoch=args.batch*50,
        validation_data=generate_thunderhill_batches(df_val, args),
        nb_val_samples=args.batch*10,
        callbacks=[
            checkpointer,
            ModelCheckpoint(os.path.join(args.output, 'model.h5'), monitor='val_loss', save_best_only=True),
            early_stop,
            logger,
            board
        ]
    )

    # for X, y in generate_thunderhill_batches(df_train, args):
    #     print(X.shape, y.shape)