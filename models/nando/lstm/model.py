# Import basic

# from keras.models import Sequential
import argparse

from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from keras.layers import Input, Dense, Flatten, ELU, merge, LSTM
from keras.models import Model, load_model, Sequential, model_from_json
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from loader import getDataFromFolder
from config import *
import numpy as np
import pickle
from loader import *
from transformations import *

CNN_INPUT_SIZE = 2496
SEQ_LENGTH = 3
BATCH_SIZE = 10


def load_from_h5(model_file, out):
    _model = load_model(model_file)
    model = Model(input=_model.layers[0].input, output=_model.layers[out].output)

    for layer in model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='mse')
    return model

def getFeaturesPath(impath):
    basename = os.path.basename(os.path.splitext(impath)[0]) + '.p'
    # root = os.path.join(FEATURES, impath.split('/')[5])
    root = impath.split('/')[5]
    if not os.path.exists(root):
        os.makedirs(root)
    fpath = os.path.join(root, basename)
    return fpath

def extractCNNFeatures(df, model_file, layer_out=11):
    features_extractor = load_from_h5(model_file, layer_out)
    for idx, row in df.iterrows():
        impath = row['center']
        fpath = getFeaturesPath(impath)
        img = ReadImg(impath)
        img = Preproc(img)
        cnn_features = features_extractor.predict(np.reshape(img, (1, HEIGHT, WIDTH, DEPTH)))[0]
        pickle.dump(cnn_features, open(fpath, 'wb'))

def generate_lstm_batches(df, seq_length, batch_size):
    batch_x = []
    batch_y = []
    while True:
        X = []
        i = np.random.randint(0, df.shape[0] - seq_length)
        y = df.iloc[i+seq_length]['steering']
        for idx, row in df.iloc[i: i + seq_length].iterrows():
            impath = row['center']
            img = ReadImg(impath)
            img = Preproc(img)
            cnn_features = pickle.load(open(getFeaturesPath(impath), 'rb'))
            X.append(cnn_features)

        batch_x.append(np.reshape(X, (1, seq_length, CNN_INPUT_SIZE)))
        batch_y.append(y)

        if len(batch_x) == batch_size:
            yield np.vstack(batch_x), np.vstack(batch_y)
            batch_x = []
            batch_y = []


def Lstm(batch, seq, cnn_size, dropout):
    lstm = Sequential()
    lstm.add(
        LSTM(
            256,
            batch_input_shape=(batch, seq, cnn_size),
            dropout_W=dropout,
            dropout_U=dropout,
            return_sequences=True,
            stateful=True
        )
    )

    lstm.add(
        LSTM(
            256,
            dropout_W=dropout,
            dropout_U=dropout,
            return_sequences=True,
            stateful=True
        )
    )

    lstm.add(Flatten())
    lstm.add(Dense(1, init='he_normal'))

    lstm.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mse']
    )

    return lstm

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
    parser.add_argument('--seq', type=int, default=SEQ_LENGTH, help='Look back for frame')
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

    df = getDataFromFolder(args.dataset, args.output, randomize=False, balance=True, split=False)
    # extractCNNFeatures(df, args.weights)

    train_size = int(0.7*len(df))
    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:]
    print('TRAIN:', len(df_train))
    print('VALIDATION:', len(df_val))

    model = Lstm(args.batch, args.seq, CNN_INPUT_SIZE, args.dropout)

    print(model.summary())

    # Saves the model...
    with open(os.path.join(args.output, 'model.json'), 'w') as f:
        f.write(model.to_json())

    checkpointer = ModelCheckpoint(os.path.join(args.output, 'weights.{epoch:02d}-{val_loss:.3f}.hdf5'))
    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    logger = CSVLogger(filename=os.path.join(args.output, 'history.csv'))
    board = TensorBoard(log_dir=args.output, histogram_freq=0, write_graph=True, write_images=True)

    history = model.fit_generator(
        generate_lstm_batches(df_train, args.seq, args.batch),
        nb_epoch=args.epoch,
        samples_per_epoch=args.batch*20000,
        validation_data=generate_lstm_batches(df_val, args.seq, args.batch),
        nb_val_samples=args.batch*5000,
        callbacks=[
            checkpointer,
            early_stop,
            logger,
            board
        ]
    )