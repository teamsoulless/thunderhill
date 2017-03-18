from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Convolution2D, SeparableConv2D
from keras.regularizers import l2
from keras.layers import Activation
from keras.layers.advanced_activations import ELU, PReLU

from batch_renorm import BatchRenormalization


def nvidia_with_bn_and_dropout(params):
    model = Sequential([
        Lambda(lambda x: x / 255. - 0.5, input_shape=(66, 200, 3)),

        Convolution2D(3, 1, 1, border_mode='valid'),

        # 66x200x3
        Convolution2D(24, 5, 5, init='he_normal'),
        BatchRenormalization(),
        Activation('elu'),
        MaxPooling2D(),

        # 31x98x24
        Convolution2D(36, 5, 5, init='he_normal'),
        BatchRenormalization(),
        Activation('elu'),
        MaxPooling2D(border_mode='same'),

        # 14x47x36
        Convolution2D(48, 5, 5, init='he_normal'),
        BatchRenormalization(),
        Activation('elu'),
        # Dropout(params.keep_prob),
        MaxPooling2D(border_mode='same'),

        # 5x22x48
        Convolution2D(64, 3, 3, init='he_normal'),
        Dropout(params.keep_prob),
        BatchRenormalization(),
        Activation('elu'),

        # 3x20x64
        Convolution2D(64, 3, 3, init='he_normal'),
        Dropout(params.keep_prob),
        BatchRenormalization(),
        Activation('elu'),

        # 1x18x64
        Flatten(),

        Dense(100),
        Dropout(params.keep_prob),
        Activation('elu'),

        Dense(50),
        Activation('elu'),

        Dense(10),
        Activation('elu'),

        Dense(1)
    ])
    return model


def cg23(params):
    model = Sequential([
        Lambda(lambda x: x/255. - 0.5, input_shape=(160, 320, 3)),

        Convolution2D(3, 1, 1),

        Convolution2D(24, 8, 8, activation='elu', subsample=(4, 4)),
        Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)),
        Convolution2D(48, 2, 2, activation='elu', subsample=(2, 2)),
        Convolution2D(64, 3, 3, activation='elu'),
        Convolution2D(64, 3, 3, activation='elu'),
        Convolution2D(128, 3, 3, activation='elu'),
        Convolution2D(128, 3, 3, activation='elu'),

        Flatten(),

        Dense(100, activation='elu'),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        Dense(1)
    ])
    return model


def dev(params):
    model = Sequential([
        Lambda(lambda x: x/255. - 0.5, input_shape=(160, 320, 3)),

        Convolution2D(3, 1, 1),

        # 160x320x3
        Convolution2D(24, 8, 8, activation='elu', border_mode='valid', subsample=(4, 4)),
        BatchRenormalization(),

        #
    ])
