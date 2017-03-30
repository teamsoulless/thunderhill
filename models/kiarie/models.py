#This class contains all the models tested against data set
import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, ELU
from keras.optimizers import Adam
from keras.callbacks import Callback
import matplotlib.image as mpimg
from keras.layers import Input, Embedding, Merge, Flatten, recurrent, Dropout, RepeatVector, Dense, core, Reshape, Lambda, merge, Convolution2D, MaxPooling2D, Activation
from keras.layers import Input
import numpy as np
from keras.models import Model
import config

class models:
    """
    
    Author: Kiarie Ndegwa
    Team Udacity - Self Racing league 1-2nd April 2017
    Date: 24/3/2017
    
    """
   
    def __init__(self, model):
        self.model = self.model_pick(model)
        
    def model_pick(self, model = "Rambo"):
        if model == "Rambo":
            return self.Rambo()
        elif mode == "NVIDIA":
            return self.NVIDIA()
        
    def Rambo(self):
        row = config.ROWS
        col = config.COLS
        ch = config.CH
        
        #First branch
        main_input = Input(shape=(row, col, ch), name='main_input')

        branch1 = Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same")(main_input)
        branch1 = Activation('relu')(branch1)
        branch1 = Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same")(branch1)
        branch1 = Activation('relu')(branch1)
        branch1 = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same")(branch1)
        branch1 = Flatten()(branch1)
        branch1 = Activation('relu')(branch1)
        branch1 = Dense(512)(branch1)

        #Second branch
        branch2 = Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same")(main_input)
        branch2 = Activation('relu')(branch2)
        branch2 = Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same")(branch2)
        branch2 = Activation('relu')(branch2)
        branch2 = Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same")(branch2)
        branch2 = Activation('relu')(branch2)
        branch2 = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same")(branch2)
        branch2 = Activation('relu')(branch2)
        branch2 = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same")(branch2)
        branch2 = Flatten()(branch2)
        branch2 = Activation('relu')(branch2)
        branch2 = Dense(100)(branch2)
        branch2 = Activation('relu')(branch2)
        branch2 = Dense(50)(branch2)
        branch2 = Activation('relu')(branch2)
        branch2 = Dense(10)(branch2)

        #Third branch
        branch3 = Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same")(main_input)
        branch3 = Activation('relu')(branch3)
        branch3 = Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same")(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same")(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same")(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same")(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same")(branch3)
        branch3 = Flatten()(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Dense(100)(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Dense(50)(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Dense(10)(branch3)

        #Final merge
	#Dropout
        dropout_prob = 0.25
        output = merge([branch1, branch2, branch3], mode='concat', concat_axis=1)
        output = Dropout(dropout_prob)(output)
        output = Activation('relu')(output)
        output = Dense(1)(output)
        model =  Model(input = [main_input], output = [output])
        model.compile(optimizer="adam", loss="mse")
        return model
    
    def Rambo2(self):
        input_img = Input(shape=(3, 256, 256))

        tower_1 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
        tower_1 = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(tower_1)

        tower_2 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
        tower_2 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
        tower_3 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(tower_3)

        output = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=1)
        model =  Model(input = [input_img], output = [output])
        model.compile(optimizer="adam", loss="mse")
        return model
    
    def NVIDIA(self):    
        input_shape = (64,  64, 3)

        weight_init='glorot_uniform'
        padding = 'valid'
        dropout_prob = 0.25

            # Define model
        model = Sequential()

        model.add(Lambda(lambda X: X / 255. - 0.5, input_shape=input_shape))

        model.add(Convolution2D(24, 5, 5,
                                        border_mode=padding,
                                        init = weight_init, subsample = (2, 2)))
        model.add(ELU())
        model.add(Convolution2D(36, 5, 5,
                                border_mode=padding,
                                init = weight_init, subsample = (2, 2)))
        model.add(ELU())
        model.add(Convolution2D(48, 5, 5,
                                border_mode=padding,
                                init = weight_init, subsample = (2, 2)))
        model.add(ELU())
        model.add(Convolution2D(64, 3, 3,
                                border_mode=padding,
                                init = weight_init, subsample = (1, 1)))
        model.add(ELU())
        model.add(Convolution2D(64, 3, 3,
                                border_mode=padding,
                                init = weight_init, subsample = (1, 1)))

        model.add(Flatten())
        model.add(Dropout(dropout_prob))
        model.add(ELU())

        model.add(Dense(100, init = weight_init))
        model.add(Dropout(dropout_prob))
        model.add(ELU())

        model.add(Dense(50, init = weight_init))
        model.add(Dropout(dropout_prob))
        model.add(ELU())

        model.add(Dense(10, init = weight_init))
        model.add(Dropout(dropout_prob))
        model.add(ELU())

        model.add(Dense(1, init = weight_init, name = 'output'))
        # Compile it
        model.compile(loss = 'mse', optimizer = Adam(lr = 0.001))
        return model
