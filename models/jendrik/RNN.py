'''
Created on Mar 17, 2017

@author: jendrik
'''

import tensorflow as tf

def weightVariable(shape, name=''):
        initial = tf.truncated_normal(shape=shape, mean=0, stddev=.1)
        return tf.Variable(initial, name = name)
    
def biasVariable(shape, name=''):
        initial = tf.constant(.1, shape=shape)
        return tf.Variable(initial, name = name)

class RNN(object):
    '''
    classdocs
    '''

    
    def __init__(self, shapeOfInput, numberOfOutputs):
        '''
        Constructor
        '''
        
        if len(shapeOfInput) != 4: raise Exception("X has not the dimension 4.")
        batchSize = shapeOfInput[0]
        height = shapeOfInput[1]
        width = shapeOfInput[2]
        numberOfChannels = shapeOfInput[3]
        
        self.x = tf.placeholder(tf.float32, shape=[None, shapeOfInput[1],
                                shapeOfInput[2],shapeOfInput[3]], name='x')
        self.keepProb = tf.placeholder(tf.float32, name='keepProb')
        
        #First convolution layer
        self.wConv1 = weightVariable([8,8, numberOfChannels, 24], name = 'wConv1')
        self.bConv1 = weightVariable([24], name = 'bConv1')
        self.hConv1 = tf.nn.elu(tf.nn.conv2d(self.x, self.wConv1, strides=(1,2,2,1), padding='VALID') + self.bConv1, name='hConv1')
        
        self.wConv2 = weightVariable([5,5, 24, 36], name = 'wConv2')
        self.bConv2 = weightVariable([36], name = 'bConv2')
        self.hConv2 = tf.nn.elu(tf.nn.conv2d(self.hConv1, self.wConv2, strides=(1,2,2,1), padding='VALID') + self.bConv2, name='hConv2')
        
        self.wConv3 = weightVariable([5,5, 36, 48], name = 'wConv3')
        self.bConv3 = weightVariable([48], name = 'bConv3')
        self.hConv3 = tf.nn.elu(tf.nn.conv2d(self.hConv2, self.wConv3, strides=(1,2,2,1), padding='VALID') + self.bConv3, name='hConv3')
        
        self.wConv4 = weightVariable([5,5, 48, 64], name = 'wConv4')
        self.bConv4 = weightVariable([64], name = 'bConv4')
        self.hConv4 = tf.nn.elu(tf.nn.conv2d(self.hConv3, self.wConv4, padding='VALID', strides=(1,1,1,1)) + self.bConv4, name='hConv4')
        
        self.wConv5 = weightVariable([5,5, 64, 64], name = 'wConv4')
        self.bConv5 = weightVariable([64], name = 'bConv4')
        self.hConv5 = tf.nn.elu(tf.nn.conv2d(self.hConv4, self.wConv5, padding='VALID', strides=(1,1,1,1)) + self.bConv5, name='hConv4')
        print(self.hConv5.get_shape())
        
        
        self.hConvFlat = tf.reshape(self.hConv5, shape=[-1, int(
            self.hConv5.get_shape()[1]*self.hConv5.get_shape()[2]*self.hConv5.get_shape()[3])],
                                    name = 'hConvFlat')
        
        self.lstm = tf.contrib.rnn.BasicLSTMCell(1024, activation=tf.nn.elu, state_is_tuple=True)
        
        self.cellState = tf.placeholder(tf.float32, shape=[batchSize, self.lstm.state_size], name='cellState')
        
        print(self.lstm.state_size)
        self.proj, self.newCellState = self.lstm(self.hConvFlat, self.cellState)
        
        self.wFull1 = weightVariable([int(self.proj.get_shape()[1]), 100], name='wFull1')
        self.bFull1 = biasVariable([100], name='bFull1')
        self.hFull1 = tf.nn.elu(tf.matmul(self.proj, self.wFull1) + self.bFull1, name='hFull1')
        
        self.wFull2 = weightVariable([100, 50], name='wFull2')
        self.bFull2 = biasVariable([50], name='bFull2')
        self.hFull2 = tf.nn.elu(tf.matmul(self.hFull1, self.wFull2) + self.bFull2, name='hFull2')
        
        self.hDrop = tf.nn.dropout(self.hFull2, self.keepProb, name='hDrop')
        
        self.wFull3 = weightVariable([50, 10], name='wFull3')
        self.bFull3 = biasVariable([10], name='bFull3')
        self.hFull3 = tf.nn.elu(tf.matmul(self.hFull2, self.wFull3) + self.bFull3, name='hFull3')
        
        self.wFull4 = weightVariable([10, numberOfOutputs], name='wFull4')
        self.bFull4 = biasVariable([numberOfOutputs], name='bFull4')
        self.hFull4 = tf.nn.sigmoid(tf.matmul(self.hFull3, self.wFull4) + self.bFull4, name='hFull4')
        self.out = self.hFull4*2 -1
        
    
        