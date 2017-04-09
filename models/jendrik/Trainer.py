'''
Created on 30 Jan 2017

@author: Jendrik
'''

import tensorflow as tf
import numpy as np
class Trainer(object):
    
    
    def __init__(self, network,learningRate, batchSize=1):
        self.network = network
        self.y_ = tf.placeholder(tf.float32, shape=[None, network.out.get_shape()[1]])
        self.batchSize = batchSize
        self.weight = tf.placeholder(tf.float32, shape=[batchSize])
        self.mse = tf.reduce_mean(self.weight*tf.square(network.out - self.y_))
        self.optimizer = tf.train.AdamOptimizer(learningRate)
        self.trainStep = self.optimizer.minimize(self.mse)
        correctPrediction = tf.equal(tf.argmax(network.out, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        self.saver = tf.train.Saver()
        with tf.name_scope('summaries'):
            tf.summary.scalar('trainAccuracy', self.accuracy)
            
    def buildSubBatch(self, i ,x, y=None):
        if(y==None):
            return x[i*self.batchSize:(i+1)*self.batchSize,:,:,:]
        if (i+1)*self.batchSize < len(x):
            if(len(y.shape) == 1):
                yBatch = self.network.encodeIndices(y[i*self.batchSize:(i+1)*self.batchSize])
                return x[i*self.batchSize:(i+1)*self.batchSize,:,:,:], yBatch
            else:
                xBatch = x[i*self.batchSize:(i+1)*self.batchSize,:,:,:]
                yBatch = y[i*self.batchSize:(i+1)*self.batchSize,:]
                return xBatch, yBatch
        else:
            if(len(y.shape) == 1):
                yBatch = self.network.encodeIndices(y[i*self.batchSize:])
                return x[i*self.batchSize:,:,:,:], yBatch
            else:
                xBatch = x[i*self.batchSize:,:,:,:]
                yBatch = y[i*self.batchSize:,:]
                return xBatch, yBatch
    
    def testAccuracy(self, sess, xVal, yVal, weight, cellState):
        acc, newCellState =  sess.run([self.accuracy, self.network.newCellState], feed_dict={
                    self.network.x:xVal, self.y_:yVal, self.network.keepProb: 1,
                    self.network.cellState: cellState})
        return weight*acc, newCellState
    
    def train(self, sess, xTrain, yTrain, weight, cellState, keepProb):
        """
            Performs a training step of the nsetwork going through the full 
            dataset provided
            Variables:
                sess - the session in which the execution shall be performed
                xTrain - contextData
                yTrain - labels
        """
        acc, cellRes, _ = sess.run([self.accuracy, self.network.newCellState, self.trainStep], feed_dict={
                    self.network.x:xTrain, self.y_:yTrain, self.network.keepProb: keepProb,
                    self.network.cellState: cellState,
                    self.weight: weight})
        return acc, cellRes
        #print("Train Accuracy %g" % self.testAccuracy(sess, xTrain, yTrain))
            