# ----------------------------------------------------------------------------------
# Self Racing Cars - RNN for Steering, Throttle and Brake - LSTM_Sim.py
# ----------------------------------------------------------------------------------

'''
This script is setup to train the 1st place RNN Solution from Udacity's Challenge #2
with simulator data. The model is setup to predict steering, throttle and brake.

Data augmentation techniques have been added to increase the training data size,
improve the model's performance and to generalize the model's capabilities.

The steps are: 
1. Setup the data and augment before sending to batch generator
2. Generate images/outputs to train 
3. Train/validate the model

Original By: Illia Andreken, Modified By: Chris Gundling
'''

import numpy as np
import pandas as pd
import csv
import cv2
import random
import matplotlib as mpl
import matplotlib.image as mpimg
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os
slim = tf.contrib.slim

from numpy import sin, cos
from collections import defaultdict
from os import path
from scipy.misc import imread, imresize, imsave
from scipy import ndimage
from scipy import misc
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import ImageOps
from tensorflow.python.util import nest

# -------------------------------------------------------------------
# Parameter Setup
# -------------------------------------------------------------------

# Define constants
SEQ_LEN = 1
BATCH_SIZE = 1 
LEFT_CONTEXT = 5

# Input image parameters
HEIGHT = 66
WIDTH = 200
CHANNELS = 3 # RGB

# LSTM paramters for keeping model state
RNN_SIZE = 32
RNN_PROJ = 32

# Output dimension setup
CSV_HEADER = "center,left,right,steering,throttle,brake,speed".split(",")
OUTPUTS = CSV_HEADER[3:6] # steering, throttle, brake
OUTPUT_DIM = len(OUTPUTS) # predict: steering angle, throttle and brake

# --------------------------------------------------------------------
# Batch Generator for Training/Validation
# --------------------------------------------------------------------
class BatchGenerator(object):
    def __init__(self, sequence, seq_len, batch_size):
        self.sequence = sequence
        self.seq_len = seq_len
        self.batch_size = batch_size
        chunk_size = int(1 + (len(sequence) - 1) / batch_size) # chunk is around 5000 examples
        self.indices = [(i*chunk_size) % len(sequence) for i in range(batch_size)]
        
    def next(self):
        while True:
            output = []
            for i in range(self.batch_size):
                idx = self.indices[i]
                #if type(idx) != int:
                #    print('not int',idx)
                left_pad = self.sequence[idx - LEFT_CONTEXT:idx]
                if len(left_pad) < LEFT_CONTEXT:
                    left_pad = [self.sequence[0]] * (LEFT_CONTEXT - len(left_pad)) + list(left_pad)
                assert len(left_pad) == LEFT_CONTEXT
                leftover = len(self.sequence) - idx
                if leftover >= self.seq_len:
                    result = self.sequence[idx:idx + self.seq_len]
                else:
                    result = self.sequence[idx:] + self.sequence[:self.seq_len - leftover]
                assert len(result) == self.seq_len
                self.indices[i] = (idx + self.seq_len) % len(self.sequence)
                print([a for a in result])
                numbers = list(zip(*result))
                #print(numbers[:][0])
                #print(numbers[:][1])
                images = numbers[0][0] #list(zip(*result))
                targets = numbers[0][1]
                numbers2 = list(zip(*left_pad))
                images_left_pad = numbers2 # list(zip(*left_pad))
                print(images)
                print(images_left_pad)
                output.append((np.stack(images_left_pad + images), np.stack(targets)))
            
            output = list(zip(*output))
            output[0] = np.stack(output[0]) # batch_size x (LEFT_CONTEXT + seq_len)
            output[1] = np.stack(output[1]) # batch_size x seq_len x OUTPUT_DIM
            return output

# --------------------------------------------------------------------        
# Functions used for Data Read/Processing
# --------------------------------------------------------------------
def read_csv(filename):
    with open(filename, 'r') as f:
        lines = [ln.strip().strip(' ').split(",")[0:7] for ln in f.readlines()]
        lines_c = map(lambda x: (x[0], np.float32(x[3:6])), lines) # imagefile, outputs
        lines_l = map(lambda x: (x[1].strip(' '), np.float32(x[3:6])), lines) # imagefile, outputs
        lines_r = map(lambda x: (x[2].strip(' '), np.float32(x[3:6])), lines) # imagefile, outputs
        return lines_c, lines_l, lines_r


def read_csv2(filename):
    with open(filename, 'r') as f:
        lines = [ln.strip().strip(' ').split(",")[0:7] for ln in f.readlines()]
        lines = map(lambda x: (x[0], np.float32(x[3:6])), lines) # imagefile, outputs
        return lines


def process_csv(filename, filename2, val=5):
    sum_f = np.float128([0.0] * OUTPUT_DIM)
    sum_sq_f = np.float128([0.0] * OUTPUT_DIM)
    lines_c, lines_l, lines_r = read_csv(filename)
    # leave val% for validation
    # for each iteration 3800 examples to training, 200 to validation
    train_seq = []
    valid_seq = []
    lines_total = (lines_c, lines_l, lines_r)
    cnt = 0
    for i in range(3):
        for ln in lines_total[i]:
            if i == 1:
                ln_new = list(ln)
                ln_new[1][0] += 0.25
                ln = tuple(ln_new)
                val = 0
            if i == 2:
                ln_new = list(ln)
                ln_new[1][0] -= 0.25
                ln = tuple(ln_new)
                val = 0
            ln_new = list(ln)
            ln = tuple(ln_new)
            if cnt < SEQ_LEN * BATCH_SIZE * (100 - val):
                train_seq.append(ln)
                sum_f += ln[1]
                sum_sq_f += ln[1] * ln[1]
            else:
                valid_seq.append(ln)
            cnt += 1
            cnt %= SEQ_LEN * BATCH_SIZE * 100

    lines = read_csv2(filename2)
    val = 15
    for ln in lines:
        ln_new = list(ln)
        ln = tuple(ln_new)
        if cnt < SEQ_LEN * BATCH_SIZE * (100 - val):
            train_seq.append(ln)
            sum_f += ln[1]
            sum_sq_f += ln[1] * ln[1]
        else:
            valid_seq.append(ln)
        cnt += 1
        cnt %= SEQ_LEN * BATCH_SIZE * 100
    mean = sum_f / len(train_seq)
    var = sum_sq_f / len(train_seq) - mean * mean
    std = np.sqrt(var)
    print(len(train_seq), len(valid_seq))
    return (train_seq, valid_seq), (mean, std)

(train_seq, valid_seq), (mean, std) = process_csv(filename="data/driving_log_11.csv", filename2="data/driving_log_20.csv", val=15)
test_seq = read_csv("data/data_test.csv") # dummy output file

# --------------------------------------------------------------------
# Build the Model
# --------------------------------------------------------------------
layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True, activation_fn=None, trainable=True)

def get_optimizer(loss, lrate):
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
    gradvars = optimizer.compute_gradients(loss)
    gradients, v = zip(*gradvars)
    #print [x.name for x in v]
    gradients, _ = tf.clip_by_global_norm(gradients, 15.0)
    return optimizer.apply_gradients(zip(gradients, v))

def apply_vision_simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
    video = tf.reshape(image, shape=[batch_size, LEFT_CONTEXT + seq_len, HEIGHT, WIDTH, CHANNELS])
    
    with tf.variable_scope(scope, 'Vision', [image], reuse=reuse):

        # input size for Challenge #2     : [4, 15, 480, 640, 3]
        # input size for NVIDIA simulator : [4, 25,  66, 200, 3]

        # layer 1
        net = slim.convolution(video, num_outputs=64, kernel_size=[2,5,5], stride=[1,2,2], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux1 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128, activation_fn=None)
        
        # NVIDIA sim net size : [4, 24, 31,  98, 64] 

        # layer 2
        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,2,2], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux2 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128, activation_fn=None)
        
        # NVIDIA sim net size : [4, 23, 14, 47, 64]

        # layer 3
        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,2,2], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux3 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128, activation_fn=None)
        
        # NVIDIA sim net size : [4, 22,  5, 22, 64]

        # layer 4
        net = slim.convolution(net, num_outputs=64, kernel_size=[2,3,3], stride=[1,1,1], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux4 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128, activation_fn=None)
        
        # NVIDIA sim net size : [4, 21,  3, 20, 64]

        # layer 5 for NVIDIA
        net = slim.convolution(net, num_outputs=64, kernel_size=[2,3,3], stride=[1,1,1], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux5 = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 128, activation_fn=None)

        # NVIDIA sim net size : [4, 20, 1, 18, 64]
        # at this point the NVIDIA tensor 'net' is of shape batch_size x seq_len x ...

        # net fc 1
        net = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 512, activation_fn=tf.nn.elu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)

        # net fc 2
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.elu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)

        # net fc 3
        net = slim.fully_connected(net, 128, activation_fn=None)

        # output size : [4, 20, 128]
        return layer_norm(tf.nn.elu(net + aux1 + aux2 + aux3 + aux4 + aux5)) # aux[1-5] are residual connections (shortcuts)


class SamplingRNNCell(tf.contrib.rnn.RNNCell):
#class SamplingRNNCell(tf.nn.rnn_cell.RNNCell):
    """Simple sampling RNN cell"""

    def __init__(self, num_outputs, use_ground_truth, internal_cell):
        """
        if use_ground_truth then don't sample
        """
        self._num_outputs = num_outputs
        self._use_ground_truth = use_ground_truth # boolean
        self._internal_cell = internal_cell # may be LSTM or GRU or anything
  
    @property
    def state_size(self):
        return self._num_outputs, self._internal_cell.state_size # previous output and bottleneck state

    @property
    def output_size(self):
        return self._num_outputs # steering angle, throttle, brake

    def __call__(self, inputs, state, scope=None):
        (visual_feats, current_ground_truth) = inputs
        prev_output, prev_state_internal = state
        context = tf.concat([prev_output, visual_feats],1)
        new_output_internal, new_state_internal = internal_cell(context, prev_state_internal) # here the internal cell (e.g. LSTM) is called
        new_output = tf.contrib.layers.fully_connected(
                     inputs=tf.concat([new_output_internal, prev_output, visual_feats],1),
                     num_outputs=self._num_outputs,
                     activation_fn=None,
                     scope="OutputProjection")

        # if self._use_ground_truth == True, pass the ground truth as state; otherwise, use the model's predictions
        return new_output, (current_ground_truth if self._use_ground_truth else new_output, new_state_internal)
    
# -------------------------------------------------------------------
# Build the Graph
# -------------------------------------------------------------------    
graph = tf.Graph()

with graph.as_default():

    # Hyperparameters
    learning_rate = tf.placeholder_with_default(input=1e-4, shape=())
    keep_prob = tf.placeholder_with_default(input=1.0, shape=())
    aux_cost_weight = tf.placeholder_with_default(input=0.1, shape=())
    
    # Inputs to Vision
    inputs = tf.placeholder(shape=(BATCH_SIZE,LEFT_CONTEXT+SEQ_LEN), dtype=tf.string) # image paths (left_context + seq_len)*batch_size
    targets = tf.placeholder(shape=(BATCH_SIZE,SEQ_LEN,OUTPUT_DIM), dtype=tf.float32) # seq_len x batch_size x OUTPUT_DIM
    #targets_normalized = (targets - mean) / std
    targets_normalized = targets
    #input_images = tf.pack([tf.image.decode_jpeg(tf.read_file(x))
    #                        for x in tf.unpack(tf.reshape(inputs, shape=[(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE]))])
    input_images = tf.stack([tf.image.decode_jpeg(tf.read_file(x))
                            for x in tf.unstack(tf.reshape(inputs, shape=[(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE]))])
    # Pre-process (crop and normalize)
    input_images = tf.identity(input_images, name="input_images")
    
    input_images = input_images[:, 40:140, :, :]
    image_size = HEIGHT, WIDTH
    input_images = tf.image.resize_images(input_images, size=image_size)
    #input_images.set_shape([(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
    #image_tensor = input_images[0,:,:,:]
    input_images = -1.0 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
    input_images.set_shape([(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])

    visual_conditions_reshaped = apply_vision_simple(image=input_images, keep_prob=keep_prob, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    visual_conditions = tf.reshape(visual_conditions_reshaped, [BATCH_SIZE, SEQ_LEN, -1])
    visual_conditions = tf.nn.dropout(x=visual_conditions, keep_prob=keep_prob)
    
    # Inputs to RNN
    #rnn_inputs_with_ground_truth = (visual_conditions, targets_normalized)
    rnn_inputs_autoregressive = (visual_conditions, tf.zeros(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), dtype=tf.float32))
    
    # LSTM Cell with for GT and autoregressive
    #internal_cell = tf.nn.rnn_cell.LSTMCell(num_units=RNN_SIZE, num_proj=RNN_PROJ)
    internal_cell = tf.contrib.rnn.LSTMCell(num_units=RNN_SIZE, num_proj=RNN_PROJ)
    #cell_with_ground_truth = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=True, internal_cell=internal_cell)
    cell_autoregressive = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=False, internal_cell=internal_cell)
    
    def get_initial_state(complex_state_tuple_sizes):
        #flat_sizes = tf.nn.rnn_cell.nest.flatten(complex_state_tuple_sizes)
        flat_sizes = nest.flatten(complex_state_tuple_sizes)
        init_state_flat = [tf.tile(
            multiples=[BATCH_SIZE, 1], 
            input=tf.get_variable("controller_initial_state_%d" % i, initializer=tf.zeros_initializer(), shape=([1, s]), dtype=tf.float32))
            for i,s in enumerate(flat_sizes)]
        #init_state = tf.nn.rnn_cell.nest.pack_sequence_as(complex_state_tuple_sizes, init_state_flat)
        init_state = nest.pack_sequence_as(complex_state_tuple_sizes, init_state_flat)
        #print(tf.shape(init_state_flat))
        return init_state

    def deep_copy_initial_state(complex_state_tuple):
        #flat_state = tf.nn.rnn_cell.nest.flatten(complex_state_tuple)
        flat_state = nest.flatten(complex_state_tuple)
        flat_copy = [tf.identity(s) for s in flat_state]
        #deep_copy = tf.nn.rnn_cell.nest.pack_sequence_as(complex_state_tuple, flat_copy)
        deep_copy = nest.pack_sequence_as(complex_state_tuple, flat_copy)
        return deep_copy
    
    controller_initial_state_variables = get_initial_state(cell_autoregressive.state_size)
    controller_initial_state_autoregressive = deep_copy_initial_state(controller_initial_state_variables)
    #controller_initial_state_gt = deep_copy_initial_state(controller_initial_state_variables)
    #print(tf.shape(controller_initial_state_autoregressive))
    
    #with tf.variable_scope("predictor"):
    #    out_gt, controller_final_state_gt = tf.nn.dynamic_rnn(cell=cell_with_ground_truth, inputs=rnn_inputs_with_ground_truth, 
    #                      sequence_length=[SEQ_LEN]*BATCH_SIZE, initial_state=controller_initial_state_gt, dtype=tf.float32,
    #                      swap_memory=True, time_major=False)

    with tf.variable_scope("predictor"):#, #reuse=True):
        out_autoregressive, controller_final_state_autoregressive = tf.nn.dynamic_rnn(cell=cell_autoregressive, inputs=rnn_inputs_autoregressive, 
                          sequence_length=[SEQ_LEN]*BATCH_SIZE, initial_state=controller_initial_state_autoregressive, dtype=tf.float32,
                          swap_memory=True, time_major=False)
    
    #mse_gt = tf.reduce_mean(tf.squared_difference(out_gt, targets_normalized))
    #mse_autoregressive = tf.reduce_mean(tf.squared_difference(out_autoregressive, targets_normalized))
    #mse_autoregressive_steering = tf.reduce_mean(tf.squared_difference(out_autoregressive[:, :, 0], targets_normalized[:, :, 0]))
    
    steering_predictions = (out_autoregressive[:, :, 0])# * std[0]) + mean[0]
    output_steering = tf.identity(steering_predictions, name="output_steering")
    
    throttle_predictions = (out_autoregressive[:, :, 1])# * std[1]) + mean[1]
    output_throttle = tf.identity(throttle_predictions, name="output_throttle")
   
    brake_predictions = (out_autoregressive[:, :, 2])# * std[2]) + mean[2]
    output_brake = tf.identity(brake_predictions, name="output_brake")
    
    controller_final_state_0 = tf.identity(controller_final_state_autoregressive[0], name="controller_final_state_0")
    controller_final_state_1 = tf.identity(controller_final_state_autoregressive[1][0], name="controller_final_state_1")
    controller_final_state_2 = tf.identity(controller_final_state_autoregressive[1][1], name="controller_final_state_2")
  
    #total_loss = mse_autoregressive_steering + aux_cost_weight * (mse_gt + mse_autoregressive)
    
    #optimizer = get_optimizer(total_loss, learning_rate)

    #tf.summary.scalar("MAIN TRAIN METRIC: rmse_autoregressive_steering", tf.sqrt(mse_autoregressive_steering))
    #tf.summary.scalar("rmse_gt", tf.sqrt(mse_gt))
    #tf.summary.scalar("rmse_autoregressive", tf.sqrt(mse_autoregressive))
    
    #summaries = tf.summary.merge_all() #tf.merge_all_summaries()
    #train_writer = tf.summary.FileWriter('v4/train_summary', graph=graph)
    #valid_writer = tf.summary.FileWriter('v4/valid_summary', graph=graph)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

# --------------------------------------------------------------------
# Training, Validate and Test the Model
# --------------------------------------------------------------------
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

checkpoint_dir = os.getcwd() + "/v4"

global_train_step = 0
global_valid_step = 0

KEEP_PROB_TRAIN = 1.0

def do_epoch(session, sequences, mode):
    global global_train_step, global_valid_step
    test_predictions = {}
    valid_predictions = {}
    batch_generator = BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    total_num_steps = 2 #1 + (batch_generator.indices[1] - 1) / SEQ_LEN
    controller_final_state_gt_cur, controller_final_state_autoregressive_cur = None, None
    acc_loss = np.float128(0.0)

    for step in range(int(total_num_steps)):
        feed_inputs, feed_targets = batch_generator.next()
        feed_dict = {inputs : feed_inputs, targets : feed_targets}

        if controller_final_state_autoregressive_cur is not None:
            feed_dict.update({controller_initial_state_autoregressive : controller_final_state_autoregressive_cur})

        if controller_final_state_gt_cur is not None:
            feed_dict.update({controller_final_state_gt : controller_final_state_gt_cur})

        if mode == "train":
            feed_dict.update({keep_prob : KEEP_PROB_TRAIN})
            summary, _, loss, controller_final_state_gt_cur, controller_final_state_autoregressive_cur = \
                session.run([summaries, optimizer, mse_autoregressive_steering, controller_final_state_gt, controller_final_state_autoregressive],
                           feed_dict = feed_dict)
            #img = session.run(image_tensor,feed_dict = feed_dict)
            #imsave('x.jpg',img)
            train_writer.add_summary(summary, global_train_step)
            global_train_step += 1

        elif mode == "valid":
            model_predictions, model_predictions_t, model_predictions_b, summary, loss, controller_final_state_autoregressive_cur = \
                session.run([steering_predictions, throttle_predictions, brake_predictions, summaries, mse_autoregressive_steering, controller_final_state_autoregressive],
                           feed_dict = feed_dict)
            valid_writer.add_summary(summary, global_valid_step)
            global_valid_step += 1  
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            steering_targets = feed_targets[:, :, 0].flatten()
            model_predictions = model_predictions.flatten()
            stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions)**2])
            for i, img in enumerate(feed_inputs):
                valid_predictions[img] = stats[:, i]

        elif mode == "test":
            model_predictions, model_predictions_t, model_predictions_b, controller_final_state_autoregressive_cur = \
                session.run([steering_predictions, throttle_predictions, brake_predictions, controller_final_state_autoregressive],
                           feed_dict = feed_dict)           
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            model_predictions = model_predictions.flatten()
            for i, img in enumerate(feed_inputs):
                test_predictions[img] = model_predictions[i]

        if mode != "test":
            acc_loss += loss
            print('\r', step + 1, "/", int(total_num_steps), np.sqrt(acc_loss / (step+1)))
    return (np.sqrt(acc_loss / total_num_steps), valid_predictions) if mode != "test" else (None, test_predictions)
    
# --------------------------------------------------------------------
# Train/Validate and Save the Model
# --------------------------------------------------------------------

NUM_EPOCHS=1
best_validation_score = None

with tf.Session(graph=graph) as session: #(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    session.run(tf.global_variables_initializer()) #tf.initialize_all_variables())
    print('Initialized')
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt:
        print("Restoring from", ckpt)
        saver.restore(sess=session, save_path=ckpt)
    
    for epoch in range(NUM_EPOCHS):
        print("Starting epoch %d" % epoch)
        #print("Validation:")
        #valid_score, valid_predictions = do_epoch(session=session, sequences=valid_seq, mode="valid")
        
        #if best_validation_score is None: 
        #    best_validation_score = valid_score
        
        #if valid_score < best_validation_score:
        #    saver.save(session, 'v4/checkpoint-sdc-ch2')
        #    best_validation_score = valid_score
        #    print('\r', "SAVED at epoch %d" % epoch)

        with open("v4/test-predictions-epoch%d" % epoch, "w") as out:
            _, test_predictions = do_epoch(session=session, sequences=test_seq, mode="test")
            print(out, "frame_id,steering_angle")
            print('Graph Rebuilt')
            for img, pred in test_predictions.items():
                #img = img.replace("challenge_2/Test-final/center/", "")
                print(out, "%s,%f" % (img, pred))
        
    saver.save(session, 'v4/checkpoint-sdc-ch2')
