"""
Dependencies:
    Keras
    Numpy
    OpenCV 3
    Pandas
    SciKit-Learn
"""

import numpy as np
import os
from datetime import datetime
from collections import namedtuple
from glob import glob

from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import utils
import models

__author__ = 'Jacob Thalman'
__email__ = 'jpthalman@gmail.com'


utils.setGlobals()
np.random.seed(1234)

# Create hyper-parameters
Parameters = namedtuple('Parameters', [
    # General settings
    'batch_size', 'max_epochs', 'output_dims',
    # Model settings
    'l2_reg', 'keep_prob',
    # Optimizer settings
    'learning_rate', 'epsilon', 'decay',
    # Training settings
    'min_delta', 'patience', 'kwargs'
  ])

params = Parameters(
    # General settings
    batch_size=64, max_epochs=100, output_dims=2,
    # Model settings
    l2_reg=0.0, keep_prob=0.5,
    # Optimizer settings
    learning_rate=1e-4, epsilon=1e-8, decay=0.0,
    # Training settings
    min_delta=0.0, patience=9, kwargs={'prob': 0.0}
  )


path = os.getcwd() + '/'

data = utils.load_data(path + 'thunderhill_data/dataset_sim_003_km_320x160/', 'driving_log.csv')
sim3_in, sim3_out = utils.process_data_multi_output(data, delay=8)

cutpoint = int(0.15*sim3_out.shape[0])
train_in, train_out, val_in, val_out = sim3_in[cutpoint:], sim3_out[cutpoint:], sim3_in[:cutpoint], sim3_out[:cutpoint]

print('\nTraining size: %d | Validation size: %d\n' % (train_out.shape[0], val_out.shape[0]))


# Model construction
model = models.dev(params)
optimizer = adam(
    lr=params.learning_rate,
    epsilon=params.epsilon,
    decay=params.decay
  )
model.compile(optimizer=optimizer, loss='mse')

print('\n', model.summary(), '\n')


# Clear TensorBoard Logs
for file in os.listdir('./logs/'):
    os.remove('./logs/' + file)

# Remove previous model files
try:
    for file in glob('models/*.h5'):
        os.remove(file)
    os.remove('model.h5')
except FileNotFoundError:
    pass


# Model training
filepath = 'models/model_{epoch:03d}_{val_loss:0.5f}.h5'
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=params.min_delta, patience=params.patience,
                  mode='min'),
    ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True,  mode='min'),
    TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
  ]

history = model.fit_generator(
    generator=utils.multi_output_batch_generator(inputs=train_in, outputs=train_out, batch_size=params.batch_size,
                                                 augmentor=utils.augment_image, kwargs=params.kwargs),
    samples_per_epoch=400,
    nb_epoch=params.max_epochs,

    # Halve the batch size, as `utils.val_augmentor` doubles the batch size by flipping the images and angles
    validation_data=utils.multi_output_batch_generator(
        inputs=val_in,
        outputs=val_out,
        batch_size=params.batch_size,
        augmentor=lambda x, y: (x, y),
        validation=True
      ),

    # Double the size of the validation set, as we are flipping the images to balance the right/left
    # distribution.
    nb_val_samples=val_out.shape[0]//params.batch_size,
    callbacks=callbacks
  )

print('\nFinished at: ' + str(datetime.now()) + '\n')
