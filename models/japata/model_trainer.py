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

# Create hyper-parameters
Parameters = namedtuple('Parameters', [
    # General settings
    'batch_size', 'max_epochs',
    # Model settings
    'l2_reg', 'keep_prob',
    # Optimizer settings
    'learning_rate', 'epsilon', 'decay',
    # Training settings
    'min_delta', 'patience', 'kwargs'
  ])

params = Parameters(
    # General settings
    batch_size=64, max_epochs=100,
    # Model settings
    l2_reg=0.0, keep_prob=0.5,
    # Optimizer settings
    learning_rate=1e-4, epsilon=1e-8, decay=0.0,
    # Training settings
    min_delta=1e-4, patience=4, kwargs={'prob': 1.0}
  )


path = os.getcwd() + '/'

paths, angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'thunderhill_data/dataset_sim_001_km_320x160/', 'driving_log.csv'),
    condition_lambda=lambda x: abs(x) < 1e-5,
    keep_percent=0.3,
    drop_camera='both'
  )

rec_paths, rec_angs = utils.concat_all_cameras(
    data=utils.load_data(path + 'thunderhill_data/dataset_sim_002_km_320x160_recovery/', 'driving_log.csv'),
    condition_lambda=lambda x: abs(x) < 1e-5,
    keep_percent=0.3,
    drop_camera='both'
  )

# Aggregate all sets into one.
all_paths = np.concatenate((paths, rec_paths), axis=0)
all_angs = np.concatenate((angs, rec_angs), axis=0)

train_paths, val_paths, train_angs, val_angs = utils.split_data(
    features=all_paths,
    labels=all_angs,
    test_size=0.15
  )

print('Training size: %d | Validation size: %d' % (train_paths.shape[0], val_paths.shape[0]))


# Model construction
model = models.cg23(params)
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
    generator=utils.batch_generator(ims=train_paths, angs=train_angs, batch_size=params.batch_size,
                                    augmentor=utils.augment_image, kwargs=params.kwargs),
    samples_per_epoch=25600,
    nb_epoch=params.max_epochs,
    # Halve the batch size, as `utils.val_augmentor` doubles the batch size by flipping the images and angles
    validation_data=utils.batch_generator(ims=val_paths, angs=val_angs, batch_size=params.batch_size//2,
                                          augmentor=utils.val_augmentor, validation=True),
    # Double the size of the validation set, as we are flipping the images to balance the right/left
    # distribution.
    nb_val_samples=2*val_angs.shape[0],
    callbacks=callbacks
  )

print('Finished at: ' + str(datetime.now()))
