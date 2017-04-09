import os

BATCH_SIZE = 256
EPOCHS = 1000
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
WIDTH = 160
HEIGHT = 80
DEPTH = 3

ADJUSTMENT = 0.005
ALPHA = 0.001
DROPOUT = 0.5
OUTPUT = '.hdf5_checkpoints'

# columns information
COLUMNS = [
    'path', 'heading', 'longitude', 'latitude', 'quarternion0', 'quarternion1', 'quarternion2', 'quarternion3',
    'vel0','vel1','vel2','steering','throttle','brake', 'speed'
]
SKIPME = []
COLUMNS_TO_NORMALIZE = ['speed']


MAX = 33.706074