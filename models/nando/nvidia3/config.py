import os

BATCH_SIZE = 512
EPOCHS = 1000
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
WIDTH = 160
HEIGHT = 80
DEPTH = 3

ADJUSTMENT = 0.001
ALPHA = 0.001
DROPOUT = 0.5
OUTPUT = '.hdf5_checkpoints'

# columns information
COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed", "position", "rotation"]
SKIP = ["dataset_sim_000_km_few_laps"]
COLUMNS_TO_NORMALIZE = ['speed']
SCALER = 'scaler.p'