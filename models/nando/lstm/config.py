import os

BATCH_SIZE = 1
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
COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed", "position","rotation"]
SKIP = ["dataset_sim_000_km_few_laps"]
COLUMNS_TO_NORMALIZE = ['positionX', 'positionY', 'positionZ', 'rotationX', 'rotationY', 'rotationZ']
FEATURES = ['positionX', 'positionY', 'positionZ', 'rotationX', 'rotationY', 'rotationZ', '']

SVR_MODEL = 'svr.p'
SCALER = 'scaler.p'

SAMPLES_PER_EPOCH = BATCH_SIZE*50
NB_VALIDATION_SAMPLE = BATCH_SIZE*10

# LSTM
SEQ_LENGTH = 20
CNN_INPUT_SIZE = 2496
LAYER = 11

# LSTM OUTPUTS
FEATURES = '/Users/nando/workspace/selfdrivingcar/thunderhill/lstm'
CNN_FEATURES = os.path.join(OUTPUT, 'CNNfeatures.p')
CNN_LABELS = os.path.join(OUTPUT, 'CNNLabels.p')
LSTM_FEATURES = os.path.join(OUTPUT, 'LSTMfeatures.p')
LSTM_LABELS = os.path.join(OUTPUT, 'LSTMLabels.p')