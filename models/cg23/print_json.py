from keras.layers.advanced_activations import ELU
from keras.models import load_model
import tensorflow as tf
tf.python.control_flow_ops = tf
import json

model = load_model('model.h5')
json_string = model.to_json()

with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)
