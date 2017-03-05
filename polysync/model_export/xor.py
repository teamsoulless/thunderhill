from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import graph_util


X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

y_train = np.array([[0], [1], [1], [0]], "float32")

model = Sequential()
model.add(Dense(16, name='Foo', input_dim=2, activation='relu'))
model.add(Dense(1, name='Bar', activation='sigmoid'))

#model.summary()

model.compile(optimizer='adam',
              loss='mse',
              metrics=['binary_accuracy'])

model.fit(X_train, y_train, nb_epoch=5000)

print (model.predict(X_train))

sess = K.get_session()
#print(tf_session.graph_def)

nodes = set([n.name for n in tf.get_default_graph().as_graph_def().node])
nodes.remove('dense_input_1')

tf.train.write_graph(graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['Sigmoid'], nodes), 'models/', 'graph.pb', as_text=False)
