import tensorflow as tf
import numpy as np
from collections import deque

class KomandaModel(object):
    """Steering angle prediction model for Udacity challenge 2.
    """
    def __init__(self, checkpoint_dir, metagraph_file):
        self.graph =tf.Graph()
        self.LEFT_CONTEXT = 5 # TODO remove hardcode; store it in the graph
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(metagraph_file)
            ckpt = tf.train.latest_checkpoint('./')
        self.session = tf.Session(graph=self.graph)
        saver.restore(self.session, ckpt)
        #self.ops = self.session.graph.get_operations()
        #for op in self.ops:
        #    print(op.values())
        self.input_images = deque() # will be of size self.LEFT_CONTEXT + 1
        self.internal_state = [] # will hold controller_{final -> initial}_state_{0,1,2}

        # TODO controller state names should be stored in the graph
        self.input_tensors = list(map(self.graph.get_tensor_by_name, ["input_images:0", "controller_initial_state_0:0", "controller_initial_state_1:0", "controller_initial_state_2:0"]))
        self.output_tensors = list(map(self.graph.get_tensor_by_name, ["output_steering:0", "output_throttle:0","output_brake:0","controller_final_state_0:0", "controller_final_state_1:0", "controller_final_state_2:0"]))

    def predict(self, img):
        if len(self.input_images) == 0:
            self.input_images += [img] * (self.LEFT_CONTEXT + 1)
        else:
            self.input_images.popleft()
            self.input_images.append(img)
        input_images_tensor = np.stack(self.input_images)
        if not self.internal_state:
            feed_dict = {self.input_tensors[0] : input_images_tensor}
        else:
            feed_dict = dict(zip(self.input_tensors, [input_images_tensor] + self.internal_state))
        steering, throttle, brake, c0, c1, c2 = self.session.run(self.output_tensors, feed_dict=feed_dict)
        #self.internal_state = [c0, c1[0,:,:], c2[0,:,:]]
        self.internal_state = [c0, c1, c2]
        return steering[0][0], throttle[0][0], brake[0][0]