### Stateful RNN/LSTM Fully Autonomous Model

SRC_LSTM.py – Main program for building and training tensorflow slim model.

SRC_LSTM_rebuild.py – Rebuild the graph for test time with single image inputs.

drive.py – Use the stored states to predict steering/throttle/brake from single images

These scripts have been setup to train and test the 1st place stateful RNN/LSTM Solution from Udacity's Challenge #2 with simulator/polysync data. They were originally written by Ilia Edrenkin and have been modified for the SRC team. The original model was setup to minimize the RSME on the test set (toy problem) and many adjustments were required to get the model to drive in the simulator.

This model performs a mapping from sequences of images to sequences of steering angle, throttle and brake measurements. The model is based on three key components: 1) The input image sequences are processed with a 3D convolution stack, where the discrete time axis is interpreted as the first "depth" dimension. That allows the model to learn motion detectors and understand the dynamics of driving. 2) The model predicts not only the steering angle, throttle and brake. 3) The model is stateful: the two upper layers are a LSTM and a vanilla RNN, respectively. The model is optimized jointly for the autoregressive and ground truth modes: in the former, model's own outputs are fed into next timestep, in the latter, real targets are used as the context. At test time the model uses the “autoregressive” mode, where the previous predictions are used as input to the next predictions. I tried using different augmentation methods, but since the model learns from sequences, any augmentations (i.e. shifts, rotations, etc…) should be input to the model as smooth sequences. The only augmentation method that I am currently using is horizontal shifts to replicate left/right cameras since the polysync vehicle will only have a single forward facing camera. The images are also cropped and resized to achieve the (66,200) size that works well with the CNN architecture. The CNN architecture is similar to NVIDIA’s, but uses shortcut connections to each layer.
