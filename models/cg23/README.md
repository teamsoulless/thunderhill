This model was able to drive in the simulator at ~50mph. The video is below. A writeup on the similar approach that I used for P3 is here: https://medium.com/towards-data-science/cnn-model-comparison-in-udacitys-driving-simulator-9261de09b45#.zii457ypz

The differences for the SRC model are:
- Top-down images used to train and test
- NVIDIA model layers modified for 320X160 images
- Left/right images usage commented out
