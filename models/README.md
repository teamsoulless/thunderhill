/models is used to store information related to training and testing Deep Learning models.

Thunderhill-Sim-Testing.ipynb performs birds-eye transform on 320X160 images from simulator and then creates a movie of the results.


How to run NANDO's script

nohup python model.py  --dataset /home/ubuntu/thunderhill_data/ --output .hdf5_checkpoints --weights model.h5 &
tail -F nohup.out