#This script contains all global variables needed in the training script
"""
    Author: Kiarie Ndegwa
    Team Udacity - Self Racing League 1-2nd April 2017
    Date: 24/3/2017

 """ 

#pathfile to data frame
PATH_FILE = "../../../thunderhill_data/"
    
#driver log
LOG =  "driving_log2.0.csv"

#folder name
TRAIN = "dataset_sim_004_km_320x160_cones_brakes/"

#get standard frame shapes
ROWS, COLS, CH = (160, 320, 3)
    
#get batch size
BATCH_SIZE = 64
    
#number of epochs
EPOCHS = 25