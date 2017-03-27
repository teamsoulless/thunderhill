import random
import numpy as np
from data_augmentation_functions import trainer
import cv2
import config

class generators_funcs:
    """
    
    Author: Kiarie Ndegwa
    Team Udacity - Self Racing league 1-2nd April 2017
    Date: 24/3/2017
    
    """     
    def train_generator(self, data, batch_s=64):
        #Instantiate trainer
        train = trainer()
  
        while 1:
            x_img = []
            y_ang = []
            for i in range(batch_s):
                i_line = np.random.randint(len(data))
                line_data = data.iloc[[i_line]].reset_index()
                #image, steer, throttle, brake
                x,y, _, _  = train.data_aug_gen(line_data)
                
                if x is None:
                    batch_s+=1
                elif x is not None:
                    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
                    x_img.append(x)
                    y_ang.append(y)

            x_img = np.asarray(x_img).reshape(batch_s, config.ROWS, config.COLS, config.CH)
            y_ang  = np.asarray(y_ang).reshape(batch_s, 1) 
            yield x_img, y_ang


    def val_generator(self, data, batch_s=32): 
        #Instantiate trainer
        train = trainer()
        
        while 1:
            x_img = []
            y_ang = []
            for i in range(batch_s):
                i_line = np.random.randint(len(data))
                line_data = data.iloc[[i_line]].reset_index()
                path_file = line_data['frame'][0].strip()
                image = cv2.imread(config.PATH_FILE+config.TRAIN+path_file)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
                x = train.prepImage(image)
                y = line_data['steering'][0]
                y = np.array([[y]])
                x_img.append(x)
                y_ang.append(y)

            x_img = np.asarray(x_img).reshape(batch_s, config.ROWS, config.COLS, config.CH)
            y_ang  = np.asarray(y_ang).reshape(batch_s, 1)    
            yield x_img, y_ang