#This class contains all the functions needed to augment the training data set
import cv2
import matplotlib.pyplot as plt
import math
import config
import numpy as np

class trainer:
    """
    
    Author: Kiarie Ndegwa
    Team Udacity - Self Racing League 1-2nd April 2017
    Date: 24/3/2017
    
    """ 

    def data_aug_gen(self, d_frame):
        #Line data is extracted from pd data frame
        path_file = config.PATH_FILE+config.TRAIN+d_frame['frame'][0].strip()

        path_file = path_file.replace(" ", "_")
        #get corresponding steering angle for center camera
        steer = d_frame['steering'][0]

        #get corresponding brake and throttle
        throttle = d_frame['throttle'][0]
        brake = d_frame['brake'][0]

        #read in image if contained within IMG folder

        image = cv2.imread(path_file)
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        #augment image brightness randomly
        aug = np.random.randint(2)
        if aug == 1:
            image =  self.bright_aug(image)

        #preprocess image data, make image 64 by 64
        image = self.prepImage(image)
        image = np.array(image)

        #flip image randomly
        ind_flip = np.random.randint(2)

        if ind_flip==1:
            image,steer = self.hor_flip(image, steer)

        return image, steer, throttle, brake
    
    #Size of resized images
    def bright_aug(self, img):
        # 1 Brightness augmentation
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        r_bright = .25+np.random.uniform()
        img[:,:,2] = img[:,:,2]*r_bright
        img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
        return img

    def trans_image(self, image,steer,h_range, v_range):
        #Horizontal shift
        rows,cols, _ = ROWS, COLS

        tr_x = random.randint(-h_range//2, h_range//2)
        steer_ang = steer+(tr_x*.2)

        #Vertical shift
        tr_y = random.randint(-v_range//2, v_range//2)

        #Affine transform matrix
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
        #image_tr = cv2.cvtColor(image_tr,cv2.COLOR_BGR2RGB)
        return image_tr,steer_ang

    def hor_flip(self, img, steer_ang):
        #.3 Flip images to simulate driving in opposite direction
        img=cv2.flip(img,1)
        #img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img, -steer_ang

    #Preprocess data before training network
    #Image preprocessing
    def prepImage(self, img):
        #Get rid of top 1/5 of image, and bottom 25 pixels
        COL = config.COLS
        ROW = config.ROWS
        
        shape = img.shape
        img = img[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
        img = cv2.resize(img,(COL,ROW), interpolation=cv2.INTER_AREA)   
        return img
    
    def random_horizontal_flip(self, x, y):
        flip = np.random.randint(2)
        if flip:
            x = cv2.flip(x, 1)
            y = -y
        return x, y

    def random_translation(self, img, steering):
        trans_range = 30  # Pixel shift

        # Compute translation and corresponding steering angle
        tr_x = np.random.uniform(-trans_range, trans_range)
        steering = steering + (tr_x / trans_range) * 0.17

        rows = img.shape[0]
        cols = img.shape[1]

        #Warp image
        M = np.float32([[1,0,tr_x],[0,1,0]])
        img = cv2.warpAffine(img,M,(cols,rows))

        return img, steering

    def data_augmentation(self, x, y):
        # random horizontal shift
        x, y = random_translation(x, y)

        # random horizontal flip
        x, y = random_horizontal_flip(x, y)

        return x, y
