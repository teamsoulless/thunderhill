'''
Created on Feb 19, 2017

@author: jjordening
'''

import cv2
import numpy as np

def minValImage(arr, channel = 0):
    """
    Determines the minimum value of arr in a channel
    """
    return np.min(np.min(np.min(arr[:,:,:,channel], axis=1),axis=1),axis=0)

def maxValImage(arr, channel = 0):
    """
    Determines the maximium value of arr in a channel
    """
    return np.max(np.max(np.max(arr[:,:,:,channel], axis=0),axis=0),axis=0)

def addGrayLayer(image):
    """
    Adds a gray layer as a fourth channel to an image
    
    Input: 
        image
        
    Output:
        an array of images with the channels RGBGray
    """
    gray = cv2.cvtColor(image, cv2.CV_HLS2GRAY)
    return np.concatenate((image, gray.reshape((image.shape[0], image.shape[1],1))), axis=2)

def addGradientLayer(image, sobel_kernel=3, magThresh=(0, 255), dirThresh=(0, np.pi/2)):
    """
    Adds a gray layer as a fourth channel to an image
    
    Input: 
        image
        
    Output:
        an array of images with the channels RGBGray
    """
    sobelx = cv2.Sobel(image[:,:,1], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image[:,:,1], cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradMag = np.sqrt(sobelx**2 + sobely**2)
    absGradDir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Rescale to 8 bit
    scaleFactor = np.max(gradMag)/255
    gradMag = (gradMag/scaleFactor).astype(np.uint8)
    # Create a binary image of ones where thresholds are met, zeros otherwise
    binaryOutput = np.zeros_like(gradMag)
    binaryOutput[(gradMag >= magThresh[0]) & (gradMag <= magThresh[1]) &
                    (absGradDir >= dirThresh[0]) & (absGradDir <= dirThresh[1])] = 1 
    # Return the binary image
    return np.concatenate((image, binaryOutput.reshape((image.shape[0], image.shape[1],1))) ,axis=2)
    
    

def applyNormalisation(image):
    """
        Applies a normalisation to an image with the channels RGBGray.
        It applies a CLAHE normalisation to the gray layer and then normalises the
        values such, that they have a mean of 0 and a deviation of 1
        
        Input: 
            image
        
        Output:
            an array of images with the channels RGBGray
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image[:,:,3] = clahe.apply(image[:,:,3])
    spread =  np.max(np.max(image,axis=0), axis=0)-np.min(np.min(image,axis=0), axis=0)
    spread = np.array([val if val > 0 else 1 for val in spread])
    image = (2.*(image[:,:] - np.min(np.min(image,axis=0), axis=0))[:,:]/(spread)) -1
    return image

def preprocessImage(image, transform):
    """
    This function represents the default preprocessing for 
    an image to prepare them for the network
    """
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image = cv2.convertScaleAbs(image, alpha=(1))
    image = addGradientLayer(image, 7, (100,255), (0, np.pi/2))
    #image = np.concatenate((image, transform(image)), axis=2)
    image = image[image.shape[0]//3:,20:-20,:]
    return applyNormalisation(image)

def preprocessImages(arr, transform):
    """
    This function represents the default preprocessing for 
    images to prepare them for the network
    """
    return np.array([preprocessImage(image, transform) for image in arr])


def perspectiveTransform(img, M):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
        

def shiftImg(arr, horizontal, vertical):
    """
        This function shifts an image horizontally and vertically
        Input:
            horizontal - amplitude of shift in pixels (positive to the right
            negative to the left)
            vertical - aplitude of the ishift in pixels (positive downwards 
            negative upwards)
    """
    
    M = np.float32([[1,0,1*horizontal],[0,1,vertical]])
    return cv2.warpAffine(arr,M,(arr.shape[1], arr.shape[0]))

def mirrorImage(img):
    """
        This function mirrors the handed image around the y-axis
    """
    return img[:,::-1]

def rotateImage(img, angle):
    """
        Rotates image around the point in the middle of the bottom of the picture by
        angle degrees.
    """
    rotation = cv2.getRotationMatrix2D((img[0].shape[0], img[0].shape[1]), angle, 1)
    return cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))
    
def rotateImages(arr, angles):
    """
        Rotates multiple images by the given angles.
    """
    arr = [rotateImage(img, angle) for img, angle in zip(arr, angles)]
    return arr

