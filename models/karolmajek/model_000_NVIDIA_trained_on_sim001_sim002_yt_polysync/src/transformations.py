import cv2
import numpy as np


class Transform:

    def __init__(self):
        pass

    def apply(self):
        pass


class Grayscale(Transform):

    def apply(self, img):
        width, weight, depth = img.shape
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(width, weight, 1)

    def toString(self):
        return '{}'.format('Grayscale')


class RGB2HSV(Transform):

    def apply(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

class Normalizer(Transform):

    def __init__(self, a=-0.5, b=0.5, min=0, max=255):
        self.a = a
        self.b = b
        self.min = min
        self.max = max

    def apply(self, img):
        return self.a + (((img-self.min)*(self.b-self.a))/(self.max-self.min))

    def toString(self):
        return '{}'.format('Normalize')


class Preprocess(Transform):

    def __init__(self, transforms):
        self.transforms = transforms

    def apply(self, img):
        for trans in self.transforms:
            img = trans.apply(img)
        return img

    def applies(self, X_train):
        return np.array([self.apply(x) for x in X_train])


class Rotate(Transform):
    def __init__(self, angle=None):
        self.angle = angle

    def apply(self, img):
        rows, cols, depth = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), self.getAngle(), 1)
        return cv2.warpAffine(img, M, (cols, rows))

    def getAngle(self):
        if self.angle is None:
            return np.random.randint(-15, 15)
        return self.angle

    def toString(self):
        return '{} by {}'.format('Rotate', self.angle)


class Translate(Transform):
    def __init__(self, by_x=None, by_y=None):
        self.by_x = by_x
        self.by_y = by_y

    def apply(self, img):
        rows, cols, depth = img.shape
        M = np.float32([[1, 0, self.getX()], [0, 1, self.getY()]])
        return cv2.warpAffine(img, M, (cols, rows))

    def getX(self):
        if self.by_x is None:
            return np.random.randint(-2, 2)
        return self.by_x

    def getY(self):
        if self.by_y is None:
            return np.random.randint(-2, 2)
        return self.by_y

    def toString(self):
        return '{} by X {} by Y {}'.format('Translate', self.by_x, self.by_y)


class Identity(Transform):

    def apply(self, img):
        return img

    def toString(self):
        return '{}'.format('Identity')

class Skew(Transform):

    def __init__(self, pts1, pts2):
        self.pts1 = pts1
        self.pts2 = pts2

    def apply(self, img):
        img_size = (img.shape[1], img.shape[0])
        self.M = cv2.getPerspectiveTransform(self.pts1, self.pts2)
        self.invM = cv2.getPerspectiveTransform(self.pts2, self.pts1)
        dst = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return dst

    def toString(self):
        return '{}'.format('Skew')


class Resize(Transform):

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def apply(self, img):
        return cv2.resize(img, dsize=(self.width, self.height))


class Crop(Transform):

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def apply(self, img):
        return img[self.y_min:self.y_max, self.x_min:self.x_max]


class GuassianBlur(Transform):

    def __init__(self, radius=None):
        self.radius = radius

    def apply(self, img):
        return cv2.GaussianBlur(img,(self.getRadius(), self.getRadius()),0)

    def getRadius(self):
        if self.radius is None:
            return np.random.choice([3, 5, 7])
        return self.radius

    def toString(self):
        return '{}'.format('Guassian Blur')


""" http://docs.opencv.org/3.2.0/d5/daf/tutorial_py_histogram_equalization.html
"""
class Equalizer(Transform):

    def __init__(self, clipLimit=2, tileSize=(8, 8)):
        self.trans = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileSize)

    def apply(self, img):
        return self.trans.apply(img)


"""
Preprocessing images.
"""
def Preproc(img):
    if img.size == 0:
        return img

    img=img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

    img=-0.5+img/255
    #
    # kernel_size = 5
    # scale = 1
    # delta = 0
    # ddepth = cv2.CV_16S
    #
    # # hsv[:,:,1] = cv2.GaussianBlur(hsv[:,:,1],(5,5),0)
    # hsv[:,:,2] = cv2.GaussianBlur(hsv[:,:,2],(5,5),0)
    #
    # gray_lap = cv2.Laplacian(hsv[:,:,2],ddepth,ksize = kernel_size,scale = scale,delta = delta)
    # lap_v = cv2.convertScaleAbs(gray_lap)
    #
    # _min=np.min(lap_v)
    # _max=np.max(lap_v)
    # lap_v=(lap_v-_min)/(_max-_min)
    #
    # # lap_v[lap_v[:,:]>=0.5]=1
    # lap_v[lap_v[:,:]<0.4]=0
    #
    # red=img[:,:,2]
    # green=img[:,:,1]
    # blue=img[:,:,0]
    #
    # red[:,:]=hsv[:,:,2]
    # green[:,:]=xyz[:,:,1]
    # blue[:,:]=lap_v
    #
    #
    # red=(red-np.min(red))/(np.max(red)-np.min(red))
    # green=(green-np.min(green))/(np.max(green)-np.min(green))
    # blue=(blue-np.min(blue))/(np.max(blue)-np.min(blue))
    #
    # img=np.concatenate((blue.reshape(blue.shape[0],blue.shape[1],1),green.reshape(green.shape[0],green.shape[1],1),red.reshape(red.shape[0],red.shape[1],1)),axis=2)
    #
    # img=img-0.5
    #
    # height, width, depth = img.shape
    #
    # src = np.float32([[100, 75], [0, 120], [200, 75], [320, 120]])
    # dst = np.float32([[80, 0], [100, 160], [215, 0], [215, 160]])
    #
    preproc = Preprocess([
        # RGB2HSV(),
        # Crop(0, width, 200, height),  # x_min, x_max, y_min, y_max
        Resize(320,160),
        # Skew(src, dst),
        # Normalizer(a=-0.5, b=0.5)
    ])
    tmp=preproc.apply(img)

    return tmp

class Flip(Transform):

    def __init__(self, horizontal=True):
        self.horizontal = horizontal

    def apply(self, img):
        if self.horizontal:
            return cv2.flip(img, 1)
        return cv2.flip(img, 0)

"""
What worked....
0.01
Tried with 70-70 which worked best...


--------------------
.hdf5_checkpoints-6  ---- random-translation: -30
    BATCH_SIZE = 128
    EPOCHS = 30
    FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
    ALPHA = 0.01 on ConvNetLayers
    DROPOUT = 0.5 on FC layer

    Preproc
        - RandomShift
        - RandomFlip
        - RandomBrightness

"""
def Shift(_img, by_x=0, by_y=0):
    height = _img.shape[0]
    width = _img.shape[1]
    img = Translate(by_x=by_x, by_y=by_y).apply(_img)
    if by_x >= 0 and by_y >= 0:
        img = Crop(by_x, width, by_y, height).apply(img)
    elif by_x >= 0 and by_y < 0:
        img = Crop(by_x, width, 0, height+by_y).apply(img)
    elif by_x < 0 and by_y < 0:
        img = Crop(0, width + by_x, 0, height + by_y).apply(img)
    elif by_x < 0 and by_y >= 0:
        img = Crop(0, width + by_x, by_y, height).apply(img)
    img = Resize(width, height).apply(img)
    return img

"""
Randomly shift images
"""
def RandomShift(img, steering):
    # if np.random.uniform() < 0.5:
    #     return img, steering
    tx = np.random.randint(-5,5) * 30
    # steering += tx*0.004
    steering += tx*0.005
    return Shift(img, tx, np.random.randint(-50, 10)), steering

"""
Randomly flip the images
"""
def RandomFlip(img, steering):
    # if np.random.uniform() < 0.5:
    #     return img, steering
    return [(img, steering),(Flip().apply(img), -steering)]

def brigthness(image, brigthness):
    table = np.array([i+ brigthness    for i in np.arange(0, 256)])
    table[table<0]=0
    table[table>255]=255
    table=table.astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
"""
Randomly change the brightness
"""
def RandomBrightness(img, steering):
    # if np.random.uniform() < 0.5:
    #     return img, steering
    # img[:, :, :] = img[:, :, :] * np.random.uniform()*2
    return brigthness(img,-100+200*np.random.uniform()), steering
