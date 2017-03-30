import cv2
import numpy as np
from config import WIDTH, HEIGHT

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
        dst = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)
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
        return cv2.GaussianBlur(img, (self.getRadius(), self.getRadius()), 0)

    def getRadius(self):
        if self.radius is None:
            self.radius = np.random.choice([1, 3, 5])
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

    height, width, depth = img.shape

    preproc = Preprocess([
        Resize(WIDTH, HEIGHT)
    ])

    return preproc.apply(img)


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
        img = Crop(0, width + by_x, 0, height+by_y).apply(img)
    elif by_x < 0 and by_y >= 0:
        img = Crop(0, width + by_x, by_y, height).apply(img)
    img = Resize(width, height).apply(img)
    return img

"""
Randomly shift images
"""
def RandomShift(img, steering, adjustment=0.005):
    tx = np.random.randint(-10, 10)
    steering += tx*adjustment
    steering = min(steering, 1)
    steering = max(steering, -1)
    return Shift(img, tx, np.random.randint(-50, 10)), steering

"""
Randomly flip the images
"""
def RandomFlip(img, steering):
    if np.random.uniform() < 0.5:
        return img, steering
    return Flip().apply(img), -steering

"""
Randomly change the brightness
"""
def RandomBrightness(img, steering):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:, :, 2] = img[:, :, 2] * np.random.uniform(0.25, 1.01)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB), steering

"""
Randomly rotate the images.
"""
def RandomRotation(img, steering):

    rot_angle = np.random.uniform(-5, 5)
    return Rotate(rot_angle).apply(img), steering

"""
Randomly blur the images
"""
def RandomBlur(img, steering):
    if np.random.uniform() < 0.5:
        return img, steering
    return GuassianBlur().apply(img), steering