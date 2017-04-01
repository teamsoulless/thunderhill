from PIL import Image
from glob import glob
import cv2
import numpy as np
import os
import io
from array import array

fname='image.raw'
file = open(fname, 'rb')
raw_image = Image.frombytes('RGB', [960,480], file.read(), 'raw')
raw_image.save('raw.bmp')
img = np.array(raw_image)
file.close()
 
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(fname.replace('.raw','.jpeg'), rgb)

image_array = cv2.resize(img, (320, 160))
cv2.imwrite('image_resized.jpg',image_array) 

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
cv2.imwrite('image_hvs.jpg', hsv)

hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
cv2.imwrite('image_hls.jpg', hls)

img_read = cv2.imread('image.jpeg', cv2.COLOR_BGR2RGB)
img_raw = cv2.imread('raw.bmp')
new_image = np.abs(img_read - img_raw)
cv2.imwrite('negative.jpg', new_image)