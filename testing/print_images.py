from PIL import Image
from glob import glob
import cv2
import numpy as np
import os
import io
from array import array
from matplotlib import pyplot as plt

fname='image.raw'
file = open(fname, 'rb')
raw_image = Image.frombytes('RGB', [960,480], file.read(), 'raw')
raw_image.save('raw.bmp')
img = np.array(raw_image)
file.close()

plt.imsave(fname.replace('.raw','.jpeg'), img)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('cv_' + fname.replace('.raw','.jpeg'), rgb)

img_read = cv2.imread('image.jpeg')
new_image = np.abs(img_read[:,:,0] - img[:,:,0])
print(np.max(new_image))
print(np.max(img_read))
print(np.max(img))
cv2.imwrite('image1.jpg',new_image)
cv2.imwrite('image2.jpg',img_read)
cv2.imwrite('image3.jpg',img)

image_array = cv2.resize(img, (320, 160))
cv2.imwrite('image_resized.jpg',image_array) 

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
cv2.imwrite('image_hvs.jpg', hsv)

hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
cv2.imwrite('image_hls.jpg', hls)

img_read = cv2.imread('image.jpeg', cv2.COLOR_BGR2RGB)
img_raw = cv2.imread('raw.bmp', cv2.COLOR_BGR2RGB)
new_image = np.abs(img_read - img_raw)
print(np.max(new_image))
print(np.max(img_read))
print(np.max(img_raw))
cv2.imwrite('negative.jpg', new_image)