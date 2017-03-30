from PIL import Image
from glob import glob
import cv2
import numpy as np
files=glob('*.raw')
files=sorted(files)

for fname in files:
  print(fname)
  file = open(fname, 'rb')
  img = np.array(Image.frombytes('RGB', [960,480], file.read(), 'raw'))
  file.close()
  cv2.imwrite(fname.replace('.raw','.jpeg'),img)
  cv2.imshow('img',img)
  cv2.waitKey(1)
