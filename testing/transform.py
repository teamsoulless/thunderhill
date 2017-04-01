from PIL import Image
import glob
import cv2
import numpy as np
import os
import io
from matplotlib import pyplot as plt


for data_set in glob.glob(os.path.join('**', '*.jpeg')):
	print('Image: ', data_set)
	fname= data_set	
	file = open(fname, 'rb')
	raw_image = Image.frombytes('RGB', [960,480], file.read(), 'raw')
	#raw_image.save('raw.bmp')
	img = np.array(raw_image)
	file.close()
	plt.imsave(fname, img)
	
	#rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#cv2.imwrite('cv_' + fname.replace('.raw','.jpeg'), rgb