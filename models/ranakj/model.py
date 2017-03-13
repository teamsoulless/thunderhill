import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.misc.pilutil import imresize
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Lambda, Dropout, Lambda, ELU, PReLU, Cropping2D
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.objectives import mean_squared_error
from keras.callbacks import ModelCheckpoint

def resize_image(image):
    cropped_image = image[32:135, :]
    resized_image = imresize(cropped_image, .90, interp='bilinear', mode=None)
    return img_to_array(resized_image)

def random_brightness(image):
    change_pct = 0.25 + np.random.uniform()
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * change_pct
    img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_brightness

def translate(image, x_translate):
    y_translate = (35*np.random.uniform()) - (35/2)
    translate_matrix = np.float32([[1, 0, x_translate], [0, 1, y_translate]])
    return cv2.warpAffine(image, translate_matrix, (image.shape[1], image.shape[0]))

def load_images(driving_data, core_path='./data/'):
    combined_images = []
    steerings = []
    correction = 0.14  # this is a parameter to tune

    for left_image, center_image, right_image, steering_angle in zip(driving_data['left'], driving_data['center'], driving_data['right'], driving_data['steering']):
        lname = core_path + left_image
        cname = core_path + center_image
        rname = core_path + right_image

        limage = mpimg.imread(lname.replace(" ", ""))
        cimage = mpimg.imread(cname.replace(" ", ""))
        rimage = mpimg.imread(rname.replace(" ", ""))

        for image in np.array([cimage, limage, rimage]):
            image_copy = np.copy(image)
            # image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            image_copy = resize_image(image_copy)
            combined_images.append(image_copy)
        steering_center = float(steering_angle)
        # create adjusted steering measurements for the side camera images
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        steerings.append(steering_center)
        steerings.append(steering_left)
        steerings.append(steering_right)
    return combined_images, steerings

def preprocess_image_flip_and_brightness(x, y):
    augmented_images = []
    augmented_steering_angles = []
    for image, steering_angle in zip(x, y):
        x_translate = (100 * np.random.uniform()) - (100 / 2)
        aug_angle = steering_angle + ((x_translate / 100) * 2) * .3
        image_bright = random_brightness(image)
        augmented_images.append(image_bright)
        augmented_steering_angles.append(steering_angle)
        flipped_image = cv2.flip(image, 1)
        flipped_streering_angle = float(aug_angle) * -1.0
        augmented_images.append(flipped_image)
        augmented_steering_angles.append(flipped_streering_angle)
    return augmented_images, augmented_steering_angles

def generator(images, steering_angles, batch_size=32):
    num_samples = len(images)
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_images = images[offset:offset + batch_size]
            batch_steering_angles = steering_angles[offset:offset + batch_size]
            batch_images, batch_steering_angles = preprocess_image_flip_and_brightness(batch_images, batch_steering_angles)
            # trim image to only see section with road
            x = np.array(batch_images)
            y = np.array(batch_steering_angles)
            yield shuffle(x, y, random_state=0)

# Load the data
data = {}
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
# Read in CSV file
csv_loc = "data/driving_log.csv"
driving_data = pd.read_csv(csv_loc)

print('Read CSV Lines : ', len(driving_data))

# Drop 75% of Steering Angles with value = 0
# driving_data = driving_data.drop(driving_data[driving_data['steering'] == 0].sample(frac=0.75).index)

images, steering_angles = load_images(driving_data)

print('Number of Images in recorded set : ', len(images))

X_train, X_val, y_train, y_val = train_test_split(images, steering_angles,  test_size=0.2, random_state=1)


train_generator = generator(X_train, y_train, batch_size=32)
validation_generator = generator(X_val, y_val, batch_size=32)

print('Loaded Images!!')

# model
img_shape = images[0].shape

print('Image Shape : ', img_shape)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=img_shape))
model.add(Convolution2D(24, 5, 5,border_mode='valid', activation='elu', subsample=(2,2)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2,2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2,2)))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam, metrics=['mse', 'accuracy'])

model.summary()

# checkpoint
checkpoint = ModelCheckpoint("model-{epoch:02d}.h5")

# fit the model
history_object = model.fit_generator(train_generator, samples_per_epoch=len(X_train), nb_epoch=5, validation_data=validation_generator, nb_val_samples=len(X_val), callbacks=[checkpoint], verbose=1)

# save model
print('Saving Model Weights!!')
model.save('model.h5')
#
# with open('model.json','w') as f:
#     json.dump(model.to_json(), f, ensure_ascii=False)

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
# figure = plt.figure(1)
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# figure.savefig('mse_graph.jpg')


