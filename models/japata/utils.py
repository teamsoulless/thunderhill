import numpy as np
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from decorators import n_images


def setGlobals():
    global _color_space, _reshape, _gradients, _src, _dst

    _color_space = {'src': cv2.COLOR_HSV2RGB, 'dst': cv2.COLOR_RGB2HSV}
    _reshape = {'src': (320, 160), 'dst': (320, 160)}
    _gradients = {'cond': False, 'channel': 2}
    _src = np.array([
        [110, 90],
        [0, 160],
        [210, 90],
        [320, 160]
      ]).astype(np.float32)
    _dst = None  # np.array([
      #   [110, 90],
      #   [110, 160],
      #   [210, 90],
      #   [210, 160]
      # ]).astype(np.float32)


def getGlobals():
    return _color_space, _reshape, _gradients, _src, _dst


def load_data(path, file):
    """
    Opens driving_log.csv and returns center, left, right, and steering in a dictionary.

    :param path: Full file path to file
    :param file: The name of the file to load

    :type path: String
    :type file: String

    :return: Dictionary containing the camera file paths and steering angles.
    :rtype: Dictionary with keys = ['angles', 'center', 'left', 'right']
    """
    df = pd.read_csv(path + file, names=['CenterImage', 'LeftImage', 'RightImage', 'SteeringAngle',
                                         'Throttle', 'Break', 'Speed', 'Lat', 'Long'])
    data = {
        'angles': df['SteeringAngle'].astype('float32').as_matrix(),
        'center': np.array([path + str(im).replace(' ', '').replace('\\', '/') for im in df['CenterImage'].as_matrix()]),
        'right': np.array([path + str(im).replace(' ', '').replace('\\', '/') for im in df['RightImage'].as_matrix()]),
        'left': np.array([path + str(im).replace(' ', '').replace('\\', '/') for im in df['LeftImage'].as_matrix()])
      }
    return data


def keep_n_percent_of_data_where(data, values, condition_lambda, percent):
    """
    Keeps n-percent of a dataset where a condition over the values is true.

    For example, if you want to remove 90% of the samples in a dataset where the steering angles are
    close to zero, the code would look like this:

        images, angles = keep_n_percent_of_data_where(images, angles, lambda x: abs(x) < 1e-5, 0.1)

    Note that the lambda should return True for the values you would like to filter.

    :param data: The dataset
    :param values: The values to filter over
    :param condition_lambda: A lambda by which to filter the dataset
    :param percent: The percent of the dataset-value pair (where `condition_lambda` is true) to KEEP

    :type data: np.ndarray
    :type values: np.ndarray
    :type condition_lambda: lambda or function
    :type percent: float

    :return: Filtered tuple: (filtered_data, filtered_values)
    :rtype: Tuple
    """
    assert data.shape[0] == values.shape[0], 'Different # of data points and values.'

    cond_true = condition_lambda(values)
    data_true, data_false = data[cond_true, ...], data[~cond_true, ...]
    val_true, val_false = values[cond_true], values[~cond_true]

    cutoff = int(percent * data_true.shape[0])
    # Shuffle before clipping the top (1-n)%
    data_true, val_true = shuffle(data_true, val_true)
    # Only keep n% of the data points where the condition is true
    clipped_data_true, clipped_val_true = data_true[:cutoff, ...], val_true[:cutoff]

    filtered_data = np.concatenate((data_false, clipped_data_true), axis=0)
    filtered_values = np.concatenate((val_false, clipped_val_true), axis=0)
    return filtered_data, filtered_values


def transform_ang(angle, camera, recovery_dist=15, camera_shift=1.5, angle_range=25):
    """
    Transforms a steering angle to the perspective of the left or right camera.

    Input angle should be normalized between [-1, 1] with the `angle_range` representing
    the full scale of angles.

    :param angle: The steering angle of the center camera.
    :param camera: The camera to transform the perspective to. Either 'left' or 'right'.
    :param recovery_dist: The estimated distance in meters for the car to recover to the center
        of the lane at the current steering angle.
    :param camera_shift: The distance in meters from the center camera to the outside cameras.
    :param angle_range: The maximum degree the steering angles can take. Should be a positive
        number [0, inf) and represent the angle in degrees, not radians.
    :return: Corrected steering angle, normalized between [-1, 1].
    """
    sign = -np.sign(angle)
    rad = np.deg2rad(abs(angle_range*angle))
    tan = np.tan(np.pi/2 - rad)

    if camera == 'right':
        arctan = np.arctan((sign*recovery_dist + camera_shift*tan) / (tan*recovery_dist))
    elif camera == 'left':
        arctan = np.arctan((sign*recovery_dist - camera_shift*tan) / (tan*recovery_dist))
    else:
        raise ValueError("Argument `camera` must be 'left' or 'right'.")

    true_ang = -np.rad2deg(arctan)/angle_range
    clipped_ang = max(min(true_ang, 1.0), -1.0)
    return clipped_ang


def concat_all_cameras(data, condition_lambda, keep_percent, drop_camera=''):
    """
    Concatenates left, right, and center paths and angles, shifting the left/right angles by `angle_shift`.

    An example of the usage of the condition lambda is if you want to remove 90% of the samples in a dataset
    where the steering angles are close to zero, the code would look like this:

        images, angles = concat_all_cameras(
            data=data,
            angle_shift=0.1,
            condition_lambda=lambda x: abs(x) < 1e-5,
            keep_percent=0.1
          )

    Note that the lambda should return True for the values you would like to filter.

    :param data: Dictionary containing ['angles', 'center', 'left', 'right']
    :param condition_lambda: Condition by which to keep data.
    :param keep_percent: Percent of data to keep where `condition_lambda` is true.
    :param drop_camera: Identifies a camera to not include load into the set.
    :return: Tuple containing (paths, angles)
    """
    # Remove n% of the frames where the steering angle is close to zero
    ims, angs = keep_n_percent_of_data_where(
        data=np.array([data['center'], data['left'], data['right']]).T,
        values=data['angles'],
        condition_lambda=condition_lambda,
        percent=keep_percent
      )

    center = np.array([im for im in ims[..., 0]])
    left = np.array([im for im in ims[..., 1]])
    right = np.array([im for im in ims[..., 2]])

    transformed_left_angs = np.array([transform_ang(ang, 'left') for ang in angs])
    transformed_right_angs = np.array([transform_ang(ang, 'right') for ang in angs])

    if drop_camera == 'left':
        filtered_paths = np.concatenate((center, right), axis=0)
        filtered_angs = np.concatenate((angs, transformed_right_angs), axis=0)
    elif drop_camera == 'right':
        filtered_paths = np.concatenate((center, left), axis=0)
        filtered_angs = np.concatenate((angs, transformed_left_angs), axis=0)
    elif drop_camera == 'both':
        filtered_paths = center
        filtered_angs = angs
    elif drop_camera == '':
        filtered_paths = np.concatenate((center, right, left), axis=0)
        filtered_angs = np.concatenate((angs, transformed_right_angs, transformed_left_angs), axis=0)
    else:
        raise ValueError('`drop_camera` must be "left", "right", or "both".')

    # Modify the steering angles of the left and right cameras's images to simulate
    # steering back towards the middle. Aggregate all sets into one.
    return filtered_paths, filtered_angs


def split_data(features, labels, test_size=0.2, shuffle_return=True):
    """
    Splits the dataset and labels into training and testing sets, with proportions (1-test_size) and test_size.

    :param features: The dataset to split
    :param labels: The labels for the dataset
    :param test_size: The proportion of the dataset to siphon into the test set
    :param shuffle_return: If True, shuffles the dataset and labels before splitting
    :return: (X_train, X_test, y_train, y_test)
    """
    if shuffle_return:
        features, labels = shuffle(features, labels)
    return train_test_split(features, labels, test_size=test_size)


def process_image(im):
    """
    Crop image, convert to HSV, and resize.

    :param im: Image to normalize
    :return: Normalized image with shape (h, w, ch)

    :type im: np.ndarray with shape (h, w, 3)
    :rtype: np.ndarray with shape (h, w, ch)
    """
    assert im.ndim == 3 and im.shape[2] == 3, 'Must be a BGR image with shape (h, w, 3)'

    if _color_space is not None:
        im = cv2.cvtColor(im, _color_space['dst'])
    if _gradients['cond']:
        im = cv2.Sobel(im[..., _gradients['channel']], cv2.CV_64F, 1, 1, ksize=11)
        im = abs(im)
    if _src is not None and _dst is not None:
        try:
            if process_image.M is None:
                process_image.M = cv2.getPerspectiveTransform(_src, _dst)
        except AttributeError:
            process_image.M = cv2.getPerspectiveTransform(_src, _dst)

        im = cv2.warpPerspective(im, process_image.M, (im.shape[1], im.shape[0]))
    if _reshape['src'] is not None and _reshape['dst'] is not None:
        im = cv2.resize(im, _reshape['dst'])

    k = np.random.choice((1, 3, 5))
    im = cv2.GaussianBlur(im, (k, k), 0)
    return im


def rectify_image(im):
    if _reshape['src'] is not None and _reshape['dst'] is not None:
        im = cv2.resize(im, _reshape['src'])
    if _src is not None and _dst is not None:
        try:
            if rectify_image.M is None:
                rectify_image.M = cv2.getPerspectiveTransform(_dst, _src)
        except AttributeError:
            rectify_image.M = cv2.getPerspectiveTransform(_dst, _src)

        im = cv2.warpPerspective(im, rectify_image.M, (im.shape[1], im.shape[0]))
    if _color_space is not None and not _gradients['cond'] and im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, _color_space['src'])
    return im


@n_images  # Decorator to generalize single image method to multiple images
def flip_image(image, angle):
    """
    Mirrors the image from left to right and flips the sign of the angle.

    :param image: Image to flip
    :param angle: Angle to flip
    :return: (flipped_images, flipped_angle)
    """
    flipped = cv2.flip(image, 1)
    if flipped.ndim == 2:
        flipped = np.expand_dims(flipped, -1)
    return flipped, -angle


def add_random_shadow(im):
    """
    Adds a random shadow to the image. Must be a 2D numpy.ndarray.

    If you would like to add a shadow to a color image, run this function over the
    brightness channel only. E.G. convert to HSV and use this function on the V channel.

    :param im: The image to shadow. Must be only the brightness channel
    :return: 2D array with a random shadow and augmented total brightness.
    """
    assert im.ndim == 3, 'Image must have dimensions (h, w, ch)'

    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    val = im[..., 2]

    h, w, ch = im.shape
    val = val.astype(np.float32)

    # Define line to create shadow on by creating an image mask
    top_y, bot_y = np.random.randint(2*w//10, 8*w//10, size=2)
    left_x, right_x = np.random.randint(2*h//10, 8*h//10, size=2)

    XX, YY = np.mgrid[0:h, 0:w]
    shadow = np.zeros_like(val, dtype=np.float32)

    # Randomly create a vertical or horizontal mask
    if np.random.choice(['vertical', 'horizontal']) == 'vertical':
        shadow[XX*(bot_y-top_y) - h*(YY-top_y) >= 0.0] = 1.0
    else:
        shadow[(XX-left_x)*(0-w) - (right_x-left_x)*(YY-w) >= 0.0] = 1.0

    # Randomly choose a side of the line and darken it
    mask = shadow == np.random.randint(0, 2)
    val[mask] *= np.random.uniform(0.5, 0.9)

    # Randomly augment total brightness
    val *= np.random.uniform(0.5, 1.2)

    # Clip values that are >255
    val[val > 255.] = 255.

    im[..., 2] = val
    im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
    return im.astype(np.uint8)


def random_image_shift(im, ang, x_shift=40, y_shift=3, per_pix_adj=5e-3):
    """
    Randomly shift the image left/right and up/down.

    :param im: The image
    :param ang: The corresponding steering angle.
    :param x_shift: Max number of pixels to shift the image left/right.
    :param y_shift: Max number of pixels to shift the image up/down.
    :param per_pix_adj: Adjust the steering angle by this amount per pixel of horizontal shift.
    :return: (augmented image, augmented steering angle)
    """
    h, w, _ = im.shape
    xshift = np.random.randint(-x_shift, x_shift+1)
    yshift = np.random.randint(-y_shift, y_shift+1)

    src = np.array([[0, 0], [w, 0], [w, h]]).astype(np.float32)
    dst = np.array([
        [0 + xshift, 0 + yshift],
        [w + xshift, 0 + yshift],
        [w + xshift, h + yshift]
      ]).astype(np.float32)
    M = cv2.getAffineTransform(src, dst)

    shifted_im = cv2.warpAffine(im, M, (w, h))
    shifted_ang = ang + x_shift*per_pix_adj
    return shifted_im, shifted_ang


def random_image_rotation(im, rotation=1, scale=2e-3):
    """
    Randomized rotation and scaling of an image. May help make the model more robust to jittering of the camera.

    :param im: The image
    :param rotation: Max number of pixels +/- to rotate.
    :param scale: Max scaling factor +/- to zoom.
    :return: Transformed image
    """
    h, w, _ = im.shape
    M = cv2.getRotationMatrix2D(
        (h // 2, w // 2),
        np.random.uniform(-rotation, rotation),
        np.random.uniform(1.0 - scale, 1.0 + scale)
      )
    return cv2.warpAffine(im, M, (w, h))


@n_images  # Decorator to generalize single image method to multiple images
def augment_image(image, value, prob, im_normalizer=process_image):
    """
    Augments an image and steering angle with probability `prob`.

    This technique randomly adjusts the brightness, occludes the image with 30 random black squares,
    and slightly rotates, shifts and scales the image. These augmentations are meant to make the
    model more robust to conditions different to those in the training set.

    :param image: The image to augment
    :param value: The steering angle associated with the image
    :param prob: The probability of augmenting the image
    :param im_normalizer: Function to normalize the image
    :return: Tuple with (augmented_image, augmented_value)

    :type image: np.ndarray
    :type value: float
    :type prob: float [0.0, 1.0]
    :type im_normalizer: function
    :rtype: tuple with values (augmented_image, augmented_value)
    """
    assert image.ndim == 3, 'Image must have dimensions (h, w, ch)'

    h, w, color_channels = image.shape

    # Flip the image and angle half the time. Effectively doubles the size of
    # the dataset while balancing the left and right turn proportions.
    if np.random.uniform(0.0, 1.0) < 0.5:
        image, value = cv2.flip(image, 1), -value

    # Return un-augmented image and value with probability (1-prob)
    if np.random.uniform(0.0, 1.0) > prob:
        image = im_normalizer(image)
        return image, value

    # Shifts/Affine transforms
    shadowed = add_random_shadow(image)
    rotated = random_image_rotation(shadowed)
    shifted, value = random_image_shift(rotated, value)

    augmented = im_normalizer(shifted)

    # Ensure there is a color channel
    if augmented.ndim == 2:
        augmented = np.expand_dims(augmented, -1)
    return augmented, value


def val_augmentor(ims, vals):
    """
    Normalizes images/vals into first set, flips and concats into second set, concats both sets, and returns.

    :param ims: Images to normalize/flip
    :param vals: Angles to normalize/flip
    :return: (normalized/flipped images, normalized/flipped angles)
    """
    normalized = np.array([process_image(im) for im in ims])
    flipped_ims, flipped_vals = flip_image(normalized, vals)
    return np.concatenate((normalized, flipped_ims), axis=0), \
           np.concatenate((vals, flipped_vals), axis=0)


def batch_generator(ims, angs, batch_size, augmentor, kwargs={}, validation=False):
    """
    Continuously generates batches from the provided images paths and angles.

    This method follows this process. Generates a batch of size `batch_size` in sequential order through
    the data. It is important that the data is shuffled prior to being fed into the generator. If the
    dataset is not evenly divisible by the batch size, their will be an orphan batch at the end. To
    address this, data is randomly selected to fill the batch. This option can be disabled by setting
    `validation` to True. Once the batch has been constructed, the images are loaded from the image paths
    using `cv2.imread`. Note that this means the images will be read in BGR format. Lastly, the images
    and angles are fed into the provided augmentation function, `augmentor`, and then yielded.

    :param ims: The filepaths to the images
    :param angs: The steering angles corresponding to the image paths
    :param batch_size: The size of batches to generate
    :param augmentor: A function which takes inputs (images, angles, **kwargs)
                      and returns (images, angles)
    :param kwargs: A dictionary containing any additional argument-value pairs for the
                   `augmentor` function
    :param validation: A boolean indicating whether or not to expand the orphan batch
                       from the generator. See description above.
    :return: Generator producing an infinite number of batches adhering to the above policies
    """
    n_obs = ims.shape[0]
    assert n_obs == angs.shape[0], 'Different # of data and labels.'

    while True:
        # ims, angs = shuffle(ims, angs)
        batch_starts = np.arange(0, n_obs, batch_size)

        # Decay the augmentation probability on every full pass through the data
        if 'prob' in kwargs.keys():
            kwargs['prob'] *= 0.98

        for batch in np.random.permutation(batch_starts):
            next_idx = batch + batch_size
            batch_x = ims[batch:min(next_idx, n_obs), ...]
            batch_y = angs[batch:min(next_idx, n_obs), ...]

            # Ensure consistent batch size by adding random images to the last
            # batch iff n_obs % batch_size != 0.
            if next_idx > n_obs and not validation:
                rand_idx = np.random.randint(0, n_obs-1, next_idx - n_obs)
                batch_x = np.concatenate((batch_x, ims[rand_idx, ...]), axis=0)
                batch_y = np.concatenate((batch_y, angs[rand_idx, ...]), axis=0)

            # Load the images from their paths
            loaded_ims = []
            for im_path in batch_x:
                im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
                if im.shape != (160, 320, 3):
                    im = cv2.resize(im, (320, 160))
                loaded_ims.append(im)
            batch_x = np.array(loaded_ims)

            # Augment the images with the given function
            batch_x, batch_y = augmentor(batch_x, batch_y, **kwargs)
            yield batch_x, batch_y
