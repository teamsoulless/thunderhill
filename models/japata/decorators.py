import numpy as np
import time
from functools import update_wrapper


def decorator(d):
    """Updates a decorated function's documentation sting with that
       of its un-decorated function."""
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d


def disabled(f):
    """This decorator is used to easily disable other decorators in a long
       program. E.G. if you have hundreds of functions using 'timeit' and
       not longer want the overhead, just put 'timeit = disabled' at the
       start of the program."""
    return f


@decorator
def n_images(f):
    """
    Generalizes a function that acts on an image/angle pair to multiple images/angles.

    Given a function that takes in a single image and steering angle, this decorator will
    generalize the function to act on batches of images and angles.

    :param f: The function to generalize.
    :return: A function which will apply `f` to a batch of images/angles
    """
    def _f(images, values, *args, **kwargs):
        assert isinstance(images, np.ndarray), '`images` must be a np.ndarray'
        assert isinstance(values, np.ndarray), '`values` must be a np.ndarray'
        assert images.ndim == 4, '`images` must be a 4d np.ndarray with dims (n_obs, h, w, ch), not %r.' % images.shape

        n_obs = images.shape[0]
        assert n_obs == values.shape[0], 'Different # of data and values.'

        new_ims, new_vals = [], []
        for i in range(n_obs):
            im, val = f(images[i, ...], values[i, ...], *args, **kwargs)
            new_ims.append(im)
            new_vals.append(val)
        return np.array(new_ims), np.array(new_vals)
    return _f


@decorator
def timeit(f):
    """
    Stores the last runtime of the function in `f.last_runtime`, and the average runtime in `f.avg_runtime`.

    :param f: The function to time.
    :return: The same function, but with runtime statistics.
    """
    runtimes = []

    def _f(*args, **kwargs):
        t0 = time.clock()

        result = f(*args, **kwargs)

        _f.last_runtime = time.clock() - t0
        runtimes.append(_f.last_runtime)
        _f.avg_runtime = np.mean(runtimes)
        return result
    return _f
