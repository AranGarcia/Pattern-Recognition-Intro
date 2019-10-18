import numpy as np
from skimage import io

_IMG = None


def init(fname):
    global _IMG
    _IMG = io.imread(fname)


def get_class_samples(centers, apc, sd=10):
    '''
    Given an image file path, this function returns neighboring pixels around
    chosen centers, using a normal distribution around the x,y axis of the
    image with standard deviation of sd.

    get_img_samples() -> pix, classes, coords
    '''
    try:
        _IMG.shape
    except Exception:
        raise ValueError('Image not intialized.')
    # R, G, B
    class_samples = []
    positions = []
    for i, c in enumerate(centers):
        # Sample coordinates
        x_arr = np.random.normal(loc=c[1], scale=sd, size=apc)
        y_arr = np.random.normal(loc=c[0], scale=sd, size=apc)
        x_arr = np.ndarray.astype(x_arr, int)
        y_arr = np.ndarray.astype(y_arr, int)

        # Delete out of bound pixels
        in_bounds_x = np.logical_and(x_arr >= 0, x_arr < _IMG.shape[0])
        in_bounds_y = np.logical_and(y_arr >= 0, y_arr < _IMG.shape[1])
        in_bounds_coords = np.logical_and(in_bounds_x, in_bounds_y)
        x_arr = x_arr[in_bounds_coords]
        y_arr = y_arr[in_bounds_coords]

        class_samples.append(_IMG[x_arr, y_arr])
        positions.append(np.stack((x_arr, y_arr), axis=-1))

    samps = np.concatenate(class_samples)
    # Class vector (same rows as samples_per_class)
    class_vector = np.repeat(
        np.arange(1, len(centers) + 1),
        [arr.shape[0] for arr in class_samples])

    return samps, class_vector, np.concatenate(positions)


def get_sample(coord):
    global _IMG
    try:
        return _IMG[coord[1], coord[0]]
    except Exception as e:
        raise ValueError(f'Unable to sample image: {e}')
