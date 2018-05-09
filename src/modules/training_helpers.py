import numpy as np
from configfile import config
from scipy.ndimage.measurements import center_of_mass
import logging

logging.basicConfig(level=logging.DEBUG)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

def apply_mean_std(im, mean_var):
    """
    Supercedes the standardize function. Takes the mean/var  file generated during preprocessed data generation and
    applies the normalization step to the patch.
    :param im: patch of size  (num_egs, channels, x, y, z) or (channels, x, y, z)
    :param mean_var: dictionary containing mean/var value calculated in preprocess.py
    :return: normalized patch
    """

    # expects a dictionary of means and VARIANCES, NOT STD
    for m in range(0, 4):
        if len(np.shape(im)) > 4:
            im[:, m, ...] = (im[:, m, ...] - mean_var['mn'][m]) / np.sqrt(mean_var['var'][m])
        else:
            im[m, ...] = (im[m, ...] - mean_var['mn'][m]) / np.sqrt(mean_var['var'][m])

    return im


def check_valid(patch_coords):
    """
    Check if the selected coordinates are valid, and do not fall outside the scan region
    :param patch_coords: coordinates to validate
    :return: patch_coords if valid, otherwise None
    """

    xmin, xmax, ymin, ymax, zmin, zmax = patch_coords

    if xmin >= 0 and xmax < config['size_after_cropping'][0] and \
            ymin >= 0 and ymax < config['size_after_cropping'][1] and \
            zmin >= 0 and zmax < config['size_after_cropping'][2]:
        return patch_coords
    else:
        return None


def calculateCOM_STD(segmask):
    """
    Calculate Center of Mass (COM) of the whole tumor region weighted according to the subregion. The necrotic region
    gets the highest weight.
    :param segmask: segmentation mask containing different voxel labels
    :return: COM coordinates, standard deviations
    """
    seg_reweighted = np.copy(segmask)

    # brute force way to make sure the COM calculation is weighted correctly. We need more
    # weight on necrotic region, than edema.
    seg_reweighted[np.where(segmask == 1)] = 10  # necrotic, the most inner region, has highest weight
    seg_reweighted[np.where(segmask == 4)] = 9  # enhancing
    seg_reweighted[np.where(segmask == 3)] = 8  # non-enhancing
    seg_reweighted[np.where(segmask == 2)] = 7  # edema

    # calculate COM
    m_x, m_y, m_z = center_of_mass(seg_reweighted)

    x, y, z = np.where(segmask > 0)
    std_x = np.max(x) - np.min(x)
    std_y = np.max(y) - np.min(y)
    std_z = np.max(z) - np.min(z)

    return [m_x, m_y, m_z], [std_x, std_y, std_z]


def generate_patches(X, Y, t_i, mean_var, debug_mode=False):
    """
    Generator for generating patches to train. Make sure you specify samples_per_epoch in the
    Keras fit_generator function to num_patient * num_patches. That will ensure that the training
    goes through the whole patient data.
    :param X: HDF5 dataset containing patient studies
    :param Y: HDF5 dataset containing segmentation masks for the patients
    :param t_i: Indices to access data. These are generated outside the generator to specify which indices are training.
    :param mean_var: Dictionary containing mean and variance values for each modality. Applied before patches are extracted.
    :return: Yields batches indefinitely.
    """

    # get the required patch size
    patch_size_x, patch_size_y, patch_size_z = config['patch_size']

    # initialize the patches array to be used for training
    x_patches = np.empty(config['numpy_patch_size'])
    y_patches = np.empty((config['num_patches_per_patient'], config['patch_size'][0], config['patch_size'][1],
                         config['patch_size'][2]))
    if len(t_i) > 50:  # a very ad-hoc way of knowing this is probably called for training generation
        prefix = '[Training]'
    elif len(t_i) <= 50:  # a very ad-hoc way of knowing this is probably called for training generation
        prefix = '[Testing]'

    while 1:
        logger.debug(prefix + ' Iteration over all patient data starts')
        for _enum, t_idx in enumerate(t_i):
            logger.debug(prefix + ' Generating patches from Patient ID = {}, num = {}'.format(t_idx, _enum))

            x = apply_mean_std(X[t_idx], mean_var)
            y = Y[t_idx]

            com, std = calculateCOM_STD(y)

            # Generate patches
            for _t in range(0, config['num_patches_per_patient']):
                k = 0
                # not all proposals will be valid, hence keep generating patch coordinates until you find a valid one
                while k is not None:

                    # randomly sample cube center coordinates from multivariate gaussian
                    xc, yc, zc = np.random.multivariate_normal(mean=com,
                                                               cov=np.diag(np.array(std) * config['std_scale']))
                    xmin = int(xc) - (patch_size_x / 2)
                    xmax = int(xc) + (patch_size_x / 2)

                    ymin = int(yc) - (patch_size_y / 2)
                    ymax = int(yc) + (patch_size_y / 2)

                    zmin = int(zc) - (patch_size_z / 2)
                    zmax = int(zc) + (patch_size_z / 2)

                    patch_coords = [xmin, xmax, ymin, ymax, zmin, zmax]
                    t = check_valid(patch_coords)
                    if t is not None:
                        k = None
                if debug_mode == True:
                    from vizhelpercode import viewInMayavi
                    viewInMayavi(t, y) # use this for visualizing the patches

                # this condition is kind of useless here, since right now we only support the 'th' oredering fully.
                if config['data_order'] == 'th':
                    x_patches[_t,...] = x[:, t[0]:t[1], t[2]:t[3], t[4]:t[5]]
                    y_patches[_t,...] = y[t[0]:t[1], t[2]:t[3], t[4]:t[5]]
                else:
                    x_patches[_t, ...] = x[t[0]:t[1], t[2]:t[3], t[4]:t[5], :]
                    y_patches[_t, ...] = y[t[0]:t[1], t[2]:t[3], t[4]:t[5]]

            yield x_patches, y_patches


def generate_patch_batches(X, Y, t_i, mean_var, batch_size=10, debug_mode=False):
    '''
    Generate patch batches, apply augmentation, and create multiple masks for multi-class segmentation

    Tested: True

    :param X:
    :param Y:
    :param t_i:
    :param mean_var:
    :param batch_size:
    :param debug_mode:
    :return:
    '''
    while 1:
        for x_patches, y_patches in generate_patches(X, Y, t_i, mean_var=mean_var, debug_mode=debug_mode):

            for _t in range(0, x_patches.shape[0], batch_size):
                # add augmentation code here

                # yield batches
                x_batch = x_patches[_t:_t + batch_size, ...]
                y_batch = y_patches[_t:_t + batch_size, ...]

                # we need to convert y_batch into a numpy array with n_labels channels, for training a multi-class
                # segmentation network

                y_batch_channel_wise = np.zeros((y_batch.shape[0], config['num_labels'],
                                                 y_batch.shape[1], y_batch.shape[2], y_batch.shape[3]))

                labels = [1,2,4]

                for idx, i in enumerate(labels):
                    # inside the channel 'i' in y_batch_channel_wise, set all voxels which have label 'i' equal to 'i'.
                    # in the zeroth channel, find the voxels which are 1, and set all those corresponding voxels as 1
                    # there's no seperate  channel  for background class.
                    y_batch_channel_wise[:, idx, ...][np.where(y_batch == i)] = 1


                yield x_batch, y_batch_channel_wise
