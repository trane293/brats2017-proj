import numpy as np
from configfile import config
from scipy.ndimage.measurements import _stats
import logging
import cPickle as pickle
from augment import augment_data

#from vizhelpercode import viewInMayavi, viewArbitraryVolume

logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

def sum(input, labels=None, index=None):
    count, sum = _stats(input, labels, index)
    return sum

def center_of_mass(input, labels=None, index=None):
    normalizer = sum(input, labels, index)
    grids = np.ogrid[[slice(0, i) for i in input.shape]]

    results = [sum(input * grids[dir].astype(float), labels, index) / normalizer
               for dir in range(input.ndim)]

    if np.isscalar(results[0]):
        return tuple(results)

    return [tuple(v) for v in np.array(results).T]


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


def standardize(images, findMeanVarOnly=True, saveDump=None, applyToTest=None):
    """
    This function standardizes the input data to zero mean and unit variance. It is capable of calculating the
    mean and std values from the input data, or can also apply user specified mean/std values to the images.

    :param images: numpy ndarray of shape (num_qg, channels, x, y, z) to apply mean/std normalization to
    :param findMeanVarOnly: only find the mean and variance of the input data, do not normalize
    :param saveDump: if True, saves the calculated mean/variance values to the disk in pickle form
    :param applyToTest: apply user specified mean/var values to given images. checkLargestCropSize.ipynb has more info
    :return: standardized images, and vals (if mean/val was calculated by the function
    """

    # takes a dictionary
    if applyToTest != None:
        logger.debug('Applying to test data using provided values')
        images = apply_mean_std(images, applyToTest)
        return images

    logger.info('Calculating mean value..')
    vals = {
        'mn': [],
        'var': []
    }
    for i in range(4):
        vals['mn'].append(np.mean(images[:, i, :, :, :]))

    logger.info('Calculating variance..')
    for i in range(4):
        vals['var'].append(np.var(images[:, i, :, :, :]))

    if findMeanVarOnly == False:
        logger.info('Starting standardization process..')

        for i in range(4):
            images[:, i, :, :, :] = ((images[:, i, :, :, :] - vals['mn'][i]) / float(vals['var'][i]))

        logger.info('Data standardized!')

    if saveDump != None:
        logger.info('Dumping mean and var values to disk..')
        pickle.dump(vals, open(saveDump, 'wb'))
    logger.info('Done!')

    return images, vals



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
    seg_reweighted[np.where(segmask == 1)] = 10.0  # necrotic, the most inner region, has highest weight
    seg_reweighted[np.where(segmask == 4)] = 9.0  # enhancing
    seg_reweighted[np.where(segmask == 3)] = 8.0  # non-enhancing
    seg_reweighted[np.where(segmask == 2)] = 7.0  # edema

    # calculate COM
    m_x, m_y, m_z = center_of_mass(seg_reweighted)

    x, y, z = np.where(segmask > 0)
    std_x = np.max(x) - np.min(x)
    std_y = np.max(y) - np.min(y)
    std_z = np.max(z) - np.min(z)

    return [m_x, m_y, m_z], [std_x, std_y, std_z]


def generate_patches(X, Y, t_i, mean_var, debug_mode=False, gen_name='Training', applyNorm=True):
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
    run_num = 0
    patch_size_x, patch_size_y, patch_size_z = config['patch_size']

    # initialize the patches array to be used for training
    x_patches = np.empty(config['numpy_patch_size'])
    y_patches = np.empty((config['num_patches_per_patient'], config['patch_size'][0], config['patch_size'][1],
                         config['patch_size'][2]))
    prefix = '[' + gen_name + ']'

    while 1:
        if run_num != 0 and run_num % 3 == 0:
            logger.debug('Increasing std_scale to {}'.format(config['std_scale']))
            std_scale = config['std_scale']
        else:
            logger.debug('Lowering std_scale to 1.0')
            std_scale = 1.0

        logger.warn(prefix + ' Iteration over all patient data starts')
        for _enum, t_idx in enumerate(t_i):
            logger.info(prefix + ' Generating patches from Patient ID = {}, num = {}'.format(t_idx, _enum))


            if applyNorm == True:
                x = apply_mean_std(X[t_idx], mean_var)
            else:
                x = X[t_idx]

            y = Y[t_idx]

            com, std = calculateCOM_STD(y)

            # Generate patches
            for _t in range(0, config['num_patches_per_patient']):
                k = 0
                # not all proposals will be valid, hence keep generating patch coordinates until you find a valid one
                while k is not None:

                    # randomly sample cube center coordinates from multivariate gaussian
                    xc, yc, zc = np.random.multivariate_normal(mean=com,
                                                               cov=np.diag(np.array(std) * std_scale))
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
                    viewInMayavi(t, y) # use this for visualizing the patches

                # this condition is kind of useless here, since right now we only support the 'th' oredering fully.
                if config['data_order'] == 'th':
                    x_patches[_t,...] = x[:, t[0]:t[1], t[2]:t[3], t[4]:t[5]]
                    y_patches[_t,...] = y[t[0]:t[1], t[2]:t[3], t[4]:t[5]]
                    if debug_mode == True:
                        viewArbitraryVolume(x_patches[_t], slice_idx=2, modality=1)
                        viewArbitraryVolume(y_patches[_t], slice_idx=2, modality=1)
                else:
                    x_patches[_t, ...] = x[t[0]:t[1], t[2]:t[3], t[4]:t[5], :]
                    y_patches[_t, ...] = y[t[0]:t[1], t[2]:t[3], t[4]:t[5]]

            printPercentages(y_patches)
            run_num += 1
            yield x_patches, y_patches


def printPercentages(patches):
    patches_shape = [a for a in np.shape(patches)]

    k = 1
    for i in patches_shape:
        k = k * i

    total_pixels = k

    # total pixels with label 1
    lab = np.where(patches == 1)
    lab = lab[0].shape[0]
    logger.debug('%age pixels with label 1 (Necrotic + Non-Enhancing) = {}'.format((lab*100.0)/total_pixels))

    # total pixels with label 2
    lab = np.where(patches == 2)
    lab = lab[0].shape[0]
    logger.debug('%age pixels with label 2 (Edema) = {}'.format((lab * 100.0) / total_pixels))

    # total pixels with label 4
    lab = np.where(patches == 4)
    lab = lab[0].shape[0]
    logger.debug('%age pixels with label 4 (Enhancing) = {}'.format((lab * 100.0) / total_pixels))


def generate_patch_batches(X, Y, t_i, mean_var, batch_size=10, debug_mode=False, gen_name='Training',
                           applyNorm=True, augment=None):
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
        for x_patches, y_patches in generate_patches(X, Y, t_i, mean_var=mean_var, debug_mode=debug_mode,
                                                     gen_name=gen_name, applyNorm=applyNorm):

            for _t in range(0, x_patches.shape[0], batch_size):

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

                # add augmentation code here
                if augment != None:
                    x_batch, y_batch_channel_wise = augment_data(x_batch, y_batch_channel_wise, augment=augment)


                yield x_batch, y_batch_channel_wise
