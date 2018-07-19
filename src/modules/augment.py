import numpy as np
import random, itertools
from configfile import config
import cPickle as pickle
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import affine_transform
from transformations import translation_matrix, shear_matrix, scale_matrix

random.seed(config['seed'])
np.random.seed(config['seed'])

mean_var = pickle.load(open(config['saveMeanVarCombinedData'], 'rb'))

def generate_permutation_keys():
    """
    This function returns a set of "keys" that represent the 48 unique rotations &
    reflections of a 3D matrix.
    Each item of the set is a tuple:
    ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)
    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    48 unique rotations & reflections:
    https://en.wikipedia.org/wiki/Octahedral_symmetry#The_isometries_of_the_cube
    """
    return set(itertools.product(
        itertools.combinations_with_replacement(range(2), 2), range(2), range(2), range(2), range(2)))


def random_permutation_key():
    """
    Generates and randomly selects a permutation key. See the documentation for the
    "generate_permutation_keys" function.
    """
    return random.choice(list(generate_permutation_keys()))


def permute_data(data_in, key):
    """
    Permutes the given data according to the specification of the given key. Input data
    must be of shape (n_modalities, x, y, z).
    Input key is a tuple: (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)
    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    """
    data = np.copy(data_in)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 3))
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if flip_x:
        data = data[:, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_z:
        data = data[:, :, :, ::-1]
    if transpose:
        for i in range(data.shape[0]):
            data[i] = data[i].T
    return data


def random_permutation_x_y(x_data, y_data):
    """
    Performs random permutation on the data.
    :param x_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :param y_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :return: the permuted data
    """
    key = random_permutation_key()
    return permute_data(x_data, key), permute_data(y_data, key)


def remove_sequence(x_data, epoch):
    # randomly remove a sequence from the x_data
    # remember nothing needs to be for the y_data, since that is the ground truth and we want network to learn it.
    # for normalized data with zero mean and unit variance, this is equivalent to imputing the
    # sequence with all zeros.
    # Set all values of a random sequence = 0

    # Warmup epochs = 10
    # only do something if we're past 10 epochs
    if epoch > 10:
        sequences = [0, 1, 2, 3]
        for curr_eg in range(x_data.shape[0]):
            # there is a 50% chance that a sequence will be removed
            chance = random.uniform(0, 1)
            if chance > 0.5:
                # we are going to remove one or many sequences
                # there is a 85% chance only one sequence is removed.
                # there is 10% chance two sequences are removed
                # there is 5% chance of three sequences being removed

                hm = random.uniform(0, 1)
                if hm < 0.85: # remove one sequence
                    rm_seq = random.sample(sequences, 1)
                elif hm >= 0.85 and  hm < 0.95: # remove two sequences
                    rm_seq = random.sample(sequences, 2)
                elif hm >= 0.95: # remove three sequences
                    rm_seq = random.sample(sequences, 3)

                for curr_seq in rm_seq:
                    x_data[curr_eg,curr_seq,] = 0.0

    return x_data

def add_noise(x_data):
    '''
    There is a 30% chance that a particular patch will be added noise to.
    :param x_data:
    :return:
    '''
    global mean_var
    for curr_eg in range(x_data.shape[0]):
        chance = random.uniform(0, 1)
        if chance > 0.7:
            mean_divider = random.uniform(1, 20)
            std_divider = random.uniform(15, 30)
            for each_mod in range(0, 4):
                x_data[curr_eg,each_mod,] = x_data[curr_eg,each_mod,] + np.random.normal(loc=mean_var['mn'][each_mod]/mean_divider,
                                                                                         scale=np.sqrt(mean_var['var'][each_mod])/std_divider,
                                                                                         size=np.shape(x_data[curr_eg,each_mod,]))
    return x_data


def add_blur(x_data):
    '''
    There is a 25% chance that a particular patch will be blurred.
    :param x_data:
    :return:
    '''
    for curr_eg in range(x_data.shape[0]):
        chance = random.uniform(0, 1)
        if chance < 0.25:
            sigma = random.uniform(0.3, 0.7)
            x_data[curr_eg] = gaussian_filter(x_data[curr_eg,], sigma=sigma)

    return x_data

def translate_data(x_data, y_data):
    '''
    There is a 25% chance that translation operation will be performed.
    :param x_data:
    :param y_data:
    :return:
    '''
    for curr_eg in range(x_data.shape[0]):
        chance = random.uniform(0, 1)
        if chance < 0.25:
            dx = random.randint(10, 40)
            dy = random.randint(10, 40)
            dz = random.randint(10, 40)
            T = translation_matrix([dx, dy, dz])

            # transform the x_data
            for each_mod in range(0,4):
                x_data[curr_eg, each_mod] = affine_transform(x_data[curr_eg, each_mod], T, order=1, prefilter=False)

            # transform the y_data
            for each_label in range(0,3):
                y_data[curr_eg, each_label] = affine_transform(y_data[curr_eg, each_label], T, order=1, prefilter=False)

    return x_data, y_data


def scale_data(x_data, y_data):
    '''
    There is a 25% chance that scaling operation will be performed.
    :param x_data:
    :param y_data:
    :return:
    '''
    for curr_eg in range(x_data.shape[0]):
        chance = random.uniform(0, 1)
        if chance < 0.25:
            origin = list(np.array(config['patch_size'], copy=True) / 2)
            factor = random.uniform(0.5, 1.5)
            S = scale_matrix(factor, origin=origin)

            # transform the x_data
            for each_mod in range(0, 4):
                x_data[curr_eg, each_mod] = affine_transform(x_data[curr_eg, each_mod], S, order=1, prefilter=False)

            # transform the y_data
            for each_label in range(0, 3):
                y_data[curr_eg, each_label] = affine_transform(y_data[curr_eg, each_label], S, order=1, prefilter=False)

    return x_data, y_data


def shear_data(x_data, y_data):
    '''
    There is a 25% chance that scaling operation will be performed.
    :param x_data:
    :param y_data:
    :return:
    '''
    for curr_eg in range(x_data.shape[0]):
        chance = random.uniform(0, 1)
        if chance < 0.25:
            angle = (random.random() - 0.5) * 4 * np.pi
            direct = np.random.random(3) - 0.5
            point = list(np.array(config['patch_size'], copy=True) / 2)
            normal = np.cross(direct, np.random.random(3))
            S = shear_matrix(angle, direct, point, normal)

            # transform the x_data
            for each_mod in range(0, 4):
                x_data[curr_eg, each_mod] = affine_transform(x_data[curr_eg, each_mod], S, order=1, prefilter=False)

            # transform the y_data
            for each_label in range(0, 3):
                y_data[curr_eg, each_label] = affine_transform(y_data[curr_eg, each_label], S, order=1, prefilter=False)

    return x_data,  y_data


def augment_data(x_data, y_data, augment=None, epoch=0):
    # assuming a batch will be coming with first dimension = batch size. So we permute for all examples in this batch
    assert (len(x_data.shape) > 4) and (len(y_data.shape) > 4), 'Batch size incorrect'
    if 'permute' in augment:
        for curr_eg in range(x_data.shape[0]):
            x_data[curr_eg,], y_data[curr_eg,] = random_permutation_x_y(x_data[curr_eg,], y_data[curr_eg,])
    if 'add_noise' in augment:
        x_data = add_noise(x_data)
    if 'add_blur' in augment:
        x_data = add_blur(x_data)
    if 'remove_seq' in augment:
        x_data = remove_sequence(x_data, epoch=epoch)
    if 'affine' in augment:
        x_data, y_data = translate_data(x_data, y_data)
        x_data, y_data = scale_data(x_data, y_data)
        x_data, y_data = shear_data(x_data, y_data)

    return x_data, y_data
