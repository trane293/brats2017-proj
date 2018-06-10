import numpy as np
import random, itertools


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


def remove_sequence(x_data):
    # randomly remove a sequence from the x_data
    # remember nothing needs to be for the y_data, since that is the ground truth and we want network to learn it.
    # for normalized data with zero mean and unit variance, this is equivalent to imputing the
    # sequence with all zeros.
    # Set all values of a random sequence = 0

    # assuming a batch will be coming with first dimension = batch size. So we permute for all examples in this batch
    assert (len(x_data.shape) > 4), 'Batch size incorrect'
    sequences = [0,1,2,3]

    for curr_eg in range(x_data.shape[0]):
        r = random.choice([0, 1])
        if r == 1:
            # how many sequences to remove
            hm = random.randint(1, 3)
            rm_seq = random.sample(sequences, hm)
            for curr_seq in rm_seq:
                x_data[curr_eg,curr_seq,] = 0.0

    return x_data


def augment_data(x_data, y_data, augment=None):
    # assuming a batch will be coming with first dimension = batch size. So we permute for all examples in this batch
    assert (len(x_data.shape) > 4) and (len(y_data.shape) > 4), 'Batch size incorrect'
    if 'permute' in augment:
        for curr_eg in range(x_data.shape[0]):
            x_data[curr_eg,], y_data[curr_eg,] = random_permutation_x_y(x_data[curr_eg,], y_data[curr_eg,])
    if 'remove_seq' in augment:
        x_data = remove_sequence(x_data)

    return x_data, y_data
