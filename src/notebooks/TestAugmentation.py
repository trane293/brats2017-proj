# coding: utf-8

# In[6]:


import h5py
import sys
import numpy as np

sys.path.append('..')
from modules.configfile import config
import matplotlib.pyplot as plt
from modules.training_helpers import standardize
import cPickle as pickle

import random, itertools

def showBatch(val, modality=1, slice=30):
    if len(val.shape) > 4:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(val[0, modality, :, :, slice])
        ax[0].set_title('Patient 1')

        ax[1].imshow(val[1, modality, :, :, slice])
        ax[1].set_title('Patient 2')
        plt.show()
    else:
        plt.imshow(val[1, :, :, slice])
        plt.title('Patient 1')
        plt.show()

def convertChannelWise(seg):
    # we need to convert y_batch into a numpy array with n_labels channels, for training a multi-class
    # segmentation network

    y_batch_channel_wise = np.zeros((seg.shape[0], config['num_labels'],
                                     seg.shape[1], seg.shape[2], seg.shape[3]))

    labels = [1, 2, 4]

    for idx, i in enumerate(labels):
        # inside the channel 'i' in y_batch_channel_wise, set all voxels which have label 'i' equal to 'i'.
        # in the zeroth channel, find the voxels which are 1, and set all those corresponding voxels as 1
        # there's no seperate  channel  for background class.
        y_batch_channel_wise[:,idx, ...][np.where(seg == i)] = 1

    return y_batch_channel_wise


hdf5_file = h5py.File(config['hdf5_combined'], mode='r')
hg = hdf5_file['combined']
mean_var = pickle.load(open(config['saveMeanVarCombinedData'], 'rb'))

im = hg['training_data'][0:2, :, 50:150, 50:150, 50:150]
se = hg['training_data_segmasks'][0:2, 50:150,50:150,50:150]
se = convertChannelWise(se)

showBatch(im, slice=50)
showBatch(se, slice=50)

# # Perform data augmentation
from modules.augment import augment_data

for i in range(0, 10):
    x, y = augment_data(im, se)
    showBatch(x, modality=2, slice=50)
    showBatch(y, modality=2, slice=50)


def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)
