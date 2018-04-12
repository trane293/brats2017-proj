"""
==========================================================
            Generate Database with Bounding Boxes
==========================================================
AUTHOR: Anmol Sharma
AFFILIATION: Simon Fraser University
             Burnaby, BC, Canada
PROJECT: Analysis of Brain MRI Scans for Management of
         Malignant Tumors
COLLABORATORS: Anmol Sharma (SFU)
               Prof. Ghassan Hamarneh (SFU)
               Dr. Brian Toyota (VGH)
               Dr. Mostafa Fatehi (VGH)
DESCRIPTION: This file uses the previously generated data
             (using create_hdf5_file.py) and generates
             bounding box based labels for each patient.

             The default annotations that are provided are
             segmentation masks. This script uses those masks
             to find the minimum volume bounding box and stores
             the coordinates in a new hdf5 file.
LICENCE: Proprietary for now.
"""

import h5py
from code.modules.configfile import config
import numpy as np
import optparse
import logging
from code.modules.mischelpers import *
import os

logging.basicConfig(level=logging.DEBUG)

try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

parser = optparse.OptionParser()
parser.add_option('--l', '--include-lgg',
                  dest="include_lgg",
                  default=True
                  )

# do we want bounding box data or segmentation data?
parser.add_option('--gt', '--ground-truth',
                  dest="ground_truth",
                  default='bbox',
                  type='str'
                  )

# training split
parser.add_option('--ts', '--training-split',
                  dest="training_split",
                  default=90,
                  type=int
                  )

parser.add_option('--s', '--shuffle',
                  dest="shuffle_data",
                  default=True
                  )

options, remainder = parser.parse_args()

logger.info('Opening previously generated HDF5 file.')

# open the file in append mode, so that we can append the new database
filename = os.path.join(os.sep.join(config['hdf5_filepath_prefix'].split(os.sep)[0:-1]), 'BRATS_splits.h5')

hdf5_file = h5py.File(config['hdf5_filepath_prefix'], 'r')
hdf5_file_out = h5py.File(filename, 'w')


if u"x_train" in hdf5_file_out.keys():
    logger.warn('x_train database already exists!')
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file_out["x_train"]

if u"y_train" in hdf5_file_out.keys():
    logger.warn('y_train database already exists!')
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file_out["y_train"]

if u"x_test" in hdf5_file_out.keys():
    logger.warn('x_test database already exists!')
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file_out["x_test"]

if u"y_test" in hdf5_file_out.keys():
    logger.warn('y_test database already exists!')
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file_out["y_test"]

logger.info('Splitting the data using {}/{} ratio.'.format(options.training_split, 100-options.training_split))

if options.include_lgg == True:
    logger.info('LGG data will be including in the x_train dataset.')
    logger.info('Testing data will be sampled uniformly from the HGG/LGG dataset.')
    train_split_hgg = (config['train_shape_hgg'][0] * options.training_split) / 100
    train_split_lgg = (config['train_shape_lgg'][0] * options.training_split) / 100

    train_dataset_shape = train_split_hgg + train_split_lgg
    test_dataset_shape = config['train_shape_hgg'][0] - train_split_hgg + config['train_shape_lgg'][0] - train_split_lgg
else:
    logger.info('Creating x_train only using HGG data')
    train_split_hgg = (config['train_shape_hgg'][0] * options.training_split) / 100

    train_dataset_shape = train_split_hgg
    test_dataset_shape = config['train_shape_hgg'][0] - train_split_hgg


# create x_train dataset
x_train_shape = list(config['train_shape_hgg'])
x_train_shape[0] = train_dataset_shape

hdf5_file_out.create_dataset("x_train", x_train_shape, np.int16)
hdf5_file_out.create_dataset("training_data_pat_name", (x_train_shape[0],), dtype="S100")

# create x_test dataset
x_test_shape = list(config['train_shape_hgg'])
x_test_shape[0] = test_dataset_shape

hdf5_file_out.create_dataset("x_test", x_test_shape, np.int16)
hdf5_file_out.create_dataset("testing_data_pat_name", (x_test_shape[0],), dtype="S100")

if options.ground_truth == 'seg':
    logger.info('Ground truth will be segmentation masks')

    # create y_train
    y_train_shape = list(config['train_segmasks_shape_hgg'])
    y_train_shape[0] = train_dataset_shape
    hdf5_file_out.create_dataset("y_train", y_train_shape, np.int16)

    # create y_test as well
    y_test_shape = list(config['train_segmasks_shape_hgg'])
    y_test_shape[0] = test_dataset_shape
    hdf5_file_out.create_dataset("y_test", y_test_shape, np.int16)

else:
    logger.info('Ground truth will be bounding boxes')
    y_train_shape = [train_dataset_shape, 6]
    hdf5_file_out.create_dataset("y_train", y_train_shape, np.int16)

    y_test_shape = [test_dataset_shape, 6]
    hdf5_file_out.create_dataset("y_test", y_test_shape, np.int16)

# add some info to this dataset so that we know in the future what we were working with.
hdf5_file_out['x_train'].attrs['info'] = 'contains HGG and LGG samples, {}/{} percent split of data'.format(options.training_split, 100-options.training_split)

pat_range_hgg = range(0, train_split_hgg)
pat_range_lgg = range(0, train_split_lgg)

if options.shuffle_data == True:
    logger.info('Shuffle is true.')
    logger.info('Data will be read randomly')
    import random

    random.seed(3)
    random.shuffle(pat_range_hgg)
    random.shuffle(pat_range_lgg)

# populating x_train with HGG data first
run_count = 0
logger.info('Running on HGG Data.')
for pat in pat_range_hgg: # 210 patients in HGG

    hdf5_file_out['x_train'][run_count] = hdf5_file['training_data_hgg'][pat]
    hdf5_file_out['training_data_pat_name'][run_count] = hdf5_file['training_data_hgg_pat_name'][pat]

    if options.ground_truth == 'seg':
        hdf5_file_out['y_train'][run_count] = hdf5_file['training_data_segmasks_hgg'][pat]
    else:
        rect_obj = bbox_3D(hdf5_file['training_data_segmasks_hgg'][pat, ...], tol=0.1)
        hdf5_file_out['y_train'][run_count] = rect_obj.list_view
    logger.debug('Currently at: {}'.format(run_count))
    run_count += 1

if options.include_lgg == True:
    # populating x_train with HGG data first
    logger.info('Running on LGG Data.')
    for pat in pat_range_lgg:  # 75 patients in LGG
        hdf5_file_out['x_train'][run_count] = hdf5_file['training_data_lgg'][pat]
        hdf5_file_out['training_data_pat_name'][run_count] = hdf5_file['training_data_lgg_pat_name'][pat]

        if options.ground_truth == 'seg':
            hdf5_file_out['y_train'][run_count] = hdf5_file['training_data_segmasks_lgg'][pat]
        else:
            rect_obj = bbox_3D(hdf5_file['training_data_segmasks_lgg'][pat, ...], tol=0.1)
            hdf5_file_out['y_train'][run_count] = rect_obj.list_view
        logger.debug('Currently at: {}'.format(run_count))
        run_count += 1

logger.info('Closing HDF5 File.')
hdf5_file.close()
hdf5_file_out.close()