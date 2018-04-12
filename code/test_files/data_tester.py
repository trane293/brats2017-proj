"""
==========================================================
                Test HDF5 Database Builder
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
DESCRIPTION: This test file is used to test whether the HDF5
             database file is created correctly or not. It
             simply compares the original image file with the
             one contained in the HDF5 data store, to ensure that
             the intensity values are all same.

             The script also has the visualize test case, where
             the patients in the HDF5 store are visualized using
             a matplotlib subplot figure.
LICENCE: Proprietary for now.
"""

import h5py
import sys
sys.path.append('..')
from modules.configfile import config
from modules import dataloader
import numpy as np
import logging
import glob, os
import matplotlib.pyplot as plt

import optparse

parser = optparse.OptionParser()
parser.add_option('--viz', action="store", default=True, dest="viz")
parser.add_option('--in-depth', action="store", default=False, dest="in_depth")

options, remainder = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hdf5_file = h5py.File(config['hdf5_filepath_cropped'], mode='r')
hdf5_file.items()
hdf5_file = hdf5_file['preprocessed']

if options.viz == True:
    print('visualizing the slices..')
    data_to_viz = 'training_data_hgg'
    for pat in range(0, hdf5_file[data_to_viz].shape[0], 10):
        fig, ax = plt.subplots(nrows=2, ncols=5, squeeze=False, figsize=(20,10))
        ax = [i for ls in ax for i in ls]
        c = 0
        for a in ax:
            if c < 155:
                a.imshow(hdf5_file[data_to_viz][pat, 1,:,:,c], cmap='gray')
                a.axis('off')
                c += 15
            else:
                break
        plt.tight_layout()
        plt.suptitle('Patient {}'.format(pat))
        plt.show()


# # In depth testing
if options.in_depth == True:
    logger.info('Perfoming in depth tests..this may take some time.')
    for dataset_splits in glob.glob(os.path.join(config['data_dir_prefix'], '*')):
        if os.path.isdir(dataset_splits) and 'Validation' in dataset_splits:
            # VALIDATION data handler
            logger.info('currently loading Validation data.')
            count = 0
            # validation data does not have HGG and LGG distinctions
            for images, pats in dataloader.loadDataGenerator(dataset_splits, batch_size=config['batch_size'], loadSurvival=False, csvFilePath=None,
                                                                 loadSeg=False):
                logger.info('data equal?')
                val = np.array_equal(hdf5_file['validation_data'][count:count+config['batch_size'],...], images)
                logger.info(val)
                assert val == True

                t = 0
                for i in range(count, count + config['batch_size']):
                    logger.info('pat_name equal?')
                    val = hdf5_file['validation_data_pat_name'][i] == pats[t].split('/')[-1]
                    logging.info(val)
                    assert val == True

                    t += 1

                count += config['batch_size']

        else:
        # TRAINING data handler
            if os.path.isdir(dataset_splits) and 'Training' in dataset_splits:
                for grade_type in glob.glob(os.path.join(dataset_splits, '*')):
                    # there may be other files in there (like the survival data), ignore them.
                    if os.path.isdir(grade_type):
                        count = 0
                        logger.info('Currently loading Training data.')
                        for images, segmasks, pats in dataloader.loadDataGenerator(grade_type,                                                          batch_size=config['batch_size'], loadSurvival=False, csvFilePath=None, loadSeg=True):
                            logger.info('loading patient {} from {}'.format(count, grade_type))
                            if 'HGG' in grade_type:
                                logger.info('data equal?')
                                val = np.array_equal(hdf5_file['training_data_hgg'][count:count+config['batch_size'],...], images)
                                logger.info(val)
                                assert val == True

                                logger.info('segmasks equal?')
                                val = np.array_equal(hdf5_file['training_data_segmasks_hgg'][count:count+config['batch_size'], ...], segmasks)
                                logger.info(val)
                                assert val == True

                                t = 0
                                for i in range(count, count + config['batch_size']):
                                    logger.info('pat_name equal?')
                                    val = hdf5_file['training_data_hgg_pat_name'][i] == pats[t].split('/')[-1]
                                    logger.info(val)
                                    assert val == True
                                    t += 1
                            elif 'LGG' in grade_type:
                                logger.info('data equal?')
                                val = np.array_equal(hdf5_file['training_data_lgg'][count:count+config['batch_size'], ...], images)
                                logger.info(val)
                                assert val == True

                                logger.info('segmasks equal?')
                                val = np.array_equal(hdf5_file['training_data_segmasks_lgg'][count:count+config['batch_size'], ...], segmasks)
                                logger.info(val)
                                assert val == True

                                t = 0
                                for i in range(count, count + config['batch_size']):
                                    logger.info('pat_name equal?')
                                    val = hdf5_file['training_data_lgg_pat_name'][i] == pats[t].split('/')[-1]
                                    logger.info(val)
                                    assert val == True
                                    t += 1

                            count += config['batch_size']
# close the HDF5 file
hdf5_file.close()