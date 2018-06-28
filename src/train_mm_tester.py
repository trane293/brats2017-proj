"""
==========================================================
                        Train Models
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
DESCRIPTION: This file serves the purpose of loading one of
             pre-defined networks from the defmodel folder
             and training it. The script provides modularity
             in the sense that it can be used to load any
             of the defined models in the defmodel folder.

             Check the defmodel  folder on how to define one
             such model file, complete wtth someof the mandatory
             functions that need to  be defined in it as well.
LICENCE: Proprietary for now.
"""
import platform
import logging

logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

# to make the code portable even on cedar,you need to add conditions here
node_name = platform.node()
if node_name == 'XPS15' or 'cs-mial-31' in node_name:
    # this is my laptop, so the cedar-rm directory is at a different place
    # disable GPU
    logger.info('Disabling GPU!')
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # -1 !!!!

import h5py
import sys
from modules.configfile import config
from modules.training_helpers import *
import cPickle as pickle
import numpy as np
import random as random
random.seed(1337)
np.random.seed(1337)
import importlib
import optparse
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

parser = optparse.OptionParser()
parser.add_option('--dm', '--defmodelfile',
                  dest="defmodelfile",
                  default='multimodal',
                  type='str',
                  help='Define the name of module from defmodel directory to load the model definition from'
                  )

parser.add_option('--o', '--out-name',
                  dest="output_name",
                  default='multimodal_main.h5',
                  type='str',
                  help='Define the name of the file that was generated after training was finished. Check model-snapshots'
                  )

parser.add_option('--e', '--epochs',
                  dest="epochs",
                  default=10,
                  type='int',
                  help='Number of epochs to train with'
                  )

parser.add_option('--b', '--batch-size',
                  dest="batch_size",
                  default=2,
                  type='int',
                  help='Batch size to train with'
                  )

parser.add_option('--h', '--hdf5',
                  dest="hdf5_filepath",
                  default=None,
                  type='str',
                  help='HDF5 filepath in case loading from a non-standard location'
                  )


options, remainder = parser.parse_args()

if options.output_name is None:
    logger.info('No output name defined, using default values')
    options.output_name = os.path.join(config['model_snapshot_location'], 'model_default.h5')
    logger.info('Name of output file: {}'.format(options.output_name))
else:
    options.output_name = os.path.join(config['model_snapshot_location'], options.output_name)
    logger.info('Name of output file: {}'.format(options.output_name))


# -------------------------------------------------------------------------------------
# get model file
# -------------------------------------------------------------------------------------
# open the model file module
modeldefmodule = importlib.import_module('defmodel.'+options.defmodelfile, package=None)

# get the model
# inp_shape = tuple(config['patch_input_shape'])
inp_shape = (4, None, None, None)


# TODO: Implement get_model in multimodal.py
model = modeldefmodule.get_model(inp_shape=inp_shape) # (4, x, y, z)



