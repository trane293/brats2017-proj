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
                  default='3dunet',
                  type='str'
                  )

parser.add_option('--g', '--grade',
                  dest="grade",
                  default='HGG',
                  type='str'
                  )

parser.add_option('--o', '--out-name',
                  dest="output_name",
                  default='3dunet_patches.h5',
                  type='str'
                  )

parser.add_option('--e', '--epochs',
                  dest="epochs",
                  default=10,
                  type='int'
                  )

parser.add_option('--b', '--batch-size',
                  dest="batch_size",
                  default=10,
                  type='int'
                  )

parser.add_option('--h', '--hdf5',
                  dest="hdf5_filepath",
                  default=None,
                  type='str'
                  )

options, remainder = parser.parse_args()

# --------------------------------------------------------------------------------------
# open required files
# --------------------------------------------------------------------------------------
# open mean and variance dictionary

mean_var = pickle.load(open(config['saveMeanVarFilepath'+options.grade.upper()], 'rb'))

# open new database with cropped images

if config['gen_patches_from'] == 'cropped':
    hdf5_file = h5py.File(config['hdf5_filepath_cropped'], mode='r')
    hdf5_file_g = hdf5_file['preprocessed']
else:
    if options.hdf5_filepath != None:
        logger.info('Using supplied HDF5 Filepath {}'.format(options.hdf5_filepath))
        hdf5_file = h5py.File(options.hdf5_filepath, mode='r')
    else:
        hdf5_file = h5py.File(config['hdf5_filepath_prefix'], mode='r')
    hdf5_file_g = hdf5_file['original_data']

# get all the HGG/LGG data
training_data = hdf5_file_g['training_data_'+options.grade.lower()]
training_data_segmasks = hdf5_file_g['training_data_segmasks_'+options.grade.lower()]

# ======================================================================================

# --------------------------------------------------------------------------------------
# split data into training and testing
# --------------------------------------------------------------------------------------
indices = list(range(0, training_data.shape[0]))
random.shuffle(indices) # in-place shuffling

train_end = int((len(indices) * config['data_split']['train']) / 100.0)
train_indices = indices[0:train_end]
test_indices = indices[train_end:]

# ======================================================================================

if options.output_name is None:
    logger.info('No output name defined, using default values')
    options.output_name = os.path.join(config['model_snapshot_location'], 'model_default.h5')
    logger.info('Name of output file: {}'.format(options.output_name))
else:
    options.output_name = os.path.join(config['model_snapshot_location'], options.output_name)
    logger.info('Name of output file: {}'.format(options.output_name))

if options.defmodelfile is None:
    logger.info('No defmodel file name defined, using default model (cnn)')
    options.defmodelfile = 'cnn'

# -------------------------------------------------------------------------------------
# get model file
# -------------------------------------------------------------------------------------
# open the model file module
modeldefmodule = importlib.import_module('defmodel.'+options.defmodelfile, package=None)

# get the model
# inp_shape = tuple(config['patch_input_shape'])
inp_shape = (4, None, None, None)
model = modeldefmodule.get_model(inp_shape=inp_shape) # (4, x, y, z)

# # compile
# model = modeldefmodule.compile_model(model)
# ======================================================================================

# --------------------------------------------------------------------------------------
# intialize some callbacks
# --------------------------------------------------------------------------------------
mc = ModelCheckpoint(os.path.join(config['model_checkpoint_location'],
                                  options.output_name.split('/')[-1]+'{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', verbose=1,
                                  save_best_only=False, save_weights_only=False, mode='auto', period=1)

reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.001)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

tb = TensorBoard(log_dir='./graph', histogram_freq=0,
          write_graph=True, write_images=True)
# ======================================================================================

c = 0
epochs = options.epochs
batch_size = options.batch_size
total_per_epoch_training = (len(train_indices) * config['num_patches_per_patient'] / batch_size)
total_per_epoch_testing = (len(test_indices) * config['num_patches_per_patient'] / batch_size)

train_gen = generate_patch_batches(X=training_data, Y=training_data_segmasks,
                                   t_i=train_indices, mean_var=mean_var, batch_size=batch_size)

test_gen = generate_patch_batches(X=training_data, Y=training_data_segmasks,
                                  t_i=test_indices, mean_var=mean_var, batch_size=batch_size)

history = model.fit_generator(train_gen, steps_per_epoch=total_per_epoch_training,
                    epochs=epochs, verbose=1, callbacks=[mc, reduceLR, tb],
                    validation_data=test_gen, validation_steps=total_per_epoch_testing,
                    class_weight=None, max_queue_size=20, workers=1,
                    use_multiprocessing=False, shuffle=False, initial_epoch=0)

modeldefmodule.save_model_with_hyper_and_history(model, history=history, name=options.output_name)



