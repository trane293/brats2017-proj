import sys
sys.path.append('..')
from modules.training_helpers import *
from modules.configfile import config
from defmodel.multimodal import embedding_distance
from modules.vizhelpercode import viewArbitraryVolume
import logging
import random as random
import optparse
import cPickle as pickle
import h5py
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt

random.seed(1337)
np.random.seed(1337)

logging.basicConfig(level=logging.DEBUG)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)


parser = optparse.OptionParser()
parser.add_option('--dm', '--defmodelfile',
                  dest="defmodelfile",
                  default='cnn_patches',
                  type='str'
                  )

parser.add_option('--g', '--grade',
                  dest="grade",
                  default='HGG',
                  type='str'
                  )

parser.add_option('--o', '--out-name',
                  dest="output_name",
                  default='cnn_patches_v1.h5',
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
batch_size = 10
augment = ['permute', 'remove_seq']
train_gen = generate_patch_batches(X=training_data, Y=training_data_segmasks,
                                   t_i=train_indices, mean_var=mean_var, batch_size=batch_size, debug_mode=False,
                                   applyNorm=False, augment=augment, generate_list=True)

test_gen = generate_patch_batches(X=training_data, Y=training_data_segmasks,
                                  t_i=test_indices, mean_var=mean_var, batch_size=batch_size, debug_mode=True,
                                  applyNorm=False, augment=augment)

# def embedding_distance(y_true, y_pred):
#     return K.var(y_pred, axis=1)

custom_obj = {
    'embedding_distance': embedding_distance
}
model = load_model('/home/anmol/model.h5', custom_objects=custom_obj)

count = 0
for x_patches, y_patches in train_gen:
    # print(x_patches.shape, y_patches.shape)

    viewArbitraryVolume(x_patches[0][0,0])
    viewArbitraryVolume(x_patches[1][0,0])
    viewArbitraryVolume(x_patches[2][0,0])
    viewArbitraryVolume(x_patches[3][0,0])

    viewArbitraryVolume(y_patches[0][0,0])
    viewArbitraryVolume(y_patches[1][0,0])
    viewArbitraryVolume(y_patches[2][0,0])

    if count == 0:
        break
    count += 1
print('Hello')

model.fit_generator(train_gen, steps_per_epoch=3,
                    epochs=1, verbose=1, callbacks=None,
                    validation_data=test_gen, validation_steps=3,
                    class_weight=None, max_queue_size=100, workers=1,
                    use_multiprocessing=True, shuffle=False, initial_epoch=0)

print('Hello')





# count = 0
# for x_patches, y_patches in test_gen:
#     print(x_patches.shape, y_patches.shape)
#     if count > 20:
#         break


