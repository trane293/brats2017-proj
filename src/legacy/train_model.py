from code.modules.configfile import config
import importlib
import optparse
import os
import logging
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

parser = optparse.OptionParser()
parser.add_option('--dm', '--defmodelfile',
                  dest="defmodelfile",
                  default=None,
                  type='str'
                  )
parser.add_option('--o', '--out-name',
                  dest="output_name",
                  default=None,
                  type='str'
                  )

options, remainder = parser.parse_args()

if options.output_name is None:
    logger.info('No output name defined, using default values')
    options.output_name = '/local-scratch/cedar-rm/scratch/asa224/model-snapshots/model_default.h5'
    logger.info('Name of output file: {}'.format(options.output_name))
else:
    options.output_name = os.path.join('/local-scratch/cedar-rm/scratch/asa224/model-snapshots/', options.output_name)
    logger.info('Name of output file: {}'.format(options.output_name))

if options.defmodelfile is None:
    logger.info('No defmodel file name defined, using default model (cnn)')
    options.defmodelfile = 'cnn'

modeldefmodule = importlib.import_module('defmodel.'+options.defmodelfile, package=None)

x_train, y_train, x_test, y_test = get_data_splits_bbox(config['hdf5_filepath_prefix'],
                                                        train_start=0, train_end=190, test_start=190, test_end=None)

model = modeldefmodule.get_model()

model = modeldefmodule.compile_model(model)

# intialize some callbacks
mc = ModelCheckpoint(os.path.join('/local-scratch/cedar-rm/scratch/asa224/', 'model-checkpoints',
                                  options.output_name.split('/')[-1]+'{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', verbose=1,
                                  save_best_only=False, save_weights_only=False, mode='auto', period=1)

reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.001)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

model, history = modeldefmodule.start_training(x_train, y_train, x_test, y_test, model, callbacks=[mc])

modeldefmodule.save_model_with_hyper_and_history(model, history=history, name=options.output_name)

# model, hyperparams = modeldefmodule.open_model_with_hyper(name=options.output_name)
