from __future__ import print_function
import sys
sys.path.append('..')
from modules.configfile import config
import h5py
import numpy as np
import SimpleITK as sitk
import six

from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape

h5_main = h5py.File(config['hdf5_filepath_prefix'], mode='r')
h5 = h5_main['original_data']

samp = h5['training_data_segmasks_hgg'][0].astype(np.float32)

mask = np.zeros(np.shape(samp)).astype(np.float32)


image = sitk.GetImageFromArray(samp)
mask = sitk.GetImageFromArray(mask)

applyLog = False
applyWavelet = False

# Setting for the feature calculation.
# Currently, resampling is disabled.
# Can be enabled by setting 'resampledPixelSpacing' to a list of 3 floats (new voxel size in mm for x, y and z)
settings = {'binWidth': 4,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None}

#
# If enabled, resample image (resampled image is automatically cropped.
# If resampling is not enabled, crop image instead
#
if settings['interpolator'] is not None and settings['resampledPixelSpacing'] is not None:
  image, mask = imageoperations.resampleImage(image, mask, settings['resampledPixelSpacing'], settings['interpolator'])
else:
  bb, correctedMask = imageoperations.checkMask(image, mask)
  if correctedMask is not None:
    mask = correctedMask
  image, mask = imageoperations.cropToTumorMask(image, mask, bb)

#
# Show the first order feature calculations
#
firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)

firstOrderFeatures.enableFeatureByName('Mean', True)
# firstOrderFeatures.enableAllFeatures()

print('Will calculate the following first order features: ')
for f in firstOrderFeatures.enabledFeatures.keys():
  print('  ', f)
  print(getattr(firstOrderFeatures, 'get%sFeatureValue' % f).__doc__)

print('Calculating first order features...')
firstOrderFeatures.calculateFeatures()
print('done')

print('Calculated first order features: ')
for (key, val) in six.iteritems(firstOrderFeatures.featureValues):
  print('  ', key, ':', val)

#
# Show Shape features
#
shapeFeatures = shape.RadiomicsShape(image, mask, **settings)
shapeFeatures.enableAllFeatures()

print('Will calculate the following Shape features: ')
for f in shapeFeatures.enabledFeatures.keys():
  print('  ', f)
  print(getattr(shapeFeatures, 'get%sFeatureValue' % f).__doc__)

print('Calculating Shape features...')
shapeFeatures.calculateFeatures()
print('done')

print('Calculated Shape features: ')
for (key, val) in six.iteritems(shapeFeatures.featureValues):
  print('  ', key, ':', val)

#
# Show GLCM features
#
glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
glcmFeatures.enableAllFeatures()

print('Will calculate the following GLCM features: ')
for f in glcmFeatures.enabledFeatures.keys():
  print('  ', f)
  print(getattr(glcmFeatures, 'get%sFeatureValue' % f).__doc__)

print('Calculating GLCM features...')
glcmFeatures.calculateFeatures()
print('done')

print('Calculated GLCM features: ')
for (key, val) in six.iteritems(glcmFeatures.featureValues):
  print('  ', key, ':', val)

#
# Show GLRLM features
#
glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
glrlmFeatures.enableAllFeatures()

print('Will calculate the following GLRLM features: ')
for f in glrlmFeatures.enabledFeatures.keys():
  print('  ', f)
  print(getattr(glrlmFeatures, 'get%sFeatureValue' % f).__doc__)

print('Calculating GLRLM features...')
glrlmFeatures.calculateFeatures()
print('done')

print('Calculated GLRLM features: ')
for (key, val) in six.iteritems(glrlmFeatures.featureValues):
  print('  ', key, ':', val)

#
# Show GLSZM features
#
glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
glszmFeatures.enableAllFeatures()

print('Will calculate the following GLSZM features: ')
for f in glszmFeatures.enabledFeatures.keys():
  print('  ', f)
  print(getattr(glszmFeatures, 'get%sFeatureValue' % f).__doc__)

print('Calculating GLSZM features...')
glszmFeatures.calculateFeatures()
print('done')

print('Calculated GLSZM features: ')
for (key, val) in six.iteritems(glszmFeatures.featureValues):
  print('  ', key, ':', val)

#
# Show FirstOrder features, calculated on a LoG filtered image
#
if applyLog:
  sigmaValues = numpy.arange(5., 0., -.5)[::1]
  for logImage, imageTypeName, inputKwargs in imageoperations.getLoGImage(image, mask, sigma=sigmaValues):
    logFirstorderFeatures = firstorder.RadiomicsFirstOrder(logImage, mask, **inputKwargs)
    logFirstorderFeatures.enableAllFeatures()
    logFirstorderFeatures.calculateFeatures()
    for (key, val) in six.iteritems(logFirstorderFeatures.featureValues):
      laplacianFeatureName = '%s_%s' % (imageTypeName, key)
      print('  ', laplacianFeatureName, ':', val)
#
# Show FirstOrder features, calculated on a wavelet filtered image
#
if applyWavelet:
  for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(image, mask):
    waveletFirstOrderFeaturs = firstorder.RadiomicsFirstOrder(decompositionImage, mask, **inputKwargs)
    waveletFirstOrderFeaturs.enableAllFeatures()
    waveletFirstOrderFeaturs.calculateFeatures()
    print('Calculated firstorder features with wavelet ', decompositionName)
    for (key, val) in six.iteritems(waveletFirstOrderFeaturs.featureValues):
      waveletFeatureName = '%s_%s' % (str(decompositionName), key)
      print('  ', waveletFeatureName, ':', val)