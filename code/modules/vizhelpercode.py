"""
==========================================================
            Helper Functions for Visualization
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
DESCRIPTION: The module contains some helper functions to
             visualize 3D data. There are also functions
             that work in a jupyter notebook setting, using
             the interact widget.
LICENCE: Proprietary for now.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ipywidgets import interact, fixed
from IPython.display import clear_output
import SimpleITK as sitk
from .configfile import config
from mayavi import mlab
import numpy as np


def viewRectangleInAllSegViews(data, rect=None):
    """
    The function visualizes the segmentation mask of a patient
    in the form of 2D slices. It shows slices in all 3 views,
    axial, sagittal, and coronal. Over the segmentation mask,
    a rectangle bounding box is overlayed to show the annotation.

    :param data: A numpy ndarray of shape (x,y,z) to visualize in 2D
    :param rect: A Rect3D object
    :return:
    """
    all_slices = [0, 1, 2]
    if rect != None:
        rmin, rmax, cmin, cmax, zmin, zmax = rect.list_view
    for slice_idx in all_slices:
        print('plotting along axis {}'.format(slice_idx))
        total_slices = data.shape[slice_idx]
        offset = total_slices / 10
        fig, ax = plt.subplots(nrows=2, ncols=5, squeeze=False, figsize=(20, 10))
        ax = [i for ls in ax for i in ls]
        c = 0
        for a in ax:
            if c < total_slices:
                if slice_idx == 0:
                    a.imshow(data[c, :, :], cmap='gray')
                    if rect != None and c >= rmin and c <= rmax:
                        a.add_patch(
                            patches.Rectangle(
                                (zmin, cmin),  # (x,y) lower left
                                zmax - zmin,  # width
                                cmax - cmin,  # height
                                alpha=0.3
                            )
                        )

                elif slice_idx == 1:
                    a.imshow(data[:, c, :], cmap='gray')
                    if rect != None and c >= cmin and c <= cmax:
                        a.add_patch(
                            patches.Rectangle(
                                (zmin, rmin),  # (x,y) lower left
                                zmax - zmin,  # width
                                rmax - rmin,  # height
                                alpha=0.3
                            )
                        )
                elif slice_idx == 2:
                    a.imshow(data[:, :, c], cmap='gray')
                    if rect != None and c >= zmin and c <= zmax:
                        a.add_patch(
                            patches.Rectangle(
                                (cmin, rmin),  # (x,y) lower left
                                cmax - cmin,  # width
                                rmax - rmin,  # height
                                alpha=0.3
                            )
                        )
                #                 a.axis('off')
                c += offset
            else:
                break
        plt.tight_layout()
        plt.show()

def viewArbitraryVolume(data, slice_idx=2, modality=1):
    total_slices = data.shape[slice_idx]
    offset = total_slices / 10
    fig, ax = plt.subplots(nrows=2, ncols=5, squeeze=False, figsize=(20, 10))
    ax = [i for ls in ax for i in ls]
    c = 0
    for a in ax:
        if c < total_slices:
            if len(data.shape) > 3:
                if config['data_order'] == 'th':
                    if slice_idx <= 1:
                        a.imshow(data[modality, c, :, :], cmap='gray')
                    elif slice_idx == 2:
                        a.imshow(data[modality, :, c, :], cmap='gray')
                    elif slice_idx == 3:
                        a.imshow(data[modality, :, :, c], cmap='gray')
                else:
                    a.imshow(data[:, :, c, 1], cmap='gray')
            elif len(data.shape) == 3:  # its probably a segmentation mask
                if slice_idx == 0:
                    a.imshow(data[c, :, :], cmap='gray')
                elif slice_idx == 1:
                    a.imshow(data[:, c, :], cmap='gray')
                elif slice_idx == 2:
                    a.imshow(data[:, :, c], cmap='gray')
            a.axis('off')
            c += offset
        else:
            break
    plt.tight_layout()
    plt.show()


def viewPatientData(hdf5_file, patient=None, mod='training_data_hgg'):
    print('visualizing the slices..')
    data_to_viz = mod
    if patient == None:
        for pat in range(0, hdf5_file[data_to_viz].shape[0], 10):
            fig, ax = plt.subplots(nrows=2, ncols=5, squeeze=False, figsize=(20, 10))
            ax = [i for ls in ax for i in ls]
            c = 0
            for a in ax:
                if c < 155:
                    if len(hdf5_file[data_to_viz].shape) > 4:
                        if config['data_order'] == 'th':
                            a.imshow(hdf5_file[data_to_viz][pat, 1, :, :, c], cmap='gray')
                        else:
                            a.imshow(hdf5_file[data_to_viz][pat, :, :, c, 1], cmap='gray')
                    elif len(hdf5_file[data_to_viz].shape) == 4: # its probably a segmentation mask
                        a.imshow(hdf5_file[data_to_viz][pat, :, :, c], cmap='gray')
                    a.axis('off')
                    c += 15
                else:
                    break
            plt.tight_layout()
            plt.suptitle('Patient {}'.format(pat))
            plt.show()
    else:
        pat = patient
        fig, ax = plt.subplots(nrows=2, ncols=5, squeeze=False, figsize=(20, 10))
        ax = [i for ls in ax for i in ls]
        c = 0
        for a in ax:
            if c < 155:
                if len(hdf5_file[data_to_viz].shape) > 4:
                    if config['data_order'] == 'th':
                        a.imshow(hdf5_file[data_to_viz][pat, 1, :, :, c], cmap='gray')
                    else:
                        a.imshow(hdf5_file[data_to_viz][pat, :, :, c, 1], cmap='gray')
                elif len(hdf5_file[data_to_viz].shape) == 4: # its probably a segmentation mask
                    a.imshow(hdf5_file[data_to_viz][pat, :, :, c], cmap='gray')
                # a.axis('off')
                c += 15
            else:
                break
        plt.tight_layout()
        plt.suptitle('Patient {}'.format(pat))
        plt.show()


def showMPL(img):
    if img.GetDimension() > 2:
        print('3D Image detected, showing only the middle slice..')
        z = int(img.GetDepth()/2)
        nda = sitk.GetArrayViewFromImage(img)[z,:,:]
    else:
        nda = sitk.GetArrayViewFromImage(img)
    plt.imshow(nda, cmap='gray')
    plt.show()


def createDense(bbox, im):
    box = np.zeros(im.shape)
    box[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = 1
    return box


def viewInMayavi(patch_coords, segmask):
    dense_patch = createDense(patch_coords, segmask)

    src = mlab.pipeline.scalar_field(segmask)

    src_bbox = mlab.pipeline.scalar_field(dense_patch)

    mlab.pipeline.iso_surface(src, opacity=0.4)
    mlab.pipeline.iso_surface(src_bbox, opacity=0.2)

    mlab.show()


# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')
    
    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space. 
# TODO: Fix this
def display_images_with_alpha(image_z, alpha, fixed, moving, fixed_image_z, moving_image_z):
    plt.subplots(1,4,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(1,4,1)
    plt.imshow(fixed[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')
    plt.grid(color='r', linestyle='-', linewidth=2)
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,4,2)
    plt.imshow(moving[moving_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')
    plt.grid(color='r', linestyle='-', linewidth=2)
    
    # Draw the alpha image
    plt.subplot(1,4,3)
    img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z] 
    plt.imshow(img,cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.grid(color='r', linestyle='-', linewidth=2)
    
    # Draw the overlapped image
    plt.subplot(1,4,4)
    plt.imshow(fixed[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.imshow(moving[moving_image_z,:,:],cmap='jet');
    
    plt.show()
    
def display_images_with_alpha_old(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z] 
    plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()
    
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))        