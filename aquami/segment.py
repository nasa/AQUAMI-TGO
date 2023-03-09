#!/usr/bin/env python3
"""
This module contains functions for segmentation with various methods along with tools to clean the segmentation.
"""
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import numbers
from scipy import ndimage
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk, remove_small_objects, binary_opening, binary_closing
from skimage.restoration import denoise_bilateral
from skimage.color import rgb2gray, gray2rgb
from numpy.lib.stride_tricks import as_strided
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

import inout
import visualize
import gui
#from unet import UNet

__author__ = "Joshua Stuckner"

#-----helper function to split data into batches
# https://github.com/choosehappy/PytorchDigitalPathology
def divide_batch(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n,::]


def load_pytorch_model(path, encoder, returned_model=[None], returned_preprocessing_fn=[None]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.load(path, map_location=device)
    except FileNotFoundError:
        path = 'aquami\\' + path
        model = torch.load(path, map_location=device)
    model.eval()
    encoder_weights = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    returned_model[0] = model
    returned_preprocessing_fn[0] = preprocessing_fn
    return True

def extract_patches(arr, patch_shape=8, extraction_step=1):
    #THIS FUNCTION COMES FROM AN OLD VERSION OF SCIKIT-LEARN
    """Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches

# https://github.com/choosehappy/PytorchDigitalPathology
def segmentation_models_inference(io, model, preprocessing_fn, device = None, batch_size = 8, patch_size = 512,
                                  num_classes=3, probabilities=None):

    # This will not output the first class and assumes that the first class is wherever the other classes are not!

    io = preprocessing_fn(io)
    io_shape_orig = np.array(io.shape)
    stride_size = patch_size // 2
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, '... ', end='')
    # add half the stride as padding around the image, so that we can crop it away later
    io = np.pad(io, [(stride_size // 2, stride_size // 2), (stride_size // 2, stride_size // 2), (0, 0)],
                mode="reflect")

    io_shape_wpad = np.array(io.shape)

    # pad to match an exact multiple of unet patch size, otherwise last row/column are lost
    npad0 = int(np.ceil(io_shape_wpad[0] / patch_size) * patch_size - io_shape_wpad[0])
    npad1 = int(np.ceil(io_shape_wpad[1] / patch_size) * patch_size - io_shape_wpad[1])

    io = np.pad(io, [(0, npad0), (0, npad1), (0, 0)], mode="constant")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        arr_out = extract_patches(io, (patch_size, patch_size, 3), stride_size)

    arr_out_shape = arr_out.shape
    arr_out = arr_out.reshape(-1, patch_size, patch_size, 3)

    # in case we have a large network, lets cut the list of tiles into batches
    output = np.zeros((0, num_classes, patch_size, patch_size))
    for batch_arr in divide_batch(arr_out, batch_size):
        arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2).astype('float32')).to(device)

        # ---- get results
        output_batch = model.predict(arr_out_gpu)

        # --- pull from GPU and append to rest of output
        if probabilities is None:
            output_batch = output_batch.detach().cpu().numpy().round()
        else:
            output_batch = output_batch.detach().cpu().numpy()

        output = np.append(output, output_batch, axis=0)

    output = output.transpose((0, 2, 3, 1))

    # turn from a single list into a matrix of tiles
    output = output.reshape(arr_out_shape[0], arr_out_shape[1], patch_size, patch_size, output.shape[3])

    # remove the padding from each tile, we only keep the center
    output = output[:, :, stride_size // 2:-stride_size // 2, stride_size // 2:-stride_size // 2, :]

    # turn all the tiles into an image
    output = np.concatenate(np.concatenate(output, 1), 1)

    # incase there was extra padding to get a multiple of patch size, remove that as well
    output = output[0:io_shape_orig[0], 0:io_shape_orig[1], :]  # remove paddind, crop back
    if probabilities is None:
        return output[:, :, 1:].astype('bool')
    else:
        for i in range(num_classes-1): #don't care about background class
            output[:,:,i+1] = output[:,:,i+1] > probabilities[i]
        return output[:, :, 1:].astype('bool')

def smooth_edges(mask, smooth_radius=1):
    """
    Smoothes the edges of a binary mask.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.
    smooth_radius : float64
        The radius of the smoothing operation.  See note below.

    Returns
    -------
    smooth_mask : 2D array of bool
        Mask that has been smoothed.

    Notes
    -----
    smooth_radius sets the structure element (selem) for smoothing the edges of
    the masks. If the smooth_rad rounds up the selem is a disk with radius
    rounded up.  If smooth_radius rounds down, selem is a box.

    Radius < 1
    [[0,1,0],
     [1,1,1],
     [0,1,0]]

    Radius = 1 - 1.499
    [[1,1,1],
     [1,1,1],
     [1,1,1]]

    Radius = 1.5 - 1.99
    [[0,0,1,0,0],
     [0,1,1,1,0],
     [1,1,1,1,1],
     [0,1,1,1,0],
     [0,0,1,0,0]]

    Radius = 2 - 2.499
    [[1,1,1,1,1],
     [1,1,1,1,1],
     [1,1,1,1,1],
     [1,1,1,1,1],
     [1,1,1,1,1]]
    """

    if smooth_radius < 1:
        selem = np.array([[0,1,0],[1,1,1],[0,1,0]])
    elif round(smooth_radius, 0) > int(smooth_radius):  # If round up.
        size = int(smooth_radius + 1)
        selem = disk(round(smooth_radius, 0))
    else:
        size = 1 + 2 * int(smooth_radius)
        selem = np.ones((size, size))

    # Smooth edges.
    # It is necessary to perform this step because skelatonizing algorithms are
    # extremely sensitive to jagged edges and may otherwise give spurious
    # results.

    smooth_mask = binary_closing(mask, selem)
    smooth_mask = binary_opening(smooth_mask, selem)

    return smooth_mask


def removeSmallObjects(mask, small_objects=0, small_holes=0):
    """
    Removes small objects (white areas of mask) and small holes (black areas).

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.
    small_objects : int
        Max area of connected white pixels that will be removed.
    small_holes : int
        Max area of connected black pixels that will be removed.

    Returns
    -------
    out_mask : 2D array of bool
        Mask with small holes and objects removed.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_mask = remove_small_objects(mask, small_objects)
        out_mask = ~remove_small_objects(~out_mask, small_holes)

    return out_mask


def otsu_segment(image):
    """
    Returns binary mask of the segmented precipitates.  Uses Otsu's Method.

    Parameters
    ----------
    image : 2D array of uint8
        Image data to be segmented

    Returns
    -------
    mask : 2D array of bool
        Mask of precipitates.

    """

    # Don't operate on the passed image.
    img = np.copy(image)

    # Get shape of image and convert to grayscale.
    try:
        rows, cols = img.shape
    except ValueError:  # Convert to grayscale
        img = rgb2gray(img)
        rows, cols = img.shape

    # Ensure proper datatype.
    img = inout.uint8(img)

    # Set radius of smoothing filter based on the image resolution.
    blur_mult = 1.4  # Increasing this parameter increases the blur radius.
    blur_radius = max(1, (cols * blur_mult / 1024))

    # Smooth the image to reduce noise.
    img = ndimage.gaussian_filter(img, blur_radius)

    # Ensure proper datatype after bilateral smoothing.
    img = inout.uint8(img)

    # Use Otsu's Method to determine the global threshold value.
    threshold_global_otsu = threshold_otsu(img)
    # Generate masks of both phases based on Otsu's threshold value.
    mask = img >= threshold_global_otsu


    return mask


def local_otsu_segment(image, estSize):
    """
    Returns binary masks of the segmented bright and dark phases.  Uses
    two-dimentional Otsu's Method. 2D Otsu's Method takes significantly longer to perform than
    the regular method and is more robust against illumination and contrast
    changes throughout the image by setting a seperate threshold value for each
    pixel.

    Parameters
    ----------
    image : 2D array of uint8
        Image data to be segmented

    Returns
    -------
    mask : 2D array of bool
        Mask of precipitates.
    """

    # Don't operate on the passed image.
    img = np.copy(image)

    # Adjustable parameters.
    radius_ratio = 0.1  # Ratio of Otsu Radius / # image columns
    blur_mult = 1.4  # Increasing this parameter increases the blur radius.

    # Get shape of image and convert to grayscale.
    try:
        rows, cols = img.shape
    except ValueError:  # Convert to grayscale
        img = rgb2gray(img)
        rows, cols = img.shape

    # Smooth the image.
    blur_radius = max(1, int(round(cols * blur_mult / 1000, 0)))
    img = denoise_bilateral(img,
                            sigma_color=0.05,  # increase this to blur more
                            sigma_spatial=blur_radius,
                            multichannel=False)

    # Ensure proper datatype after bilateral smoothing.
    img = inout.uint8(img)

    # Perform Otsu thresholding
    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu

    # Perform 2D Otsu thresholding.
    radius = radius_ratio * cols
    selem = disk(radius)
    local_otsu = rank.otsu(img, selem)
    mask = img >= local_otsu

    return mask


def manualSegment(image):
    """
    Returns binary masks of the segmented bright and dark phases based on
    manually selected threshold value.

    Parameters
    ----------
    image : 2D array of uint8
        Image data to be segmented

    Returns
    -------
    mask : 2D array of bool
        Mask of precipitates.

    """

    # Adjustable parameters.
    blur_mult = 1.4  # Increasing this parameter increases the blur radius.

    img = inout.uint8(image)

    # Get shape of image and convert to grayscale.
    try:
        rows, cols = img.shape
    except ValueError:  # Convert to grayscale
        img = rgb2gray(img)
        rows, cols = img.shape

    # Smooth the image to reduce noise.
    blur_radius = max(1, (cols * blur_mult / 1024))
    img = ndimage.gaussian_filter(img, blur_radius)

    # Get the default threshold value.
    threshinit = threshold_otsu(img)

    # Launch the GUI for manual threshholding.
    p = gui.manualThreshhold(img, threshinit)
    p.show()
    mask = p.getMask()

    return mask
