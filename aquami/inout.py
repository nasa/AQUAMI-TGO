#!/usr/bin/env python3
"""
This module contains functions for the input and output of images, data, and
measurement results for BNMs
"""

from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")

import warnings
warnings.filterwarnings("ignore")
import glob
import os

import numpy as np
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import lognorm, norm
import skimage.measure
import scipy.ndimage.morphology as morphology
from skimage.segmentation import find_boundaries
from skimage.color import rgb2gray
from scipy.stats._continuous_distns import FitDataError
import imageio

import visualize

__author__ = "Joshua Stuckner"

plt.rcParams["savefig.dpi"] = 300

def uint8(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = img_as_ubyte(img)
    return img


def files_in_folder(folder, ftype='*'):
    return [file for file in glob.glob(''.join((folder + '/*.', ftype)))]


def files_in_subfolders(folder, ftype='*'):
    return [file for file in glob.glob(''.join((folder + '/**/*.', ftype)), recursive=True)]


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def load(path, convert_to_uint8=True):
    img = imageio.imread(path)
    if uint8:
        img = rgb2gray(img) if len(img.shape) == 3 else img
        img = uint8(img)
    return img

def load_mask(path):
    mask = imageio.imread(path)
    mask = rgb2gray(mask) if len(mask.shape) == 3 else mask
    mask = mask > 0
    return mask

def _uint16_to_uint8(image, display_min=0, display_max=0):
    """Helper function for uint16_to_uint8"""
    image = np.array(image, copy=True)
    if display_min == 0 and display_max == 0:
        sigma = 3
        display_min = image.mean() - sigma * image.std()
        display_max = image.mean() + sigma * image.std()
    image.clip(display_min, display_max, out=image)
    image = image - display_min
    np.true_divide(image, (display_max - display_min + 1) / 256., out=image, casting="unsafe")
    return image.astype(np.uint8)


def uint16_to_uint8(image, display_min=0, display_max=0):
    """
    Uses a lookup table to quickly convert an image from uin16 to uint 8.

    Parameters
    ----------
    image : ndarray of uint16
        Image data to convert.
    display_min : int, optional
        Pixels in image with a value less than this will be clipped to 0.
    display_max : int, optional
        Pixels in image with a value higher than this will be clipped to 255.
        If both display_max and display_min are set to 0, the image intensity
        range will be automatically selected.

    Returns
    -------
    ndarray of uint8
        Image converted to uint8
    """

    lut = np.arange(2 ** 16, dtype='uint16')
    lut = _uint16_to_uint8(lut, display_min, display_max)
    return np.take(lut, image)


def save_movie(frames, save_name, fps=2, bit_rate=-1):
    """
    Saves a movie of the images in frames.

    Parameters
    ----------
    frames : list of images
        Contains images to create a movie from.
    fps : double, optional
        Frames per second of movie.  Defaults is 20.
    save_name : string
        Path to save the movie.
    bit_rate : int
        bits per second of movie.  See matplotlib.animation.writers
    """

    fig = plt.figure()
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    ims = []

    for i in frames:
        ims.append([plt.imshow(i, animated=True)])

    try:  # FFmpeg
        ani = animation.ArtistAnimation(fig, ims, fps / 1000, True, fps / 1000)
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=bit_rate,
                                             extra_args=['-vcodec', 'libx264',
                                                         '-pix_fmt', 'yuv420p'])
        ani.save(save_name, writer=writer)

    except RuntimeError:
        print("To save the movie you must install FFmpeg.",
              " Go to http://adaptivesamples.com/how-to-install-ffmpeg-on",
              "-windows/ and follow the directions to install the FFmpeg ",
              "movie writer.", sep='')


##def save_movie(stack, fps=20, directory=None,
##               save_name=None, initialdir='/'):
##    """
##    Saves a movie of the images to the specified directory.
##
##    Parameters
##    ----------
##    stack : ndarray (3D)
##        Contains images to create a movie from.
##    fps : double, optional
##        Frames per second of movie.  Defaults is 20.
##    directory : string, optional
##        Directory to save the movie.  If not provided the user will be asked.
##    image_names : string, optional
##        What to name the movie.  If not provided the user will be asked.
##    initialdir : string, optional
##        If the save directory is not given, this parameter is used as the
##        starting directory for the save as dialog.
##    """
##
##    if directory is None:
##        # User selects the directory to save the movie.
##        root = tkinter.Tk()
##        directory = tkinter.filedialog.askdirectory(
##                parent=root,
##                initialdir=initialdir,
##                title='Select a folder to save your movie.'
##                )
##        root.destroy()
##        if len(directory) == 0:
##            print("No folder selected.  Movie will not be saved.")
##            return
##
##    if save_name is None:
##        save_name = input("What do you want to name your movie? ")
##
##    save_path = directory + '/' + save_name + '.mp4'
##
##    # Get the size of the images
##    try:
##        rows, cols = stack[0].shape
##        color = False
##    except ValueError:
##        rows, cols, _ = stack[0].shape
##        color = True
##
##    try: #FFmpeg
##        fig = plt.figure()
##        plt.axis('off')
##        ims = []
##        for im in stack:
##            ims.append([plt.imshow(im, cmap=plt.get_cmap('gray'), animated=True)])
##            ims.append([plt.imshow(im, animated=True)])
##        ani = animation.ArtistAnimation(fig, ims, fps/1000, True, fps/1000)
##        writer = animation.writers['ffmpeg'](fps=fps)
##        ani.save(save_path, writer=writer)
##    except RuntimeError:
##        try: #openCV
##            import cv2
##            video = cv2.VideoWriter(save_path, 0,
##                                    fps, (int(cols), int(rows)), isColor=True)
##
##            for i in range(len(stack)):
##                video.write(stack[i])
##            cv2.destroyAllWindows()
##            video.release()
##        except ImportError:
##            print("To save the movie you must install either openCV or FFmpeg.",
##                  " Go to http://adaptivesamples.com/how-to-install-ffmpeg-on",
##                  "-windows/ and follow the directions to install the FFmpeg ",
##                  "movie writer.", sep='')

def saveBinary(mask, path):
    """
    Saves the binary mask to the path.

    Parameters
    ----------
    mask : 2darray
        Contains mask to save.
    path : str
        Path to binary file.  ex: "c:/images/binary.tiff"
    """
    plt.imsave(path, mask * 255)


def pdfSaveImage(pdf, image, title='', cmap=None):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def pdfSaveOverlay(pdf, image, mask, title=''):
    overlay = visualize.overlay_mask(image, mask, return_overlay=True)
    plt.imshow(overlay)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def pdfSaveHist(pdf, data, title=None, xlabel=None,
                numBins=None, gauss=False, log=False):
    # Determine optimum number of bins.
    if numBins is None:
        u = len(np.unique(data))
        numBins = int(2 * u ** (1 / 2))
        if numBins < 4:
            numBins = len(np.unique(data))

    # Create the histogram.
    try:
        n, bins, patches = plt.hist(
            data, bins=numBins, density=1, edgecolor='black')
    except:
        n, bins, patches = plt.hist(
            data, bins=numBins, normed=1, edgecolor='black')
    if log:
        try:
            logfit = lognorm.fit(np.asarray(data).flatten(), floc=0)
            pdf_plot = lognorm.pdf(bins, logfit[0], loc=logfit[1], scale=logfit[2])
            plt.plot(bins, pdf_plot, 'r--', linewidth=3, label='lognorm')
        except FitDataError:
            pass
    if gauss:
        gfit = norm.fit(np.asarray(data).flatten())
        gauss_plot = norm.pdf(bins, gfit[0], gfit[1])
        plt.plot(bins, gauss_plot, 'g--', linewidth=3, label='gaussian')

    # Save the histogram
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    ax = plt.gca()
    ax.yaxis.set_visible(False)
    if gauss or log:
        plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def pdfSaveSkel(pdf, skeleton, mask, dialate=False, title=None, cmap=None):
    """
    Displays skelatal data on top of an outline of a binary mask. For example,
    displays a medial axis transform over an outline of segmented ligaments.

    Parameters
    ----------
    skeleton : 2D array
        Data to be displayed.
    mask : binary 2D array
        Mask of segmentation data, the outline of which is displayed along with
        the skel data.
    dialate : boolean, optional
        If dialate is true, the skelatal data will be made thicker in the
        display.
    title : str, optional
        Text to be displayed above the image.
    """

    skel = np.copy(skeleton)

    # Find the outlines of the mask and make an outline mask called outlines.
    outlines = find_boundaries(mask, mode='outer')
    outlines = img_as_ubyte(outlines)

    # Make the skel data thicker if dialate is true.
    if dialate:
        skel = morphology.grey_dilation(skel, size=(3, 3))

    # Scale the skel data to uint8 and add the outline mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skel = skel.astype(np.float32)  # convert to float
        skel -= skel.min()  # ensure the minimal value is 0.0
        if skel.max() != 0:
            skel /= skel.max()  # maximum value in image is now 1.0
    # apply colormap to skel data.
    if cmap is None:
        try:
            skel = np.uint8(plt.cm.spectral(skel) * 255)
        except AttributeError:
            skel = np.uint8(plt.cm.nipy_spectral(skel) * 255)
    else:
        skel = np.uint8(cmap(skel) * 255)
    for i in range(3):
        skel[:, :, i] += outlines

    # Display the results.
    try:
        plt.imshow(skel, cmap=plt.cm.spectral, interpolation='none')
    except:
        plt.imshow(skel, cmap=plt.cm.nipy_spectral, interpolation='none')
    plt.axis('off')
    figManager = plt.get_current_fig_manager()
    try:
        figManager.window.showMaximized()
    except AttributeError:  # TkAgg backend
        figManager.window.state('zoomed')
    if title is None:
        plt.gca().set_position([0, 0, 1, 1])
    else:
        plt.gca().set_position([0, 0, 1, 0.95])
        plt.title(title)
    pdf.savefig()
    plt.close()


def parsePath(path):
    folder = '/'.join(path.split('/')[:-1])
    fname = path.split('/')[-1].split('.')[:-1][0]
    ftype = path.split('.')[-1]
    fnametype = fname + '.' + ftype
    return folder, fname, ftype, fnametype


