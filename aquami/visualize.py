import warnings

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import gray2rgb, rgb2gray, label2rgb
from matplotlib import animation
from skimage import measure
import scipy.ndimage.morphology as morphology
from skimage.segmentation import find_boundaries
from skimage.util import img_as_ubyte
from scipy.stats import lognorm, norm

def show_full(img, title=None, cmap=None, interpolation='none'):
    """
    Displays a full screen figure of the image.

    Parameters
    ----------
    img : ndarray
        Image to display.
    title : str, optional
        Text to be displayed above the image.
    cmap : Colormap, optional
        Colormap that is compatible with matplotlib.pyplot
    interpolation : string, optional
        How display pixels that lie between the image pixels will be handled.
        Acceptable values are ‘none’, ‘nearest’, ‘bilinear’, ‘bicubic’,
        ‘spline16’, ‘spline36’, ‘hanning’, ‘hamming’, ‘hermite’, ‘kaiser’,
        ‘quadric’, ‘catrom’, ‘gaussian’, ‘bessel’, ‘mitchell’, ‘sinc’, ‘lanczos’
    """

    # Show grayscale if cmap not set and image is not color.
    if cmap is None and img.ndim == 2:
        cmap = plt.cm.gray

    plt.imshow(img, cmap=cmap, interpolation=interpolation)
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
    #plt.ion() Removed 3/9/23 to stop code from hanging when closing figure.
    plt.show()
    while plt.get_fignums():
        try:
            plt.pause(0.1)
        except:
            pass
    plt.ioff()


def label_overlay(im, label):
    image_label_overlay = label2rgb(label, image=im, alpha=1, bg_label=0, image_alpha=1, bg_color=(0, 0, 0))
    black_pixels_mask = np.all(image_label_overlay == [0, 0, 0], axis=-1)
    image_label_overlay[black_pixels_mask] = im[black_pixels_mask] / 255
    return image_label_overlay


def overlay_mask_single(image, mask, color='o', return_overlay=False, animate=False,
                title=None, delay=1000):
    '''
    Displays the binary mask over the original image in order to verify results.

    Parameters
    ----------
    image : image array
        Image data prior to segmentation.
    mask : binary array
        Binary segmentation of the image data.  Must be the same size as image.
    color : str, optional
        The color of the overlaid mask.
    return_overlay : bool, optional
        If true, the image with the overlaid mask is returned and the overlay
        is not displayed here.
    animate : bool, optional
        If true, an animated figure will be displayed that alternates between
        showing the raw image and the image with the overlay.

    Returns
    -------
    overlay : RGB image array, optional
        Color image with mask overlayyed on original image (only returned
        if 'return_overlay' is True).
    '''



    if title is None:
        title = 'Segmentation mask overlayed on image'

    img = np.copy(image)
    this_mask = np.copy(mask)

    # Convert the image into 3 channels for a colored mask overlay
    overlay = gray2rgb(img)

    # Set color (default to blue if a proper color string is not given).
    r = 0
    g = 0
    b = 255
    if color == 'red' or color == 'r':
        r = 255
        g = 0
        b = 0
    if color == 'green' or color == 'g':
        r = 0
        g = 255
        b = 0
    if color == 'blue' or color == 'b':
        r = 0
        g = 0
        b = 255
    if color == 'white' or color == 'w':
        r = 255
        g = 255
        b = 255
    if color == 'yellow' or color == 'y':
        r = 255
        g = 255
        b = 0
    if color == 'orange' or color == 'o':
        r = 255
        g = 128
        b = 0

    # Apply mask.
    if r != 0:
        overlay[this_mask == 1, 0] = r/255
    if g != 0:
        overlay[this_mask == 1, 1] = g/255
    if b != 0:
        overlay[this_mask == 1, 2] = b/255

    # Return or show overlay.
    if return_overlay:
        return overlay
    else:
        if animate:
            fig = plt.figure()
            ims = []
            for i in range(10):
                ims.append([plt.imshow(image, cmap=plt.cm.gray, animated=True)])
                ims.append([plt.imshow(overlay, animated=True)])
            ani = animation.ArtistAnimation(fig, ims, delay, True, delay)
            plt.axis('off')
            figManager = plt.get_current_fig_manager()
            try:
                figManager.window.showMaximized()
            except AttributeError:  # TkAgg backend
                figManager.window.state('zoomed')
            plt.gca().set_position([0, 0, 1, 0.95])
            plt.title(title)
            fig.canvas.set_window_title('Animated Mask Overlay')
            #plt.ion() Removed 3/9/23 to stop code from hanging when closing figure.
            plt.show()
            while plt.get_fignums():
                try:
                    plt.pause(0.1)
                except:
                    pass
            plt.ioff()


        else:
            show_full(overlay, title=title, interpolation='nearest')


def overlay_mask(image, mask, colors=['o', 'b', 'r'], return_overlay=False, animate=False,
                title=None, delay=1000):
    '''
    Displays the binary mask over the original image in order to verify results.

    Parameters
    ----------
    image : image array
        Image data prior to segmentation.
    mask : binary array
        Binary segmentation of the image data.  Must be the same size as image.
    color : str, optional
        The color of the overlaid mask.
    return_overlay : bool, optional
        If true, the image with the overlaid mask is returned and the overlay
        is not displayed here.
    animate : bool, optional
        If true, an animated figure will be displayed that alternates between
        showing the raw image and the image with the overlay.

    Returns
    -------
    overlay : RGB image array, optional
        Color image with mask overlayyed on original image (only returned
        if 'return_overlay' is True).
    '''


    # TODO Refractor so that it doesn't convert to lists of masks

    if title is None:
        title = 'Segmentation mask overlayed on image'


    img = np.copy(image)



    if not isinstance(mask, list):
        try: # make it work with mask tensor
            rows, cols, chan = mask.shape
            this_mask = [np.copy(mask[:,:,c]) for c in range(chan)]
        except ValueError: # 2 channels
            this_mask = [np.copy(mask)]
    else:
        this_mask = [np.copy(m) for m in mask]

    if not isinstance(colors, list):
        colors = [colors]

    colors = colors[:len(this_mask)]
    # Convert the image into 3 channels for a colored mask overlay
    overlay = gray2rgb(img)


    # Set color (default to blue if a proper color string is not given).
    for i, color in enumerate(colors):
        if color == 'red' or color == 'r':
            r = 255
            g = 0
            b = 0
        if color == 'green' or color == 'g':
            r = 0
            g = 255
            b = 0
        if color == 'blue' or color == 'b':
            r = 0
            g = 0
            b = 255
        if color == 'white' or color == 'w':
            r = 255
            g = 255
            b = 255
        if color == 'yellow' or color == 'y':
            r = 255
            g = 255
            b = 0
        if color == 'orange' or color == 'o':
            r = 255
            g = 128
            b = 0

        # Apply mask.
        if r != 0:
            overlay[this_mask[i] == 1, 0] = r
        if g != 0:
            overlay[this_mask[i] == 1, 1] = g
        if b != 0:
            overlay[this_mask[i] == 1, 2] = b

    # Return or show overlay.
    if return_overlay:
        return overlay
    else:
        if animate:
            fig = plt.figure()
            ims = []
            for i in range(10):
                ims.append([plt.imshow(image, cmap=plt.cm.gray, animated=True)])
                ims.append([plt.imshow(overlay, animated=True)])
            ani = animation.ArtistAnimation(fig, ims, delay, True, delay)
            plt.axis('off')
            figManager = plt.get_current_fig_manager()
            try:
                figManager.window.showMaximized()
            except AttributeError:  # TkAgg backend
                figManager.window.state('zoomed')
            plt.gca().set_position([0, 0, 1, 0.95])
            plt.title(title)
            try:
                fig.canvas.set_window_title('Animated Mask Overlay')
            except AttributeError:
                pass
            #plt.ion() Removed 3/9/23 to stop code from hanging when closing figure.
            plt.show()
            while plt.get_fignums():
                try:
                    plt.pause(0.1)
                except:
                    pass
            plt.ioff()


        else:
            show_full(overlay, title=title, interpolation='nearest')


def show_skel(skeleton, mask, dialate=False, title=None, returnSkel=False,
             cmap=None):
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

    # Fix error from matplotlib update
    if cmap is None:
        try:
            cmap = plt.cm.spectral
        except AttributeError:
            cmap = plt.cm.nipy_spectral

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
    skel = np.uint8(cmap(skel) * 255)  # apply colormap to skel data.
    for i in range(3):
        skel[:, :, i] += outlines

    if returnSkel:
        return skel

    # Display the results.
    plt.imshow(skel, cmap=cmap, interpolation='none')
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
    #plt.ion() Removed 3/9/23 to stop code from hanging when closing figure.
    plt.show()
    while plt.get_fignums():
        try:
            plt.pause(0.1)
        except:
            pass
    plt.ioff()

def show_hist(data, title=None, xlabel=None,
             numBins=None, gauss=False, log=False):
    """
    Displays a histogram of data with no y-axis and the option to fit gaussian
    and lognormal distribution curves to the data.

    Parameters
    ----------
    data : ndarray
        Data or measurements from which to produce a histogram.
    title : str, optional
        Title of histogram which is displayed at the top of the figure.
    xlabel : str, optional
        Title of the x-axis.  Usually what is measured along the x-axis along
        with the units.
    numBins : int, optional
        Number of possible bars in the histogram.  If not given, the function
        will attempt to automatically pick a good value.
    gauss: boolean, optional
        If true, a fitted guassian distribution curve will be plotted on the
        histogram.
    log: boolean, optional
        If true, a fitted lognormal distribution curve will be plotted on the
        histogram.
    """

    # Determine optimum number of bins.
    if numBins is None:
        u = len(np.unique(data))
        numBins = int(2*u**(1/2))
        if numBins < 4:
            numBins = len(np.unique(data))
            if numBins < 1:
                numBins = 1

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
            pdf_plot = lognorm.pdf(bins,
                                   logfit[0], loc=logfit[1], scale=logfit[2])
            plt.plot(bins, pdf_plot, 'r--', linewidth=3, label='lognorm')
        except ValueError:
            pass
    if gauss:
        gfit = norm.fit(np.asarray(data).flatten())
        gauss_plot = norm.pdf(bins, gfit[0], gfit[1])
        plt.plot(bins, gauss_plot, 'g--', linewidth=3, label='gaussian')

    # Display the histogram.
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    ax = plt.gca()
    ax.yaxis.set_visible(False)
    if gauss or log:
        plt.legend()
    #plt.ion() Removed 3/9/23 to stop code from hanging when closing figure.
    plt.show()
    while plt.get_fignums():
        try:
            plt.pause(0.1)
        except:
            pass
    plt.ioff()