import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
from threading import Thread
import queue
import time
import itertools
import math
import sys
import traceback
import imageio

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
from skimage.morphology import (skeletonize, remove_small_objects, disk,
                                binary_dilation, binary_opening, remove_small_holes, binary_erosion)
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb, rgb2gray, gray2rgb
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import flood_fill
from scipy.signal import convolve2d, find_peaks, peak_prominences, peak_widths, savgol_filter
from scipy.ndimage.morphology import distance_transform_edt
from scipy.stats import lognorm, norm
from skimage.segmentation import clear_border
import pandas as pd

import visualize
import inout
import segment
import gui


def skel(mask, diam=20, removeEdge=False, times=3):
    """
    Returns the skeletal backbone of the ligaments or pores which are one pixel
    thick lines sharing the same connectivity as the passed binary mask.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase. Must be continuously connected.
    diam : float64
        The estimated or measured average diameter of the ligaments in the mask.
    removeEdge : bool, optional
        If true, the returned skel does not include ligaments that are too close
        to the edge.

    Returns
    -------
    skel : 2D array of bool
        Skeletal backbone of the mask.  One pixel thick lines sharing the same
        connectivity as the mask.
    """

    # Get the shape of the mask.
    rows, cols = mask.shape

    # Find the initial skeletal backbone.
    skel = skeletonize(mask > 0)

    # Remove  terminal edges.

    for i in range(times):
        # Find number of 1st and 2nd neighboring pixels
        neighbors = convolve2d(skel, np.ones((3, 3)), mode='same')
        neighbors = neighbors * skel
        neighbors2 = convolve2d(skel, np.ones((5, 5)), mode='same')
        neighbors2 = neighbors2 * skel

        # Remove nodes and label each ligament section.
        nodes = neighbors > 3
        skel = np.bitwise_xor(skel, nodes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skel = remove_small_objects(skel, 1, connectivity=1)
        labels = label(skel, connectivity=2, background=0)

        # Find terminal sections.
        terminal_labels = labels[neighbors == 2]  # These are definitly terminal
        # When there is a 1 pixel fork at the end, we need to look at second
        # nearest neighbors as well.
        # neighbors[neighbors2>5] = 10
        # terminal_labels2 = labels[neighbors<5]
        # terminal_labels = np.append(terminal_labels, terminal_labels2)
        terminal_sections = np.unique(terminal_labels)
        just_terminal = np.zeros(skel.shape, dtype=bool)
        for lab in terminal_sections:
            just_terminal[labels == lab] = 1

        # Remove terminal sections.
        skel = np.bitwise_xor(skel, just_terminal)

        # Put the nodes back in.
        skel = binary_dilation(skel, selem=disk(3))
        skel = skeletonize(skel)

    # Remove sections that touch the edge of image.
    if removeEdge:
        try:
            edge = round(int(diam))
        except ValueError:
            edge = 0
        skel = np.bitwise_xor(skel, nodes)
        labels = label(skel, connectivity=2, background=0)
        elabels = np.copy(labels)
        elabels[edge:rows - edge - 1, edge:cols - edge - 1] = 0
        edge_labels = np.unique(elabels)
        for lab in edge_labels:
            skel[labels == lab] = 0

        # Put the nodes back in.
        skel = skel + nodes

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skel = remove_small_objects(skel, 8, connectivity=2)

    return skel

def distance_transform(mask):
    """
    Returns the euclidean distance each white pixel is to the nearest black
    pixel.  Just calls scipy.ndimage.morphology.distance_transform_edt.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.

    Returns
    -------
    dist : ndarray
        Size in nm of each pixel
    """

    return distance_transform_edt(mask)


def get_background(mask):
    return np.logical_not(mask.max(axis=2))

def get_foreground(mask):
    return mask.max(axis=2)

def channel_width(mask, skeleton=None, dist=None,
                  ignore_edge_dist=70, secondary_size = 400, show_steps=False, pdf=None,
                  title=None, pixel_size = 1):
    mask = ~mask
    rows, cols = mask.shape
    thresh = 0 #i gnore diameter values smaller than this as noise

    # Preprocess the mask for measurement
    if skeleton is None and mask is None:
        mask = remove_small_objects(mask, secondary_size)
        mask = remove_small_holes(mask, 1000)
        mask = segment.smooth_edges(mask, max(1, rows // 1000))


        dist = distance_transform(mask)
        dist *= 2 #convert radius to diameter
        skeleton = skel(mask)
    skel_dist = dist * skeleton

    if show_steps:
        dialate = False if cols < 1500 else True  # Thicker result at high res.
        visualize.show_skel(skel_dist, mask, dialate=dialate,
                         title=title
                         )
    if pdf is not None:
        dialate = False if cols < 1500 else True  # Thicker result at high res.
        inout.pdfSaveSkel(pdf, skel_dist, mask, dialate=dialate,
                          title=title
                          )
    # Remove the edges of the distance map.
    skel_dist = skel_dist[ignore_edge_dist:-ignore_edge_dist,
           ignore_edge_dist:-ignore_edge_dist]

    # Prepare list of all measurements.
    all_measurements = np.sort(skel_dist[skel_dist > 0].flatten())

    # Ignore small measurements as noise.
    skel_dist = skel_dist[skel_dist > thresh]

    # Scale to pixel size
    skel_dist = [v * pixel_size for v in skel_dist]
    all_measurements = [v * pixel_size for v in all_measurements]

    if show_steps:
        visualize.show_hist(skel_dist, gauss=True, log=True, title=title,
                         xlabel='Channel width [nm]')

    if pdf is not None:
        inout.pdfSaveHist(pdf, skel_dist, gauss=True, log=True,
                          title=title,
                          xlabel='Channel width [nm]')

    average = np.average(skel_dist)
    SD = np.std(skel_dist)

    return average, SD, all_measurements




def channel_length(mask, secondary_size=0, showSteps=False, pdf=None, title='Channel length', pixel_size = 1):
    """
    Calculates the average between node ligament length and standard deviation
    of the passed mask.  Only meaningful on the fully connected phase.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.
    returnAll : bool, optional
        If set to true, a list of all the measurements will be returned.
    showSteps : bool, optional
        If set to true, figures will be displayed that shows the results of
        intermediate steps when calculating the diameter.
    pdf : matplotlib.backends.backend_pdf.PdfPages, optional
        If provided, information will be output to the pdf file.

    Returns
    -------
    logavg : float64
        Average ligament length.  (Assumes lognormal distribution.)
    logstd : float64
        Standard deviation of ligament diameters.
    allMeasurements: 1D array of float64, optional
        A list of all measured ligament lengths.
    """

    # Preprocess the mask for measurement
    mask = ~mask
    rows, cols = mask.shape
    mask = remove_small_objects(mask, secondary_size)
    mask = remove_small_holes(mask, 1000)
    mask = segment.smooth_edges(mask, max(1, rows // 1000))

    estSize = 30
    # Debug mode
    debug = False
    if debug:
        showSteps = True

    # Adjustable parameters.
    # These have been highly tested and changing these may throw off results.
    small_lig = 0.6 * estSize  # Ligaments smaller than this are within a node.
    filter_out_mult = 0.4  # This * estSize is ignored in final calculation.


    rows, cols = mask.shape
    skeleton = skel(mask, estSize, removeEdge=True, times=3)

    if debug:
        visualize.show_skel(skeleton, mask)

    # get ligament and node labels
    nodes = np.copy(skeleton)
    ligaments = np.copy(skeleton)
    neighbors = convolve2d(skeleton, np.ones((3, 3)), mode='same')
    neighbors = neighbors * skeleton
    nodes[neighbors < 4] = 0
    ligaments[neighbors > 3] = 0
    ligaments = label(ligaments, background=0)
    nodes = binary_dilation(nodes, selem=disk(3))
    nodes = label(nodes, background=0)

    # get a list of ligaments connected to each node
    node_to_lig = []  # node_to_lig[n] is an array of each ligament label that is connected to node n
    unodes = np.unique(nodes)
    for n in unodes:
        node_to_lig.append(np.unique(ligaments[nodes == n]))

    # get a list of nodes connected to each ligament
    lig_to_node = []  # lig_to_node[l] is an array of each node label that is connected to ligament l
    uligs = np.unique(ligaments)
    for l in uligs:
        lig_to_node.append(np.unique(nodes[ligaments == l]))

    # Get the length of each ligament between nodes.
    lengths = np.bincount(ligaments.flatten())

    # Add ligaments that are within a single node to connected ligaments.
    small = int(round(small_lig))
    too_small = []
    for l in uligs:
        if lengths[l] <= small:
            too_small.append(l)  # keep track of ligaments that are too small
            for connected in lig_to_node[l]:
                if connected > 0 and connected != l:
                    # add half the small ligament to the connected ligaments.
                    lengths[connected] += int(round(lengths[l] / 2, 0))

    # Set to True to show which ligaments are considered to be within a
    # single node.
    if showSteps:
        ligaments_small = ligaments > 0
        ligaments_small = ligaments_small.astype('uint8')
        ligaments_small[ligaments_small == 1] = 2
        for i in too_small:
            ligaments_small[ligaments == i] = 1
        visualize.show_skel(ligaments_small, mask, dialate=False,
                         title=("Green = within node ligaments, " \
                                "White = between node ligaments"))

    # filter out background and extra small lengths
    lengths[0] = 0
    allMeasurements = lengths[lengths > 0]
    lengths = lengths[lengths > estSize * filter_out_mult]

    # Scale to pixel size
    lengths = [v * pixel_size for v in lengths]
    allMeasurements = [v * pixel_size for v in allMeasurements]

    if len(lengths) == 0:
        return 0, 0, 0

    # Get a lognormal fit.
    fit = lognorm.fit(lengths, floc=0)
    pdf_fitted = lognorm.pdf(lengths,
                             fit[0], loc=fit[1], scale=fit[2])

    # Get gaussian fit.
    gfit = norm.fit(lengths)

    # Get average and standard deviation for lognormal fit.
    logaverage = lognorm.mean(fit[0], loc=fit[1], scale=fit[2])
    logstd = lognorm.std(fit[0], loc=fit[1], scale=fit[2])

    # Get average and standard deviation for normal fit.
    average = norm.mean(gfit[0], gfit[1])
    std = norm.std(gfit[0], gfit[1])
    numbMeasured = len(lengths)

    if showSteps:
        visualize.show_skel(ligaments, mask, dialate=False)

    if showSteps:
        visualize.show_hist(lengths, gauss=True, log=True,
                         title=title,
                         xlabel='Channel length [nm]')

    if pdf is not None:
        inout.pdfSaveSkel(pdf, ligaments, mask, dialate=True)
        inout.pdfSaveHist(pdf, lengths, gauss=True, log=True,
                          title=title,
                          xlabel='Channel length [nm]')


    return average, std, allMeasurements


def area(label, show_steps=False, pdf=None, pixel_size = 1):
    """
    Calculates the area of pores or seperated ligaments.
    Only meaningful if the phase is not fully connected.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.
    show_steps : bool, optional
        If set to true, figures will be displayed that shows the results of
        intermediate steps when calculating the diameter.

    Returns
    -------
    average : float64
        Average ojbect area.
    SD : float64
        Standard deviation of areas.
    areas: 1D array of float64
        A list of all measured object areas.
    """

    areas = np.bincount(label.flatten())  # Count pixels in each ligament.

    areas[::-1].sort()  # Sort the areas (ignore background).
    areas = [value for value in areas if value != 0]
    areas = areas[1:]

    # Scale to pixel size
    areas = [a * pixel_size**2 for a in areas]

    average = np.average(areas[1:])  # Ignore background and edge blobs.
    std_dev = np.std(areas[1:])



    # animate labels
    if show_steps:
        visualize.show_full(label2rgb(label, bg_label=0))

    return average, std_dev, areas[1:]



class Measured_Micrograph():
    def __init__(self,
                 parent_gui=None,
                 micrograph_path=None,
                 phase_names=None,
                 segmentation_type='NN',
                 NN_model=None,
                 NN_preprocessing_fn=None,
                 segmentation_use_full_image=True,
                 half_size=False,
                 segmnetation_crop_area=None,
                 segmentation_assume_square=True,
                 segmentation_resize=True,
                 save_segment=False,
                 segmentation_measure_scale_bar=False,
                 show_steps=False,
                 pixel_size=1,
                 unit='nm',
                 output_folder=None,
                 output_excel=False,
                 save_pdf=False,
                 run_calculate=True,
                 ):

        # Parameter variables
        if parent_gui is not None:
            self.gui = parent_gui
        else:
            self.gui = self

        self.micrograph_path = micrograph_path
        self.output_folder = output_folder
        self.micrograph_folder = None
        self.micrograph_fname = None
        self.micrograph_ftype = None
        self.raw_micrograph = None
        self.save_segment = save_segment
        self.segmentation_use_full_image = segmentation_use_full_image
        self.half_size = half_size
        self.segmentation_crop_area = segmnetation_crop_area
        self.segmentation_assume_square = segmentation_assume_square
        self.segmentation_measure_scale_bar = segmentation_measure_scale_bar
        self.segmentation_resize = segmentation_resize
        self.selected_area = self.raw_micrograph  # Part of image to perform analysis on
        self.pixel_size = pixel_size
        self.unit = unit
        self.segmentation_type = segmentation_type
        self.NN_model = NN_model
        self.NN_preprocessing_fn = NN_preprocessing_fn
        self.phase_names = phase_names
        self.output_excel = output_excel
        self.save_pdf = save_pdf
        self.show_steps = show_steps

        # Output files
        self.pdf = None
        self.excel = None

        # segmentation variables
        self.rows = None
        self.cols = None
        self.mask = None



        # Features
        self.features = []

        # Run
        if run_calculate:
            self.calculate()

    def load_micrograph(self, path=None):
        if path is None:
            path = self.micrograph_path

        # Return 0 if the file does not exist.
        if os.path.isfile(path):
            self.raw_micrograph = inout.load(path, convert_to_uint8=False)
            self.raw_micrograph = inout.uint8(self.raw_micrograph)
            if self.show_steps: visualize.show_full(self.raw_micrograph, title='Loaded micrograph.')
            self.micrograph_fname = Path(path).stem
            if self.gui is not None:
                self.gui.write_results('Micrograph:', self.micrograph_fname)
                self.gui.write_status('Operating on ', self.micrograph_fname)
            self.features.append(Feature(None, {'File name': self.micrograph_fname}))
            self.micrograph_ftype = os.path.splitext(path)
            self.micrograph_folder, _ = os.path.split(path)
            if self.output_folder is None:
                self.output_folder = self.micrograph_folder

            # Grab the image data and scale information.
            if self.segmentation_use_full_image:
                self.selected_area = self.raw_micrograph
            elif self.segmentation_crop_area is not None:
                y0, y1, x0, x1 = self.segmentation_crop_area
                self.selected_area = self.raw_micrograph[y0:y1, x0:x1]
            else:
                self.selected_area, scale = gui.manualSelectImage(
                    self.raw_micrograph, self.segmentation_measure_scale_bar
                )
                if self.segmentation_measure_scale_bar:
                    self.selected_area = inout.uint8(self.selected_area)
                    scale = inout.uint8(scale)
                    self.pixel_size, self.unit = gui.manualPixelSize(scale)
                    self.gui.write_results("The pixel size is %.4f %s" % (self.pixel_size, self.unit))

            # Grab the scale information from the file name if appropriate.
            if 'pixel_size=' in self.micrograph_fname:
                # file name should contain pixel_size=###.##_ with trailing underscore (any number of digits).
                val = self.micrograph_fname.split('pixel_size=')[1]
                val = val.split('_')
                self.pixel_size = float(val)

            # Get image size
            if self.selected_area.ndim == 2:
                self.rows, self.cols = self.selected_area.shape
            elif self.selected_area.ndim == 3:
                self.rows, self.cols, _ = self.selected_area.shape

            # if should: make square
            if not self.segmentation_measure_scale_bar and self.segmentation_assume_square:
                if self.rows > self.cols:
                    self.selected_area = self.selected_area[:self.cols, :self.cols, ...]
                    self.rows, self.cols = self.selected_area.shape

            # if should: resize
            if self.segmentation_resize and self.segmentation_use_full_image:
                rescale_factor = 1024 / self.cols
                self.pixel_size = self.pixel_size / rescale_factor
                self.selected_area = resize(self.selected_area, (int(round(self.rows*rescale_factor)),1024))
                self.rows, self.cols = self.selected_area.shape

            if self.half_size:
                self.selected_area = resize(self.selected_area, (int(round(self.rows/2)), int(round(self.cols/2))))
                self.pixel_size = self.pixel_size * 2
                self.rows, self.cols = self.selected_area.shape[:2]



            self.features.append(Feature(None, {'Pixel size': self.pixel_size}))
            self.features.append(Feature(None, {'Unit': self.unit}))
            return 1
        else:
            return 0

    def prep_pdf(self):
        if self.output_folder == None:
            self.pdf = PdfPages(os.path.join(self.micrograph_folder,
                                             ''.join((self.micrograph_fname, '_Summary.pdf'))))
        else:
            self.pdf = PdfPages(os.path.join(self.output_folder,
                                             ''.join((self.micrograph_fname, '_Summary.pdf'))))
            
        cmap = None
        if len(self.raw_micrograph.shape) == 2:
            cmap = plt.cm.gray

        inout.pdfSaveImage(self.pdf, self.raw_micrograph, title="Micrograph.", cmap=cmap)
        inout.pdfSaveImage(self.pdf, self.selected_area, title="Selected measurement area.", cmap=cmap)

    def save_pdf_do(self):
        # Create summary
        fig = plt.figure(figsize=(8, int(len(self.features)/3)))
        #fig = plt.figure(figsize=(8, 12))

        plt.title('Summary')
        plt.axis('off')
        #fig = plt.figure(figsize=(10,20))
        fs = 10
        yloc = 0.95
        xloc = 0.01
        space = min(1/len(self.features), 0.03)
        for f in self.features:
            if f.title is None:
                for t, v in f.values.items():
                    try:
                        out = ''.join((t, ': ', v))
                    except TypeError:
                        out = ''.join((t, ': ', str(round(v, 4))))
                    plt.text(xloc, yloc, out, fontsize=fs)
                    yloc -= space
            else:
                out = ''.join((f.title, ': ', str(round(f.values['average'], 4)), ' ± ',
                               str(round(f.values['SD'], 4))))
                plt.text(xloc, yloc, out, fontsize=fs)
                yloc -= space
        self.pdf.savefig()
        plt.close()
        self.pdf.close()

    def save_binary(self):
        out_name = ''.join((self.micrograph_fname, '_binary_pixel_size=', str(self.pixel_size),
                            '_.png'))
        plt.imsave(self.mask, os.path.join(self.output_folder, out_name), cmap=plt.cm.gray)

    def segment(self):
        self.gui.write_status("Segmenting...", end=' ')
        t0 = time.time()
        if self.segmentation_type == 'binary':
            self.mask = self.selected_area > 0
        elif self.segmentation_type == "otsu":
            self.mask = segment.otsu_segment(self.selected_area)
            self.mask = ~self.mask
        elif self.segmentation_type == "manual":
            self.mask = segment.manualSegment(self.selected_area)
        elif self.segmentation_type == "accurate":
            self.mask = segment.local_otsu_segment(self.selected_area)
        elif self.segmentation_type == "NN":
            im = self.selected_area
            im = gray2rgb(im)
            im = img_as_ubyte(im)
            self.mask = segment.segmentation_models_inference(im,
                                                              self.NN_model,
                                                              self.NN_preprocessing_fn,
                                                              #patch_size=1024,
                                                              num_classes=3)
        self.gui.write_status('Done! Took', round(time.time() - t0, 2), 'seconds.')
        if self.segmentation_type != 'binary':
            if self.save_pdf:
                # add overlay to pdf
                inout.pdfSaveOverlay(self.pdf, self.selected_area, self.mask, title="Segmentation mask.")
                # save image of overlay
                imageio.imsave(
                    uri=self.micrograph_folder + '\\' + self.micrograph_fname + '_mask.png',
                    im=visualize.overlay_mask(self.selected_area, self.mask, return_overlay=True)
                )
                
            if self.show_steps:
                visualize.overlay_mask(self.selected_area, self.mask, title='Segmentation mask.',animate=True)

        assert len(self.phase_names) == self.mask.shape[-1] + 1, "Length of phase names does not match number of mask channels + 1 (for matrix)"

    def calculate_area_fraction(self):
        self.gui.write_status("Calculating area fraction...", end=' ')
        t0 = time.time()
        for i, p in enumerate(self.phase_names):
            if i == 0:
                area_fraction = np.count_nonzero(get_background(self.mask)) / self.rows / self.cols
            else:
                area_fraction = np.count_nonzero(self.mask[:,:,i-1]) / self.rows / self.cols
            self.features.append(Feature(None, {p + ' area fraction': area_fraction}))
            self.gui.write_results(p, 'area fraction:', round(area_fraction, 3))
        self.gui.write_status('Done! Took', round(time.time() - t0, 2), 'seconds.')


    def write_status(self, *strings, sep=' ', end='\n'):
        print(*strings, sep=sep, end=end)

    def write_results(self, *strings, sep=' ', end='\n'):
        print(*strings, sep=sep, end=end)


class Measured_EBC_Oxide_Micrograph(Measured_Micrograph):
    def __init__(self, cutoff_wl, remove_small, *args, **kwargs):
        self.cutoff_wl = cutoff_wl
        self.remove_small = remove_small
        super().__init__(*args, **kwargs)
        #super(Measured_Precipitate_Micrograph, self).__init__(*args, **kwargs)

    def blur_and_increase_brightness(self, im):
    # Apply Gaussian blur
        blurred_image = gaussian(im, sigma=2, mode='nearest')

        # Increase brightness by scaling pixel values (clipping at 1.0 for float images)
        brighter_image = np.clip(blurred_image * 1.5, 0, 1)

        return brighter_image


    def calculate(self):

        self.load_micrograph()
        self.selected_area = inout.uint8(self.selected_area)
        #self.selected_area = self.blur_and_increase_brightness(self.selected_area)
        if self.save_pdf: self.prep_pdf()
        self.segment()
        mult = 1 if self.rows < 1200 else 2
        self.thickness_mask = None
        #self.thickness_mask = self.clean_oxide(self.mask, 100, 2500, 'Cleaned mask for thickness meas.').copy()
        self.mask = self.clean_oxide(self.mask, self.remove_small, 2500*mult) # was 2500*mult, 2500*mult
        self.thickness_mask = self.mask if self.thickness_mask is None else self.thickness_mask
        if self.save_segment: self.save_binary()
        self.selected_area=inout.uint8(self.selected_area)

        try:
            porosity, pores = self.get_pores(self.selected_area, self.mask)
        except Exception:
            print('\n\nERROR CALCULATING POROSITY\n')
            self.gui.write_status('ERROR CALCULATING POROSITY', also_print=False)
            traceback.print_exc()
            time.sleep(0.1)
            print('\n\n')
            porosity, pores = np.nan, np.nan

        try:
            thickness_average, thickness_SD, skel_dist, length = self.oxide_thickness(self.thickness_mask)
        except Exception:
            print('\n\nERROR CALCULATING OXIDE THICKNESS\n')
            self.gui.write_status('ERROR CALCULATING OXIDE THICKNESS', also_print=False)
            traceback.print_exc()
            time.sleep(0.1)
            print('\n\n')
            thickness_average, thickness_SD, skel_dist= np.nan, np.nan, np.nan

        try:
            crack_spacing, spacing_SD, num_cracks, length_div_count_dists, all_crack_measurments = \
                self.connect_cracks(self.selected_area, self.mask, pores, thickness_average, skel_dist, length)
        except Exception:
            print('\n\nERROR CALCULATING CRACK SPACING\n')
            self.gui.write_status('ERROR CALCULATING CRACK SPACING', also_print=False)
            traceback.print_exc()
            time.sleep(0.1)
            print('\n\n')
            crack_spacing, spacing_SD, num_cracks, length_div_count_dists, all_crack_measurments = \
                np.nan, np.nan, np.nan, np.nan, np.nan

        self.cutoff_wl_unit = self.cutoff_wl
        self.cutoff_wl = int(round(self.cutoff_wl / self.pixel_size))  # convert cutoff wl to pixels
        self.gui.write_results("Cutoff Length: %d pixels" % self.cutoff_wl)

        try:
            micro_top_Ra, micro_top_Rq, micro_top_Rp, micro_top_Rv, micro_top_Rt, micro_top_RSk, \
                micro_bot_Ra, micro_bot_Rq, micro_bot_Rp, micro_bot_Rv, micro_bot_Rt, micro_bot_RSk, \
                macro_top_Ra, macro_top_Rq, macro_top_Rp, macro_top_Rv, macro_top_Rt, macro_top_RSk, \
                macro_bot_Ra, macro_bot_Rq, macro_bot_Rp, macro_bot_Rv, macro_bot_Rt, macro_bot_RSk, \
                total_top_Ra, total_top_Rq, total_top_Rp, total_top_Rv, total_top_Rt, total_top_RSk, \
                total_bot_Ra, total_bot_Rq, total_bot_Rp, total_bot_Rv, total_bot_Rt, total_bot_RSk = \
                self.get_roughness(self.cutoff_wl)
        except Exception:
            print('\n\nERROR CALCULATING ROUGHNESS\n')
            self.gui.write_status('ERROR CALCULATING ROUGHNESS', also_print=False)
            traceback.print_exc()
            time.sleep(0.1)
            print('\n\n')
            micro_top_Ra, micro_top_Rq, micro_top_Rp, micro_top_Rv, micro_top_Rt, micro_top_RSk, \
                micro_bot_Ra, micro_bot_Rq, micro_bot_Rp, micro_bot_Rv, micro_bot_Rt, micro_bot_RSk, \
                macro_top_Ra, macro_top_Rq, macro_top_Rp, macro_top_Rv, macro_top_Rt, macro_top_RSk, \
                macro_bot_Ra, macro_bot_Rq, macro_bot_Rp, macro_bot_Rv, macro_bot_Rt, macro_bot_RSk, \
                total_top_Ra, total_top_Rq, total_top_Rp, total_top_Rv, total_top_Rt, total_top_RSk, \
                total_bot_Ra, total_bot_Rq, total_bot_Rp, total_bot_Rv, total_bot_Rt, total_bot_RSk = \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan



        self.features.append(Feature(None, {'Porosity': porosity}))
        self.features.append(Feature(
            title="Oxide thickness (%s)" %self.unit,
            values={'average': thickness_average,
                    'SD': thickness_SD,
                    'length': len(skel_dist),
                    'all_measurements': skel_dist}))
        self.features.append(Feature(
            title="Crack spacing (%s)" %self.unit,
            values={'average': crack_spacing,
                    'SD': spacing_SD,
                    'num cracks': num_cracks,
                    'all_measurements': all_crack_measurments}))
        self.features.append(Feature(None, {'Crack spacing (length/count) (%s)' %self.unit: length_div_count_dists}))

        self.features.append(Feature(None, {"Cutoff Length (pixels)": self.cutoff_wl}))
        self.features.append(Feature(None, {'Cutoff Length (%s)' %self.unit: self.cutoff_wl_unit}))

        self.features.append(Feature(None, {"Micro top Ra": micro_top_Ra}))
        self.features.append(Feature(None, {"Micro top Rq": micro_top_Rq}))
        self.features.append(Feature(None, {"Micro top Rp": micro_top_Rp}))
        self.features.append(Feature(None, {"Micro top Rv": micro_top_Rv}))
        self.features.append(Feature(None, {"Micro top Rt": micro_top_Rt}))
        self.features.append(Feature(None, {"Micro top RSk": micro_top_RSk}))
        self.features.append(Feature(None, {"Micro bot Ra": micro_bot_Ra}))
        self.features.append(Feature(None, {"Micro bot Rq": micro_bot_Rq}))
        self.features.append(Feature(None, {"Micro bot Rp": micro_bot_Rp}))
        self.features.append(Feature(None, {"Micro bot Rv": micro_bot_Rv}))
        self.features.append(Feature(None, {"Micro bot Rt": micro_bot_Rt}))
        self.features.append(Feature(None, {"Micro bot RSk": micro_bot_RSk}))
        
        self.features.append(Feature(None, {"Macro top Ra": macro_top_Ra}))
        self.features.append(Feature(None, {"Macro top Rq": macro_top_Rq}))
        self.features.append(Feature(None, {"Macro top Rp": macro_top_Rp}))
        self.features.append(Feature(None, {"Macro top Rv": macro_top_Rv}))
        self.features.append(Feature(None, {"Macro top Rt": macro_top_Rt}))
        self.features.append(Feature(None, {"Macro top RSk": macro_top_RSk}))
        self.features.append(Feature(None, {"Macro bot Ra": macro_bot_Ra}))
        self.features.append(Feature(None, {"Macro bot Rq": macro_bot_Rq}))
        self.features.append(Feature(None, {"Macro bot Rp": macro_bot_Rp}))
        self.features.append(Feature(None, {"Macro bot Rv": macro_bot_Rv}))
        self.features.append(Feature(None, {"Macro bot Rt": macro_bot_Rt}))
        self.features.append(Feature(None, {"Macro bot RSk": macro_bot_RSk}))
        
        self.features.append(Feature(None, {"Total top Ra": total_top_Ra}))
        self.features.append(Feature(None, {"Total top Rq": total_top_Rq}))
        self.features.append(Feature(None, {"Total top Rp": total_top_Rp}))
        self.features.append(Feature(None, {"Total top Rv": total_top_Rv}))
        self.features.append(Feature(None, {"Total top Rt": total_top_Rt}))
        self.features.append(Feature(None, {"Total top RSk": total_top_RSk}))
        self.features.append(Feature(None, {"Total bot Ra": total_bot_Ra}))
        self.features.append(Feature(None, {"Total bot Rq": total_bot_Rq}))
        self.features.append(Feature(None, {"Total bot Rp": total_bot_Rp}))
        self.features.append(Feature(None, {"Total bot Rv": total_bot_Rv}))
        self.features.append(Feature(None, {"Total bot Rt": total_bot_Rt}))
        self.features.append(Feature(None, {"Total bot RSk": total_bot_RSk}))


        if self.save_pdf: self.save_pdf_do()
        if self.gui is not None:
            self.gui.write_status('')
            self.gui.write_results('')

    def clean_oxide(self, mask, small_objects=2500, small_holes=2500, title="Cleaned segmentation mask."):
        self.gui.write_status("Performing morphology operations...", end=' ')
        t0 = time.time()
        oxide = mask[:, :, 0]
        crack = mask[:, :, 1]
        oxide = (crack + oxide) > 0  # bitwise and
        oxide = segment.removeSmallObjects(oxide, small_objects, small_holes)
        crack = crack * binary_erosion(oxide, selem=np.ones((15, 15)))
        oxide = oxide ^ crack
        mask[:, :, 0] = oxide
        mask[:, :, 1] = crack
        if self.save_pdf:
            inout.pdfSaveOverlay(self.pdf, self.selected_area, self.mask, title=title)
        if self.show_steps:
            visualize.overlay_mask(self.selected_area, self.mask, title=title, animate=True)
        self.gui.write_status('Done! Took', round(time.time() - t0, 2), 'seconds.')
        return mask

    def get_pores(self, im, mask, gaussian_radius=1, peak_prominence=500):
        self.gui.write_status("Measuring porosity...", end=' ')
        im = rgb2gray(im) # gray scale for this
        t0 = time.time()
        pdf = self.pdf
        rows, cols = im.shape
        oxide = mask[:, :, 0]
        crack = mask[:, :, 1]
        oxide_layer = crack ^ oxide
        if len(im.shape) == 3:
            im_smooth = img_as_ubyte(rgb2gray(gaussian(im, gaussian_radius)))
        else:
            im_smooth = img_as_ubyte(gaussian(im, gaussian_radius))

        eroded_mask = binary_erosion(oxide_layer, selem=np.ones((15, 15)))
        masked_array_mask = np.ones_like(im)
        masked_array_mask[eroded_mask] = 0
        im_masked = np.ma.masked_array(im, masked_array_mask)
        otsu_thresh = threshold_otsu(im_masked.compressed())

        #needed for proper scaling
        im_masked[0,0] = 0
        im_masked[0,1] = 255

        bin_counts, bin_edges = np.histogram(im_masked.compressed(), 256)
        # fix 0 values in hist (from dtype change)
        for i in range(1, 255):
            if bin_counts[i] == 0:
                bin_counts[i] = (bin_counts[i - 1] + bin_counts[i + 1]) / 2

        peaks, _ = find_peaks(bin_counts, prominence=peak_prominence)  # find peaks

        smoothed = True
        bin_counts = savgol_filter(bin_counts, 21, 3)  # smooth peaks
        peaks, _ = find_peaks(bin_counts, prominence=peak_prominence)  # find peaks

        num_loops = 0
        desired_peaks = 1 if smoothed else 1
        gray_peak = 0 if smoothed else 0
        while len(peaks) != desired_peaks:
             if len(peaks) > desired_peaks:
                peak_prominence = peak_prominence * 1.3
                peaks, _ = find_peaks(bin_counts, prominence=peak_prominence)
             if len(peaks) < desired_peaks:
                peak_prominence = peak_prominence * 0.9
                peaks, _ = find_peaks(bin_counts, prominence=peak_prominence)
             if num_loops == 100 and len(peaks) == 1:
                 break
             if num_loops == 150:
                 break
             else:
                 num_loops += 1

        if len(peaks) == desired_peaks:
            results_half = peak_widths(bin_counts, peaks, rel_height=0.5)
            results_full = peak_widths(bin_counts, peaks, rel_height=1)
            results_70 = peak_widths(bin_counts, peaks, rel_height=0.7)
            results_80 = peak_widths(bin_counts, peaks, rel_height=0.8)
            results_90 = peak_widths(bin_counts, peaks, rel_height=0.9)
            left_gray_70 = results_70[2][gray_peak]
            plot_thresh = int(round(left_gray_70))
            left_gray_80 = results_80[2][gray_peak]
            thresh_80 = int(round(left_gray_80))
            left_gray_90 = results_90[2][gray_peak]
            thresh_90 = int(round(left_gray_90))

        else: # guess we're going to hard code it for now :(
            print('HARD CODING THRESHOLD')
            plot_thresh = 50
            thresh_80 = 50
            results_half = peak_widths(bin_counts, peaks, rel_height=0.5)
            results_70 = peak_widths(bin_counts, peaks, rel_height=0.7)
            results_80 = peak_widths(bin_counts, peaks, rel_height=0.8)
            results_90 = peak_widths(bin_counts, peaks, rel_height=0.9)

        if len(peaks) != desired_peaks:
            print("WARNING: " + str(desired_peaks) + " PEAKS NOT FOUND IN HISTOGRAM")
            print(peaks)

        thresh=plot_thresh

        # Otsu's thresh
        pores = (im_smooth < otsu_thresh) * eroded_mask
        pores = segment.removeSmallObjects(pores, 10, 200)

        pores_minus_crack = np.copy(pores)
        cracks_to_remove = binary_dilation(segment.removeSmallObjects(crack, 30, 30), np.ones((3, 3)))
        pores_minus_crack[cracks_to_remove == 1] = 0
        pores_minus_crack = segment.removeSmallObjects(pores_minus_crack, 10, 200)

        try:
            porosity = np.count_nonzero(pores_minus_crack) / np.count_nonzero(eroded_mask)
        except ZeroDivisionError:
            porosity = 0

        # if pdf is not None:
        #     inout.pdfSaveOverlay(pdf, im, pores_minus_crack,
        #                          title='Porosity = %0.3f, threshold = %.0f (otsu)' % (porosity, otsu_thresh))

        # Otsu's 1.1 thresh
        otsu_thresh = int(round(otsu_thresh))
        otsu_thresh_1_2 = int(round(otsu_thresh * 1.1))
        pores = (im_smooth < otsu_thresh_1_2) * eroded_mask
        pores = segment.removeSmallObjects(pores, 10, 200)

        pores_minus_crack = np.copy(pores)
        cracks_to_remove = binary_dilation(segment.removeSmallObjects(crack, 30, 30), np.ones((3, 3)))
        pores_minus_crack[cracks_to_remove == 1] = 0
        pores_minus_crack = segment.removeSmallObjects(pores_minus_crack, 10, 200)

        try:
            porosity = np.count_nonzero(pores_minus_crack) / np.count_nonzero(eroded_mask)
        except ZeroDivisionError:
            porosity = 0

        # if pdf is not None:
        #     inout.pdfSaveOverlay(pdf, im, pores_minus_crack,
        #                          title='Porosity = %0.3f, threshold = %.0f (otsu*1.1)' % (porosity, otsu_thresh_1_2))

        # 70% thresh
        pores = (im_smooth < thresh) * eroded_mask
        pores = segment.removeSmallObjects(pores, 10, 200)

        pores_minus_crack = np.copy(pores)
        cracks_to_remove = binary_dilation(segment.removeSmallObjects(crack, 30, 30), np.ones((3,3)))
        pores_minus_crack[cracks_to_remove == 1] = 0
        pores_minus_crack = segment.removeSmallObjects(pores_minus_crack, 10, 200)

        ## this appears to be the final one.
        try:
            porosity = np.count_nonzero(pores_minus_crack) / np.count_nonzero(eroded_mask)
        except ZeroDivisionError:
            porosity = 0

        if pdf is not None:
            inout.pdfSaveOverlay(pdf, im, pores_minus_crack, title='Porosity = %.3f, threshold = %.0f (70)' % (porosity, thresh))
        if self.show_steps:
            visualize.show_full(visualize.overlay_mask(im, pores_minus_crack, return_overlay=True),
                                title='Porosity = %.3f, threshold = %.0f (70)' % (porosity, thresh))

        # 80% thresh
        pores = (im_smooth < thresh_80) * eroded_mask
        pores = segment.removeSmallObjects(pores, 10, 200)

        pores_minus_crack = np.copy(pores)
        cracks_to_remove = binary_dilation(segment.removeSmallObjects(crack, 30, 30), np.ones((3, 3)))
        pores_minus_crack[cracks_to_remove == 1] = 0
        pores_minus_crack = segment.removeSmallObjects(pores_minus_crack, 10, 200)

        try:
            porosity_80 = np.count_nonzero(pores_minus_crack) / np.count_nonzero(eroded_mask)
        except ZeroDivisionError:
            porosity_80 = 0

        # if pdf is not None:
        #     inout.pdfSaveOverlay(pdf, im, pores_minus_crack,
        #                          title='Porosity = %0.3f, threshold = %.0f (80)' % (porosity_80, thresh_80))

        # 90% thresh
        pores = (im_smooth < thresh_90) * eroded_mask
        pores = segment.removeSmallObjects(pores, 10, 200)

        pores_minus_crack = np.copy(pores)
        cracks_to_remove = binary_dilation(segment.removeSmallObjects(crack, 30, 30), np.ones((3, 3)))
        pores_minus_crack[cracks_to_remove == 1] = 0
        pores_minus_crack = segment.removeSmallObjects(pores_minus_crack, 10, 200)

        try:
            porosity_90 = np.count_nonzero(pores_minus_crack) / np.count_nonzero(eroded_mask)
        except ZeroDivisionError:
            porosity_90 = 0

        # if pdf is not None:
        #     inout.pdfSaveOverlay(pdf, im, pores_minus_crack,
        #                          title='Porosity = %0.3f, threshold = %.0f (90)' % (porosity_90, thresh_90))


        if pdf is not None:  # plot peaks
            plt.plot(bin_counts)
            plt.plot(peaks, bin_counts[peaks], "x")
            plt.plot(plot_thresh, bin_counts[plot_thresh], "|", ms=20, color='red')
            plt.plot(otsu_thresh, bin_counts[otsu_thresh], "|", ms=20, color='green')
            plt.plot(otsu_thresh_1_2, bin_counts[otsu_thresh_1_2], "|", ms=20, color='lime')
            plt.plot(thresh_80, bin_counts[thresh_80], "|", ms=20, color='orange')
            plt.plot(thresh_90, bin_counts[thresh_90], "|", ms=20, color='yellow')
            plt.xlabel('pixel intensity')
            plt.ylabel('count')
            plt.title('Threshold from smoothed intensity histogram')
            #plt.hlines(*results_half[1:], color="C2")
            plt.hlines(*results_70[1:], color="C3")
            plt.tight_layout()
            pdf.savefig()
            plt.close()



        if self.show_steps:
            plt.plot(bin_counts)
            plt.plot(peaks, bin_counts[peaks], "x")
            plt.plot(plot_thresh, bin_counts[plot_thresh], "|", ms=20, color='red')
            plt.plot(otsu_thresh, bin_counts[otsu_thresh], "|", ms=20, color='green')
            plt.plot(otsu_thresh_1_2, bin_counts[otsu_thresh_1_2], "|", ms=20, color='lime')
            plt.plot(thresh_80, bin_counts[thresh_80], "|", ms=20, color='orange')
            plt.plot(thresh_90, bin_counts[thresh_90], "|", ms=20, color='yellow')
            plt.xlabel('pixel intensity')
            plt.ylabel('count')
            plt.title('Threshold from smoothed intensity histogram')
            #plt.hlines(*results_half[1:], color="C2")
            plt.hlines(*results_70[1:], color="C3")
            plt.tight_layout()
            plt.show()

        self.gui.write_status('Done! Took', round(time.time() - t0, 2), 'seconds.')
        self.gui.write_results('Porosity: %0.4f' % porosity)

        return porosity, pores

    def label_overlay(self, im, label):
        image_label_overlay = label2rgb(label, image=im, alpha=1, bg_label=0, image_alpha=1, bg_color=(0, 0, 0))
        black_pixels_mask = np.all(image_label_overlay == [0, 0, 0], axis=-1)
        image_label_overlay[black_pixels_mask] = gray2rgb(im)[black_pixels_mask] / 255
        return image_label_overlay

    def oxide_thickness(self, mask):
        self.gui.write_status("Measuring oxide thickness...", end=' ')
        t0 = time.time()
        pdf = self.pdf
        oxide = mask[:, :, 0]
        crack = mask[:, :, 1]
        oxide = crack ^ oxide

        # oxide_labeled = label(oxide)
        # reg = regionprops(oxide_labeled)[0]
        # length = reg['major_axis_length']

        dist = distance_transform_edt(oxide)
        skel = np.zeros_like(dist)
        skel[np.where(dist == np.max(dist, axis=0))] = 1
        skel = binary_dilation(skel, selem=np.ones((3, 3)))
        skel = skeletonize(skel)
        skel_dist = skel * dist
        skel_dist = skel_dist[skel_dist > 2]
        skel_dist = [i*2*self.pixel_size for i in skel_dist] # convert to diameter and pixel size
        average = np.average(skel_dist)
        SD = np.std(skel_dist)

        locs = np.argwhere(skel)
        row_range = len(np.unique(locs[:, 0]))
        # print(col_range)
        length = math.sqrt(self.cols ** 2 + row_range ** 2)

        if pdf is not None:
            inout.pdfSaveSkel(pdf, skel * dist, oxide, dialate=True,
                              title='Oxide thickness = %.2f ± %.2f, length = %.0f' % (average, SD, length))
            inout.pdfSaveHist(pdf, skel_dist, title='Layer thickness', numBins=20)
        if self.show_steps:
            visualize.show_skel(skel * dist, oxide, dialate=True,
                              title='Oxide thickness = %.2f ± %.2f, length = %.0f' % (average, SD, length))
            visualize.show_hist(skel_dist, title='Layer thickness', numBins=20)

        self.gui.write_status('Done! Took', round(time.time() - t0, 2), 'seconds.')
        self.gui.write_results('Oxide thickness: %.3f ± %.3f %s' % (average, SD, self.unit))
        return average, SD, skel_dist, length

    def connect_cracks(self, im, mask, pores, thickness, skel_dist, length):
        # convert thickness back to pixels
        thickness /= self.pixel_size
        skel_dist = [i / self.pixel_size for i in skel_dist]
        self.gui.write_status("Measuring cracks...", end=' ')
        t0 = time.time()
        pdf = self.pdf
        CLOSE_THICKNESS_DIV = 3  # connect close cracks if dist thickness / CLOSE_THICKNESS_DIV (2 worked ok, but close cracks were improp connected)
        if pdf is not None:
            inout.pdfSaveImage(pdf, im, title='Original Micrograph',
                               cmap=plt.cm.gray)  # show again for ease of  visualization
        cracks = mask[:, :, 1]

        crack_labels = label(cracks, background=0)
        if pdf is not None:
            inout.pdfSaveImage(pdf, self.label_overlay(im, crack_labels), title='Labelled cracks')
        if self.show_steps:
            visualize.show_full(self.label_overlay(im, crack_labels), title='Labelled cracks')

        cracks = segment.removeSmallObjects(cracks, 10, 10)



        crack_labels = label(cracks, background=0)
        # if pdf is not None:
        #     inout.pdfSaveImage(pdf, self.label_overlay(im, crack_labels), title='Removed small cracks')
        # if self.show_steps:
        #     visualize.show_full(self.label_overlay(im, crack_labels), title='Removed small cracks')
        crack_props = regionprops(crack_labels)

        if len(pores.shape) == 3:  # too many channels
            pores = pores[:, :, 0]

        pores = cracks ^ pores

        pores = binary_dilation(pores, selem=np.ones((7, 7)))
        pore_labels = label(pores, background=0)
        pore_props = regionprops(pore_labels, crack_labels)

        connection_hist = {}
        for pore in pore_props:
            labels_in_pore = np.unique(pore['intensity_image'])  # find number of cracks overlapping pore
            if len(labels_in_pore) > 2:  # first is background, need 3 or more for 2 cracks in 1 pore.
                for a, b in itertools.combinations(labels_in_pore[1:], r=2):
                    # print('cracks %i and %i' %(a, b))
                    # if the x center of the cracks are further apart than the thickness then don't connect
                    dist = abs(crack_props[b - 1]['centroid'][1] - crack_props[a - 1]['centroid'][1])
                    if dist > thickness / CLOSE_THICKNESS_DIV:
                        # print('cracks %i and %i too far apart: %.2f' %(a, b, dist))
                        pass
                    else:
                        if str(a) in connection_hist.keys():
                            # print('hist combining crack %i and %i.' %(connection_hist[str(a)], b))
                            crack_labels[crack_labels == b] = connection_hist[str(a)]
                            connection_hist[str(b)] = connection_hist[str(a)]
                        else:
                            # print('combining crack %i and %i.' %(a, b))
                            crack_labels[crack_labels == b] = a
                            # connection_hist[str(labels_in_pore[1])] = lab
                            connection_hist[str(b)] = a

        # if pdf is not None:
        #     inout.pdfSaveImage(pdf, self.label_overlay(im, crack_labels), title='Connected cracks that should be')
        # if self.show_steps:
        #     visualize.show_full(self.label_overlay(im, crack_labels), title='Connected cracks that should be')

        # connect cracks that are over top of eachother
        crack_props = regionprops(crack_labels)
        pix = {}
        for i, cp in enumerate(crack_props):
            new_uniqe_cols = np.unique(np.array(cp['coords'])[:, 1])
            for key in pix:
                # print(pix[key], new_uniqe_cols)
                if any(item in pix[key] for item in new_uniqe_cols):
                    # print('replaced')
                    pix[key] = pix[key] + list(new_uniqe_cols)
                    crack_labels[crack_labels == cp['label']] = int(key)
                    break
            else:
                pix[str(cp['label'])] = list(new_uniqe_cols)

        # if pdf is not None:
        #     inout.pdfSaveImage(pdf, self.label_overlay(im, crack_labels), title='Connected cracks that are over eachother')
        # if self.show_steps:
        #     visualize.show_full(self.label_overlay(im, crack_labels), title='Connected cracks that are over eachother')

        # remove cracks that don't span 1/2 the height
        crack_props = regionprops(crack_labels)
        # print(thickness)
        for i, cp in enumerate(crack_props):
            min_row, min_col, max_row, max_col = cp['bbox']
            this_thickness = np.max(skel_dist[min_col:max_col])
            # print(this_thickness)
            # print(cp['coords'])

            row_pix = len(np.unique(np.array(cp['coords'])[:, 0]))  # number of row pixels detected
            # uni = np.unique(np.array(cp['coords'])[:, 0])
            # if row_pix != max_row - min_row:
            #     print(this_thickness, row_pix, max_row - min_row)
            # removed = False
            if (row_pix) < (this_thickness * 0.25):  # not tall enough
                crack_labels[crack_labels == cp['label']] = 0
                # removed = True
            # print(removed, this_thickness, max_row - min_row, row_pix, uni)


        # if pdf is not None:
        #     inout.pdfSaveImage(pdf, self.label_overlay(im, crack_labels), title='Removed half cracks')
        # if self.show_steps:
        #     visualize.show_full(self.label_overlay(im, crack_labels), title='Removed half cracks')

        if pdf is not None:
            inout.pdfSaveImage(pdf, self.label_overlay(im, crack_labels), title='Cleaned cracks')
        if self.show_steps:
            visualize.show_full(self.label_overlay(im, crack_labels), title='Cleaned cracks')

        # calculate crack distance
        crack_props = regionprops(crack_labels)
        if len(crack_props) < 2:
            return 0, 0, 0, 0, 0
        centroid_row = [c['centroid'][0] for c in crack_props]
        centroid_col = [c['centroid'][1] for c in crack_props]
        centroid_col, centroid_row = (list(t) for t in zip(*sorted(zip(centroid_col, centroid_row))))

        dists = []
        for i in range(len(centroid_row) - 1):
            dists.append(
                np.sqrt((centroid_row[i + 1] - centroid_row[i]) ** 2 + (centroid_col[i + 1] - centroid_col[i]) ** 2))
        # print(dists)

        # convert to pixel size
        dists = [i * self.pixel_size for i in dists]

        length_div_count_dists = length / (len(dists) + 1) if len(dists) + 1 > 1 else 0
        length_div_count_dists *= self.pixel_size

        image_label_overlay = self.label_overlay(im, crack_labels)
        if pdf is not None:
            plt.imshow(image_label_overlay)
            plt.plot(centroid_col, centroid_row, 'wx--', ms=5, mew=2)
            for i in range(len(centroid_row) - 1):
                plt.text((centroid_col[i + 1] + centroid_col[i]) / 2 - 20,
                         (centroid_row[i + 1] + centroid_row[i]) / 2 - 10,
                         '%.f' % dists[i],
                         fontsize=8, color='white')
            plt.axis('off')
            plt.title('Final Crack Spacing: %.1f (between), %.1f (length/count)' % (
            np.average(dists), length_div_count_dists))
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        if self.show_steps:
            plt.imshow(image_label_overlay)
            plt.plot(centroid_col, centroid_row, 'wx--', ms=5, mew=2)
            for i in range(len(centroid_row) - 1):
                plt.text((centroid_col[i + 1] + centroid_col[i]) / 2 - 20,
                         (centroid_row[i + 1] + centroid_row[i]) / 2 - 10,
                         '%.f' % dists[i],
                         fontsize=8, color='white')
            plt.axis('off')
            plt.title('Final Crack Spacing: %.1f (between), %.1f (length/count)' % (
            np.average(dists), length_div_count_dists))
            plt.tight_layout()
            plt.show()

        #self.gui.write_status('Done! Took', round(time.time() - t0, 2), 'seconds.')
        #self.gui.write_results()

        self.gui.write_status('Done! Took', round(time.time() - t0, 2), 'seconds.')
        self.gui.write_results('Crack spacing: %.3f ± %.3f %s' % (np.average(dists), np.std(dists), self.unit))



        return np.average(dists), np.std(dists), len(dists) + 1, length_div_count_dists, dists


    def get_roughness(self, cutoff_wl=100):
        self.gui.write_status("Measuring oxide roughness...", end=' ')
        t0 = time.time()

        # get mask of oxide including cracks
        oxide = self.mask[:, :, 0]
        crack = self.mask[:, :, 1]
        oxide_layer = crack ^ oxide

        # find where oxide starts and ends
        labels = label(oxide_layer)
        props = regionprops_table(labels)
        props = pd.DataFrame(props)
        props['width'] = props["bbox-3"] - props["bbox-1"]
        biggest_prop = props.iloc[props['width'].argmax()]
        oxide_col_start = biggest_prop["bbox-1"]
        oxide_col_end = biggest_prop["bbox-3"]
        oxide_layer = oxide_layer[:, oxide_col_start:oxide_col_end]
        rows, cols = oxide_layer.shape


        # get the raw profiles
        top_fill = flood_fill(oxide_layer, (0, 0), 1)
        bot_fill = flood_fill(oxide_layer, (rows - 1, cols - 1), 1)
        top_y = np.argmax(bot_fill, axis=0)
        top_raw_xy = np.array([[y, i] for i, y in enumerate(top_y)])
        top_raw_line = self.coords_to_array(top_raw_xy, rows, cols)
        bot_y = np.argmax(top_fill[::-1, :], axis=0)  # reversed to get bottom values first
        bot_raw_xy = np.array([[rows - y - 1, i] for i, y in enumerate(bot_y)])
        bot_raw_line = self.coords_to_array(bot_raw_xy, rows, cols)
        top_reference = np.average(top_raw_xy[:, 0])
        bot_reference = np.average(bot_raw_xy[:, 0])

        # fit linear line to raw profile
        m, b = np.polyfit(top_raw_xy[:, 1], top_raw_xy[:, 0], 1)
        top_least_squares_xy = [[m * x + b, x] for x in range(cols)]
        top_least_squares_line = self.coords_to_array(top_least_squares_xy, rows, cols)
        m, b = np.polyfit(bot_raw_xy[:, 1], bot_raw_xy[:, 0], 1)
        bot_least_squares_xy = [[m * x + b, x] for x in range(cols)]
        bot_least_squares_line = self.coords_to_array(bot_least_squares_xy, rows, cols)

        # get primary profile by subtracting the least squares line from the raw profile
        top_primary_xy = np.array([[raw[0] - ls[0], raw[1]] for raw, ls in zip(top_raw_xy, top_least_squares_xy)])
        top_primary_line = self.coords_to_array([[c[0] + top_reference, c[1]] for c in top_primary_xy], rows, cols)
        bot_primary_xy = np.array([[raw[0] - ls[0], raw[1]] for raw, ls in zip(bot_raw_xy, bot_least_squares_xy)])
        bot_primary_line = self.coords_to_array([[c[0] + bot_reference, c[1]] for c in bot_primary_xy], rows, cols)

        # use lowpass filter to get waviness profile
        top_mean_xy = self.low_pass(top_raw_xy, cutoff_wl=cutoff_wl)  # start w/ raw for plotting purposes
        top_mean_line = self.coords_to_array(top_mean_xy, rows, cols)
        bot_mean_xy = self.low_pass(bot_raw_xy, cutoff_wl=cutoff_wl)
        bot_mean_line = self.coords_to_array(bot_mean_xy, rows, cols)
        top_mean_xy = self.low_pass(top_primary_xy, cutoff_wl=cutoff_wl)  # need to recalculate with the primary instead of raw for actual measurements
        bot_mean_xy = self.low_pass(bot_primary_xy, cutoff_wl=cutoff_wl)

        # find roughness profile
        top_roughness_xy = [[primary[0] - mean[0], primary[1]] for primary, mean in
                            zip(top_primary_xy[cutoff_wl:-cutoff_wl], top_mean_xy)]
        top_roughness_line = self.coords_to_array([[c[0] + top_reference, c[1]] for c in top_roughness_xy], rows, cols)
        bot_roughness_xy = [[primary[0] - mean[0], primary[1]] for primary, mean in
                            zip(bot_primary_xy[cutoff_wl:-cutoff_wl], bot_mean_xy)]
        bot_roughness_line = self.coords_to_array([[c[0] + bot_reference, c[1]] for c in bot_roughness_xy], rows, cols)

        # total roughness profile (just the primary profile within the cutoff range
        top_total_rough_xy = top_primary_xy[cutoff_wl:-cutoff_wl]
        bot_total_rough_xy = bot_primary_xy[cutoff_wl:-cutoff_wl]

        # calculate roughness values.

        # micro roughness
        micro_top_Ra = np.sum([abs(z*self.pixel_size) for z, _ in top_roughness_xy]) / len(top_roughness_xy)
        micro_top_Rq = np.sqrt(np.sum([(z*self.pixel_size) ** 2 for z, _ in top_roughness_xy]) / len(top_roughness_xy))
        micro_top_Rp = abs(np.max(np.array(top_roughness_xy)[:, 0])) * self.pixel_size  # highest peak
        micro_top_Rv = abs(np.min(np.array(top_roughness_xy)[:, 0])) * self.pixel_size  # lowest valley
        micro_top_Rt = micro_top_Rp + micro_top_Rv  # total height of roughness
        micro_top_RSk = (1 / micro_top_Rq ** 3) * np.sum([(z*self.pixel_size) ** 3 for z, _ in top_roughness_xy]) / len(top_roughness_xy)

        micro_bot_Ra = np.sum([abs(z * self.pixel_size) for z, _ in bot_roughness_xy]) / len(bot_roughness_xy)
        micro_bot_Rq = np.sqrt(np.sum([(z * self.pixel_size) ** 2 for z, _ in bot_roughness_xy]) / len(bot_roughness_xy))
        micro_bot_Rp = abs(np.max(np.array(bot_roughness_xy)[:, 0])) * self.pixel_size  # highest peak
        micro_bot_Rv = abs(np.min(np.array(bot_roughness_xy)[:, 0])) * self.pixel_size  # lowest valley
        micro_bot_Rt = micro_bot_Rp + micro_bot_Rv  # total height of roughness
        micro_bot_RSk = (1 / micro_bot_Rq ** 3) * np.sum([(z * self.pixel_size) ** 3 for z, _ in bot_roughness_xy]) / len(bot_roughness_xy)

        self.gui.write_results("\nMicro roughness")
        self.gui.write_results("top Ra: %.2f" % micro_top_Ra)
        self.gui.write_results("top Rq: %.2f" % micro_top_Rq)
        self.gui.write_results("top Rp: %.2f" % micro_top_Rp)
        self.gui.write_results("top Rv: %.2f" % micro_top_Rv)
        self.gui.write_results("top Rt: %.2f" % micro_top_Rt)
        self.gui.write_results("top RSk: %.2f" % micro_top_RSk)
        self.gui.write_results("bot Ra: %.2f" % micro_bot_Ra)
        self.gui.write_results("bot Rq: %.2f" % micro_bot_Rq)
        self.gui.write_results("bot Rp: %.2f" % micro_bot_Rp)
        self.gui.write_results("bot Rv: %.2f" % micro_bot_Rv)
        self.gui.write_results("bot Rt: %.2f" % micro_bot_Rt)
        self.gui.write_results("bot RSk: %.2f" % micro_bot_RSk)

        # macro roughness
        macro_top_Ra = np.sum([abs(z*self.pixel_size) for z, _ in top_mean_xy]) / len(top_mean_xy)
        macro_top_Rq = np.sqrt(np.sum([(z*self.pixel_size) ** 2 for z, _ in top_mean_xy]) / len(top_mean_xy))
        macro_top_Rp = abs(np.max(np.array(top_mean_xy)[:, 0])) * self.pixel_size  # highest peak
        macro_top_Rv = abs(np.min(np.array(top_mean_xy)[:, 0])) * self.pixel_size  # lowest valley
        macro_top_Rt = macro_top_Rp + macro_top_Rv  # total height of roughness
        macro_top_RSk = (1 / macro_top_Rq ** 3) * np.sum([(z*self.pixel_size) ** 3 for z, _ in top_mean_xy]) / len(top_mean_xy)

        macro_bot_Ra = np.sum([abs(z * self.pixel_size) for z, _ in bot_mean_xy]) / len(bot_mean_xy)
        macro_bot_Rq = np.sqrt(np.sum([(z * self.pixel_size) ** 2 for z, _ in bot_mean_xy]) / len(bot_mean_xy))
        macro_bot_Rp = abs(np.max(np.array(bot_mean_xy)[:, 0])) * self.pixel_size  # highest peak
        macro_bot_Rv = abs(np.min(np.array(bot_mean_xy)[:, 0])) * self.pixel_size  # lowest valley
        macro_bot_Rt = macro_bot_Rp + macro_bot_Rv  # total height of roughness
        macro_bot_RSk = (1 / macro_bot_Rq ** 3) * np.sum([(z * self.pixel_size) ** 3 for z, _ in bot_mean_xy]) / len(bot_mean_xy)

        self.gui.write_results("\nMacro roughness")
        self.gui.write_results("top Ra: %.2f" % macro_top_Ra)
        self.gui.write_results("top Rq: %.2f" % macro_top_Rq)
        self.gui.write_results("top Rp: %.2f" % macro_top_Rp)
        self.gui.write_results("top Rv: %.2f" % macro_top_Rv)
        self.gui.write_results("top Rt: %.2f" % macro_top_Rt)
        self.gui.write_results("top RSk: %.2f" % macro_top_RSk)
        self.gui.write_results("bot Ra: %.2f" % macro_bot_Ra)
        self.gui.write_results("bot Rq: %.2f" % macro_bot_Rq)
        self.gui.write_results("bot Rp: %.2f" % macro_bot_Rp)
        self.gui.write_results("bot Rv: %.2f" % macro_bot_Rv)
        self.gui.write_results("bot Rt: %.2f" % macro_bot_Rt)
        self.gui.write_results("bot RSk: %.2f" % macro_bot_RSk)

        # total roughness
        total_top_Ra = np.sum([abs(z*self.pixel_size) for z, _ in top_total_rough_xy]) / len(top_total_rough_xy)
        total_top_Rq = np.sqrt(np.sum([(z*self.pixel_size) ** 2 for z, _ in top_total_rough_xy]) / len(top_total_rough_xy))
        total_top_Rp = abs(np.max(np.array(top_total_rough_xy)[:, 0])) * self.pixel_size  # highest peak
        total_top_Rv = abs(np.min(np.array(top_total_rough_xy)[:, 0])) * self.pixel_size  # lowest valley
        total_top_Rt = total_top_Rp + total_top_Rv  # total height of roughness
        total_top_RSk = (1 / total_top_Rq ** 3) * np.sum([(z*self.pixel_size) ** 3 for z, _ in top_total_rough_xy]) / len(top_total_rough_xy)

        total_bot_Ra = np.sum([abs(z * self.pixel_size) for z, _ in bot_total_rough_xy]) / len(bot_total_rough_xy)
        total_bot_Rq = np.sqrt(np.sum([(z * self.pixel_size) ** 2 for z, _ in bot_total_rough_xy]) / len(bot_total_rough_xy))
        total_bot_Rp = abs(np.max(np.array(bot_total_rough_xy)[:, 0])) * self.pixel_size  # highest peak
        total_bot_Rv = abs(np.min(np.array(bot_total_rough_xy)[:, 0])) * self.pixel_size  # lowest valley
        total_bot_Rt = total_bot_Rp + total_bot_Rv  # total height of roughness
        total_bot_RSk = (1 / total_bot_Rq ** 3) * np.sum([(z * self.pixel_size) ** 3 for z, _ in bot_total_rough_xy]) / len(bot_total_rough_xy)

        self.gui.write_status('Done! Took', round(time.time() - t0, 2), 'seconds.')
        self.gui.write_results("\nTotal roughness")
        self.gui.write_results("top Ra: %.2f" % total_top_Ra)
        self.gui.write_results("top Rq: %.2f" % total_top_Rq)
        self.gui.write_results("top Rp: %.2f" % total_top_Rp)
        self.gui.write_results("top Rv: %.2f" % total_top_Rv)
        self.gui.write_results("top Rt: %.2f" % total_top_Rt)
        self.gui.write_results("top RSk: %.2f" % total_top_RSk)
        self.gui.write_results("bot Ra: %.2f" % total_bot_Ra)
        self.gui.write_results("bot Rq: %.2f" % total_bot_Rq)
        self.gui.write_results("bot Rp: %.2f" % total_bot_Rp)
        self.gui.write_results("bot Rv: %.2f" % total_bot_Rv)
        self.gui.write_results("bot Rt: %.2f" % total_bot_Rt)
        self.gui.write_results("bot RSk: %.2f" % total_bot_RSk)

        # #stack masks
        visualize_mask = np.zeros((self.rows, self.cols, 8))
        visualize_mask[:, oxide_col_start:oxide_col_end, 0] = top_raw_line
        visualize_mask[:, oxide_col_start:oxide_col_end, 1] = top_least_squares_line
        visualize_mask[:, oxide_col_start:oxide_col_end, 2] = top_mean_line
        visualize_mask[:, oxide_col_start:oxide_col_end, 3] = bot_raw_line
        visualize_mask[:, oxide_col_start:oxide_col_end, 4] = bot_least_squares_line
        visualize_mask[:, oxide_col_start:oxide_col_end, 5] = bot_mean_line
        visualize_mask[:, oxide_col_start:oxide_col_end, 6] = top_roughness_line
        visualize_mask[:, oxide_col_start:oxide_col_end, 7] = bot_roughness_line

        selem = np.zeros((3, 3, 3))
        selem[:, :, 1] = 1
        dilate_visualize_mask = binary_dilation(visualize_mask, selem=selem)

        vis = visualize.overlay_mask(self.selected_area, dilate_visualize_mask,
                                     colors=['r', 'b', 'g', 'r', 'b', 'g', 'w', 'w'],
                                     return_overlay=True)

        if self.pdf is not None:
            inout.pdfSaveImage(self.pdf, vis, title='Roughness profiles')

        if self.show_steps:
            visualize.overlay_mask(self.selected_area, dilate_visualize_mask, colors=['r','b','g','r','b','g', 'w', 'w'])

        return micro_top_Ra, micro_top_Rq, micro_top_Rp, micro_top_Rv, micro_top_Rt, micro_top_RSk, \
               micro_bot_Ra, micro_bot_Rq, micro_bot_Rp, micro_bot_Rv, micro_bot_Rt, micro_bot_RSk, \
               macro_top_Ra, macro_top_Rq, macro_top_Rp, macro_top_Rv, macro_top_Rt, macro_top_RSk, \
               macro_bot_Ra, macro_bot_Rq, macro_bot_Rp, macro_bot_Rv, macro_bot_Rt, macro_bot_RSk, \
               total_top_Ra, total_top_Rq, total_top_Rp, total_top_Rv, total_top_Rt, total_top_RSk, \
               total_bot_Ra, total_bot_Rq, total_bot_Rp, total_bot_Rv, total_bot_Rt, total_bot_RSk

    def low_pass(self, profile, cutoff_wl=100):
        ALPHA = 0.4697  # from Le Roux (2015)
        filtered = []
        weight = [1 / (ALPHA * cutoff_wl) * np.exp(-np.pi * (abs(x - cutoff_wl) / (ALPHA * cutoff_wl)) ** 2) for x in
                  range(cutoff_wl * 2)]

        for i in range(cutoff_wl, len(profile) - cutoff_wl):
            x = profile[i, 1]
            y = np.sum(np.multiply(weight, profile[i - cutoff_wl:i + cutoff_wl, 0]))
            filtered.append([y, x])

        return np.array(filtered)

    def coords_to_array(self, coords, rows=None, cols=None):
        rows = self.rows if rows is None else rows
        cols = self.cols if cols is None else cols
        a = np.zeros((rows, cols))
        for c in coords:
            a[int(round(c[0])), int(round(c[1]))] = 1
        return a



class Feature():
    def __init__(self, title=None, values={}):
        self.title = title
        self.values = values

if __name__ == "__main__":
    # test inconel UNET segmentation
    if False:
        model = [None]
        checkpoint = [None]
        segment.load_model('inconel_unet_best_model.pth', model, checkpoint)
        model = model[0]
        checkpoint = checkpoint[0]
        im = inout.load('inc_test.tif')
        mask = segment.segment_with_unet(im, model, checkpoint, patch_size=512, batch_size=20)
        visualize.overlay_mask(im, get_background(mask))

    # test measured micrograph super class
    if False:
        m = Measured_Precipitate_Micrograph(parent_gui='parent', micrograph_path='path')

    #mask = inout.load_mask('test_mask.png')
