#!/usr/bin/env python3
"""
This module contains GUI elements and the main application for the aquami_sa
package.
"""

# Stuff for pyinstaller
#https://github.com/pytorch/vision/issues/1899
def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #this is bad solution https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-when-fitting-models
import sys
from threading import Thread
import queue
import string

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

from itertools import zip_longest
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from skimage.segmentation import clear_border
from skimage.color import rgb2gray, gray2rgb, label2rgb
from skimage.restoration import denoise_bilateral
from skimage.morphology import (skeletonize, remove_small_objects, disk,
                                binary_dilation, binary_opening, remove_small_holes)
from skimage.measure import regionprops
#from scipy.ndimage.morphology import distance_transform_edt
import time
import pandas as pd

import copy

import tkSimpleDialog
import visualize
import measure
import segment
import inout

import warnings
warnings.filterwarnings("ignore")

__author__ = "Joshua Stuckner"
__version__ = '1.0'

import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 300

class selectRect(object):
    """
    Draws a rectange on a figure and keeps track of the rectangle's size and
    location.  Used to select the image data and scale bar.

    Attributes
    ----------
    x0 : float64
        X coordinate (row) of start of rectangle.
    y0 : float 64
        Y coordinate (column) of start of rectangle.
    x1 : float64
        X coordinate (row) of end of rectangle.
    y1 : float 64
        Y coordinate (column) of end of rectangle.
    """
    
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0,0), 1, 1,
                              facecolor='none',
                              edgecolor='#6CFF33',
                              linewidth=3)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        if (self.x1 is not None and self.x0 is not None and
                            self.y1 is not None and self.y0 is not None):
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
            self.ax.figure.canvas.draw()

class inputScale(tkSimpleDialog.Dialog):        
    def body(self, master):

        self.varUnit = tk.StringVar()
        options = ["nm", "μm", "mm"]
        self.varUnit.set(options[1])

        self.directions = tk.Label(master,
                text="Please manually input the scale bar number.")

        self.textInput = tk.Text(master, width=8, height=1)

        self.optUnit = tk.OptionMenu(master, self.varUnit, *options)
        self.optUnit.config(width=6)

        f = Figure()
        a = f.add_subplot(111)
        a.imshow(self.scale, cmap=plt.cm.gray, interpolation=None)
        canvas = FigureCanvasTkAgg(f, master=master)
        try:
            canvas.draw()
        except:
            canvas.show()

        
        canvas.get_tk_widget().grid(row=1, column=0, columnspan=2)
        canvas._tkcanvas.grid(row=1, column=0, columnspan=2)

        self.directions.grid(row=0, column=0, columnspan=2)

        self.textInput.grid(row=2, column=0)
        self.optUnit.grid(row=2, column=1)
 

        return self.textInput #initial focus

    def apply(self):
        text = self.textInput.get("1.0",tk.END).strip()
        self.result = float(text)


class manualThreshhold(object):
    def __init__(self, img, threshinit):
        self.img = img
        self.original = img.copy()
        self.img = denoise_bilateral(self.img,
                            sigma_color=0.05,  #increase this to blur more
                            sigma_spatial=1,
                            multichannel=False)
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.01, bottom=0.20)
        self.manThresh = 0
        self.pltimage = self.ax.imshow(self.img, interpolation='nearest')
        self.ax_thresh = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.s_thresh = Slider(self.ax_thresh, 'Threshold', 0, 255,
                               valinit=threshinit)
        self.s_thresh.on_changed(self.threshUpdate)
        self.threshUpdate(val=1)
        figManager = plt.get_current_fig_manager()
        try:
            figManager.window.showMaximized()
        except AttributeError: # TkAgg backend
            figManager.window.state('zoomed')
        plt.axis('off')       

    def threshUpdate(self, val):      
        val = self.s_thresh.val
        self.manThreshVal = self.s_thresh.val
        img = inout.uint8(self.img)
        self.mask = img >= int(val)
        overlay = visualize.overlay_mask(self.original, self.mask, colors=['o'],
                                      return_overlay=True)
        self.pltimage.set_data(overlay)        
        self.fig.canvas.draw_idle()

    def show(self):
        #plt.ion() Removed 3/9/23 to stop code from hanging when closing figure.
        plt.show()
        while plt.get_fignums():
            try:
                plt.pause(0.1)
            except:
                pass
        plt.ioff()

    def getMask(self):
        return self.mask

        
def selectFile(title="Select image", initialdir=None, multiple=False):
    """
    Launches a dialog to select a file.

    Parameters
    ----------
    initialdir : str, optional
        The start path of the select a file dialog.

    multiple : bool, optional
        If true, allows the selecting of multiple files.
        
    Returns
    -------
    file : str
        The directory and name of the selected file.
    """
    file = filedialog.askopenfilename(
            initialdir=initialdir,
            multiple=multiple,
            title=title
            )
    return file


def selectFolder():
    """
    Launches a dialog to select a folder.

    Returns
    -------
    directory : str
        The directory of the selected folder.
    """   
    directory = filedialog.askdirectory(
            title='Select file'
            )
    return directory


def saveFile(ftype=None, title='Save file'): 
    file = filedialog.asksaveasfile(
            defaultextension=ftype,
            title=title
            )
    
    path = file.name
    file.close()
    return path


def manualPixelSize(scale):
    '''
    Returns the size of each pixel in nm.  Must manually select the scale bar
    and input the size.

    Parameters
    ----------
    scale : ndarray
        Image containing the scale bar

    Returns
    -------
    pixelSize : float64
        Size in nm of each pixel
    '''

    plt.imshow(scale, cmap=plt.cm.gray, interpolation=None)
    plt.title("Draw a rectangle the same width as the scale bar and close the"+\
            " figure.")
    figManager = plt.get_current_fig_manager()
    try:
        figManager.window.showMaximized()
    except AttributeError: # TkAgg backend
        figManager.window.state('zoomed')
    a = selectRect()
    plt.axis('off')
    #plt.ion() Removed 3/9/23 to stop code from hanging when closing figure.
    plt.show()
    while plt.get_fignums():
        try:
            plt.pause(0.1)
        except:
            pass
    plt.ioff()
    barSize = abs(a.x1-a.x0)
    scaleNum = 0
    root = tk.Toplevel()
    try:
        root.iconbitmap('aqua.ico')
    except tk.TclError:
        root.iconbitmap('aquami\\aqua.ico')
    pixapp = inputScale(root, scale=scale)
    scaleNum = pixapp.result
    root.destroy()
    try:
        pixelSize = scaleNum/barSize
    except TypeError:
        pixelSize = 0
    return pixelSize, pixapp.varUnit.get()


def manualSelectImage(img, set_scale=True, return_crop_area=False):
    '''
    Used to select the image data and the scale bar in a raw microscopy image
    that may contain other metadata besides the actual image.
    
    Parameters
    ----------
    img : ndarray
        Raw microscopy image.
    bool : set_scale
        Whether the user will measure the scale bar.

    Returns
    -------
    im_data : ndarray
        Selected area containing just the image data

    scale : ndarray
        Selected area containing an image of the scale bar.
    '''

    # Get shape of image and convert to gray scale.
    try:
        rows, cols =  img.shape
    except ValueError: # Convert to gray scale
        img = rgb2gray(img)
        rows, cols =  img.shape
    
    # Select the image.
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.title('Select the image data and close this figure.')
    figManager = plt.get_current_fig_manager()
    try:
        figManager.window.showMaximized()
    except AttributeError: # TkAgg backend
        figManager.window.state('zoomed')
        
    a = selectRect()
    plt.axis('off')
    #plt.ion() Removed 3/9/23 to stop code from hanging when closing figure.
    plt.show()
    while plt.get_fignums():
        try:
            plt.pause(0.1)
        except:
            pass
    plt.ioff() 
    # Make sure the selection isn't out of bounds or reversed.
    y0 = min(a.y0, a.y1)
    y1 = max(a.y0, a.y1)
    x0 = min(a.x0, a.x1)
    x1 = max(a.x0, a.x1)
    y0 = 0 if y0 < 0 else y0
    y1 = rows if y1 > rows else y1
    x0 = 0 if x0 < 0 else x0
    x1 = cols if x1 > cols else x1
    # Crop the raw image to the image data.
    im_data = img[int(y0):int(y1), int(x0):int(x1)]
    crop_area = [int(a) for a in [y0, y1, x0, x1]]

    scale = 1
    if set_scale:
        # Select the scale bar.
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.title('Draw a box around the scale bar and close this figure.' \
                  ' No need to be precise as long as the whole bar is included.')
        figManager = plt.get_current_fig_manager()
        try:
            figManager.window.showMaximized()
        except AttributeError: # TkAgg backend
            figManager.window.state('zoomed')

        a = selectRect()
        plt.axis('off')
        #plt.ion() Removed 3/9/23 to stop code from hanging when closing figure.
        plt.show()
        while plt.get_fignums():
            try:
                plt.pause(0.1)
            except:
                pass
        plt.ioff()
        # Make sure the selection isn't out of bounds or reversed.
        y0 = min(a.y0, a.y1)
        y1 = max(a.y0, a.y1)
        x0 = min(a.x0, a.x1)
        x1 = max(a.x0, a.x1)
        y0 = 0 if y0 < 0 else y0
        y1 = rows if y1 > rows else y1
        x0 = 0 if x0 < 0 else x0
        x1 = cols if x1 > cols else x1
        # Crop the raw image to the scale bar.
        scale = img[int(y0):int(y1), int(x0):int(x1)]

    if return_crop_area:
        return crop_area, scale

    return im_data, scale


def user_input_good(input_string, allowed, boxName=''):
    """
    This function checks the input text and displays error messages if the input
    cannot be converted to the correct datatype.  Returns TRUE if the input is good.
    Returns FALSE if the input is not good.

    Params
    ======
    input_string: The string input by the end user.
    allowed (string): the datatype that is acceptable.

    Return
    ======
    bool: TRUE if the input is ok, FALSE if the input is not ok.
    """
    # First make sure the box isn't empty
    if len(input_string) == 0:
        messagebox.showwarning(
            "Input error",
            ''.join((boxName, ' value is missing.')))
        return False

    integer = ['integer', 'Int', 'int', 'Integer', '0']
    decimal = ['decimal', 'Dec', 'Decimal', 'dec', '2', 'float']
    signed_integer = ['sinteger', 'sInt', 'sint', 'sInteger', '1']
    even_signed_integer = ['esint']

    if str(allowed) in integer:
        bad = ''.join((string.ascii_letters, string.punctuation,
                       string.whitespace))
        if any((c in bad) for c in input_string):
            messagebox.showwarning(
                "Input error",
                ''.join((boxName, ' should be a positive integer value')))
            return False
        else:
            return True

    if str(allowed) in signed_integer:
        bad = ''.join((string.ascii_letters, string.punctuation,
                       string.whitespace))
        bad = ''.join(c for c in bad if c not in '-')
        if any((c in bad) for c in input_string) or '-' in input_string[1:]:
            messagebox.showwarning(
                "Input error",
                ''.join((boxName, ' should be an integer value')))
            return False
        else:
            return True

    if str(allowed) in decimal:
        bad = ''.join((string.ascii_letters, string.punctuation,
                       string.whitespace))
        bad = ''.join(c for c in bad if c not in '.')
        if any((c in bad) for c in input_string) or input_string.count('.') > 1:
            messagebox.showwarning(
                "Input error",
                ''.join((boxName, ' should be an posative decimal value')))
            return False
        else:
            return True

    if str(allowed) in even_signed_integer:
        bad = ''.join((string.ascii_letters, string.punctuation,
                       string.whitespace))
        bad = ''.join(c for c in bad if c not in '-')
        if any((c in bad) for c in input_string) or '-' in input_string[1:]:
            messagebox.showwarning(
                "Input error",
                ''.join((boxName, ' should be an integer value')))
            return False
        elif int(input_string) % 2 == 0 and int(input_string) >= 0:
            messagebox.showwarning(
                "Input error",
                ''.join(('Stacking should be an odd integer')))
            return False
        else:
            return True


class CreateToolTip(object):
    '''
    create a tooltip for a given widget
    '''
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.close)
        self.timer = []
        self.tw = None
        
    def enter(self, event=None):
        self.timer = self.widget.after(700, self.display)
        
    def display(self):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background='#ffffe6', relief='solid', borderwidth=1,
                       font=("times", "10", "normal"))
        label.pack(ipadx=5, ipady=3)
        
    def close(self, event=None):
        if self.tw is not None:
            self.tw.destroy()
        self.widget.after_cancel(self.timer)

class Aquami_Gui(tk.Frame):
    """
    This class is the main GUI for aquami_sa
    """
    
    def __init__(self, master=None):
        """
        Called when class is created.
        """
        # start loading the model
        self.model = [None]
        self.checkpoint = [None]
        self.load_model()
        # self.thread_model_load = Thread(target=segment.load_model,
        #                                           kwargs={'returned_model' : self.model,
        #                                                   'returned_checkpoint': self.checkpoint})
        # self.thread_model_load.start()

        tk.Frame.__init__(self, master)
        self.master.title('Aquami-TGO (v' + __version__ + ')')
        module_path, this_filename = os.path.split(__file__)
        try:
            self.master.iconbitmap(''.join((module_path, 'aqua.ico')))
        except:
            try:
                self.master.iconbitmap(''.join((module_path, '/aqua.ico')))
            except:
                pass

        self.initGUI()

    def initGUI(self):
        """
        Create and layout all the widgets.
        """

        self.pack(fill=tk.BOTH, expand=True)

        # Figure out sizing.
        width = 200
        height = 200
        pad = 5
        fontWidth = 8
        bigWidth = int((width*3 + pad*6) / fontWidth)
        
        # Create option frames.
        self.frameOutputOptions = tk.LabelFrame(self, text="Output Options:",
                                                width=width, height=height)
        self.framePixelSize = tk.LabelFrame(self, text="Pixel Size:",
                                          width=width, height=height)
        self.frameOptions = tk.LabelFrame(self, text="Options:",
                                          width=width, height=height)

        # Create text boxes and labels.
        self.labelStatus = tk.LabelFrame(self, text="Status:", bd=0)
        self.labelResults = tk.LabelFrame(self, text="Results:", bd=0)
        self.textStatus = ScrolledText(self.labelStatus, height=10,
                                       width=bigWidth)
        self.textResults = ScrolledText(self.labelResults, height=20,
                                        width=bigWidth)

        # Create buttons.
        self.buttonCalculate = tk.Button(self, text='Calculate',
                                         width=20, height=1, font=12, bd=3,
                                         command=lambda:self.prepare())
        # self.buttonSaveAll = tk.Button(self, text='Save Session Summary',
        #                                command=self.saveAll)

        # Arrange toplevel widgets.
        self.frameOutputOptions.grid(row=0, column=2, padx=pad, pady=pad,
                                     sticky='NESW')
        self.framePixelSize.grid(row=0, column=1, padx=pad, pady=pad,
                               sticky='NESW')
        self.frameOptions.grid(row=0, column=0, padx=pad, pady=pad,
                               sticky='NESW')

        self.buttonCalculate.grid(row=1, column=1, 
                                  padx=pad, pady=pad*3)

        self.labelStatus.grid(row=2, column=0, columnspan=3, sticky='w',
                              padx=pad, pady=pad)
        self.textStatus.grid(row=3, column=0, columnspan=3)
        self.labelResults.grid(row=4, column=0, columnspan=3, sticky='w',
                               padx=pad, pady=pad)
        self.textResults.grid(row=5, column=0, columnspan=3)
        # self.buttonSaveAll.grid(row=6, column=1, padx=pad, pady=pad)

        # Variables
        self.Features = []
        self.outFold = None

        # Options.
        # Variables.
        self.varCutoffWaveLength = tk.StringVar(value='10')
        self.varRemoveObjectSize = tk.StringVar(value='2500')

        # Create widgets.
        self.buttonSelectOutFold = tk.Button(self.frameOptions, text='Set Output Folder',
                                             command=self.setOutputFolder)
        self.textCutoffWaveLength = tk.Entry(self.frameOptions, textvariable=self.varCutoffWaveLength, width=5)
        self.labelCutoffWaveLength = tk.Label(self.frameOptions, text='Cutoff Length (unit):')
        self.textRemoveObjectSize = tk.Entry(self.frameOptions, textvariable=self.varRemoveObjectSize, width=5)
        self.labelRemoveObjectSize = tk.Label(self.frameOptions, text='Remove oxide smaller than:')

        # Pack widgets.

        self.buttonSelectOutFold.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=10)
        self.labelCutoffWaveLength.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.textCutoffWaveLength.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        self.labelRemoveObjectSize.grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.textRemoveObjectSize.grid(row=3, column=1, sticky='w', padx=5, pady=5)

        # Pixel size options.
        # Variables.
        self.varAssumeSquare = tk.BooleanVar()
        self.varPixelSelect = tk.StringVar(value='measure')
        self.varPixelAllSame = tk.BooleanVar()
        self.varHalfSize = tk.BooleanVar()
        self.varPixelSize = tk.StringVar()
        self.varUseFullImage = tk.BooleanVar()

        # Create widgets.
        self.radMeasurePixelSize = tk.Radiobutton(self.framePixelSize,
                text="Measure manually", variable=self.varPixelSelect, value='measure',
                command=self.updateOptions)
        self.checkAllSamePixelSize = tk.Checkbutton(self.framePixelSize,
                                                    text = 'All images same pixel size and shape',
                                                    variable=self.varPixelAllSame)
        self.checkHalfSize = tk.Checkbutton(self.framePixelSize,
                                                    text = 'Resize the image to 1/2 resolution',
                                                    variable=self.varHalfSize)                                            
        self.radInputPixelSize = tk.Radiobutton(self.framePixelSize,
                                                  text="Input pixel size (μm)",
                                                  variable=self.varPixelSelect,
                                                  value='input',
                                                  command=self.updateOptions)
        self.textPixelSize = tk.Entry(self.framePixelSize, textvariable=self.varPixelSize)
        self.checkAssumeSquare = tk.Checkbutton(self.framePixelSize,
                                                text="Assume square image", variable=self.varAssumeSquare,
                                                command=self.updateOptions)
        self.checkUseFullImage = tk.Checkbutton(self.framePixelSize,
                                                text="Use full image", variable=self.varUseFullImage,
                                                command=self.updateOptions)


        # Pack widgets.
        self.checkAllSamePixelSize.grid(row=0, column=0, sticky='E', pady=(5,5))
        self.checkHalfSize.grid(row=1, column=0, sticky='W', pady=(5,10))
        self.radMeasurePixelSize.grid(row=2, column=0, sticky='W')
        self.radInputPixelSize.grid(row=3, column=0, sticky='W')
        self.textPixelSize.grid(row=4, column=0, sticky='W', padx=(25,5))
        self.checkAssumeSquare.grid(row=5, column=0, sticky='w', padx=(20,5))
        self.checkUseFullImage.grid(row=6, column=0, sticky='w', padx=(20,5))

        # Check appropriate boxes.
        self.radMeasurePixelSize.select()
        self.textPixelSize.config(state=tk.DISABLED)

        # Option options.
        # Profiles
        #profiles = autoSelect.profiles()
        # Variables.
        self.varShowSteps = tk.BooleanVar()
        self.varOutputExcel = tk.BooleanVar()
        self.varSavePDF = tk.BooleanVar()
        self.varSaveMovie = tk.BooleanVar()
        self.varSaveBinary = tk.BooleanVar()

        #self.varSetScale = tk.BooleanVar()
        #self.varProfile.set(profiles[0])
        # Create widgets.
        self.checkShowSteps = tk.Checkbutton(self.frameOutputOptions,
                                             text="Show steps", variable=self.varShowSteps)
        self.checkOutputExcel = tk.Checkbutton(self.frameOutputOptions,
                                               text="Output to Excel", variable=self.varOutputExcel)
        self.checkSavePDF = tk.Checkbutton(self.frameOutputOptions,
                                           text="Save PDF", variable=self.varSavePDF)
        self.checkSaveMovie = tk.Checkbutton(self.frameOutputOptions,
                                             text="Save movie", variable=self.varSaveMovie)
        self.checkSaveBinary = tk.Checkbutton(self.frameOutputOptions,
                                              text="Save binary", variable=self.varSaveBinary)

        # self.checkSetScale = tk.Checkbutton(self.frameOutputOptions,
        #                                     text="Measure scale bar", variable=self.varSetScale)

        #self.optionProfile = tk.OptionMenu(self.frameOptions, self.varProfile,
        #        *profiles)
        #self.optionProfile.config(state=tk.DISABLED)

        # Pack widgets.
        self.checkShowSteps.grid(row=0, column=0, sticky='w')
        self.checkOutputExcel.grid(row=1, column=0, sticky='w')
        self.checkSavePDF.grid(row=2, column=0, sticky='w')
        #self.checkSaveBinary.grid(row=4, column=0, sticky='w')

        #self.checkSetScale.grid(row=6, column=0, sticky='w')
        
        # Check appropriate boxes.
        self.checkOutputExcel.select()
        #self.checkUseFullImage.select()
        self.checkSavePDF.select()
        self.updateOptions()
        self.createToolTips()
    def load_model(self, path='models/Unet__inceptionresnetv2__acc__0.980__20230301.pth', encoder='inceptionresnetv2'):
    
        # if a previous load thread is running, wait for it to finish
        try:
            self.thread_model_load.join()
        except AttributeError:
            pass

        # pre allocate model and preproceccing_fn
        self.model = [None]
        self.preprocessing_fn = [None]

        # start loading the model in a seperate thread
        self.thread_model_load = Thread(target=segment.load_pytorch_model,
                                                  kwargs={'path' : path,
                                                          'encoder' : encoder,
                                                          'returned_model': self.model, #pre allocated model and preprocessing
                                                          'returned_preprocessing_fn': self.preprocessing_fn})
        self.thread_model_load.start()


    def createToolTips(self):
        pass
        self.ttps = []
        this_dir, this_filename = os.path.split(__file__)
        widgets = [
            [self.checkShowSteps, 'opt_show_steps.txt'],
            [self.checkOutputExcel, 'opt_output_excel.txt'],
            [self.checkSavePDF, 'opt_save_pdf.txt'],
            [self.checkSaveMovie, 'opt_save_movie.txt'],
            [self.checkSaveBinary, 'opt_save_binary.txt'],
            [self.buttonSelectOutFold, 'opt_output_folder.txt'],
            [self.buttonCalculate, 'calculate.txt'],
            [self.checkAllSamePixelSize, 'ps_all_same_size.txt'],
            [self.radMeasurePixelSize, 'ps_measure_manually.txt'],
            [self.radInputPixelSize, 'ps_input_pixel_size.txt'],
            [self.checkAssumeSquare, 'ps_assume_square.txt'],
            [self.checkUseFullImage, 'ps_use_full_image.txt'],
            [self.textCutoffWaveLength, 'opt_cutoff_length.txt'],
            [self.labelCutoffWaveLength, 'opt_cutoff_length.txt'],
            [self.textRemoveObjectSize, 'opt_remove_small.txt'],
            [self.labelRemoveObjectSize, 'opt_remove_small.txt'],
            [self.checkHalfSize, 'opt_half_size.txt']
        ]

        for widget, txt in widgets:
            text_path = os.path.join(this_dir, 'ttps', txt)
            f = open(text_path)
            data = f.read()
            f.close()
            self.ttps.append(CreateToolTip(widget, data))

    def setOutputFolder(self):
        self.outFold = selectFolder()


    def updateOptions(self):
        """
        Updates the GUI options.  Deactivates some options in response to the
        selection of other options.
        """


        if self.varPixelSelect.get() == "measure":
            #self.checkAllSamePixelSize.config(state=tk.NORMAL)
            self.textPixelSize.config(state=tk.DISABLED)
            self.checkAssumeSquare.config(state=tk.DISABLED)
            self.checkUseFullImage.config(state=tk.DISABLED)
        else:
            #self.checkAllSamePixelSize.config(state=tk.DISABLED)
            self.checkAllSamePixelSize.select()
            self.textPixelSize.config(state=tk.NORMAL)
            self.checkAssumeSquare.config(state=tk.NORMAL)
            self.checkUseFullImage.config(state=tk.NORMAL)
            if self.varAssumeSquare.get():
                self.checkUseFullImage.select()
                self.checkUseFullImage.config(state=tk.DISABLED)





    def write_status(self, *strings, sep=' ', end='\n', also_print=True):
        self.write(self.textStatus, *strings, sep=sep, end=end, also_print=also_print)

    def write_results(self, *strings, sep=' ', end='\n', also_print=True):
        self.write(self.textResults, *strings, sep=sep, end=end, also_print=also_print)

    def write(self, box, *strings, sep=' ', end='\n', also_print=True):
        """
        Displays text in a text box with similar functionality as the print
        statement.

        Parameters
        ----------
        box : tk.Text or ScrolledText
            Text box that will be written to.

        *strings : str (any number of)
            Strings that will be written in the text box.
        sep : str, optional
            Text that will be inserted between *strings arguements.
        end : str, optional
            Text that will be inserted at the end.
        also_print : bool, optional
            If true will also print the results in terminal
        """

        if also_print:
            print(*strings, sep=sep, end=end)

        # Prepare the text to write.
        output = ''
        for i in range(len(strings)-1):            
            output = output+str(strings[i])+sep
        output = output+str(strings[-1])
        output = output + end

        # Write the output to the appropraite text box.
        box.insert(tk.END, output)
        box.see(tk.END)

        self.update_idletasks()
        time.sleep(0.1)

    def prepare(self):
        """
        Allows the selection of multiple files if segment is not set to binary.
        Calls the calculate function for each selected file.
        """
        self.calculate()

        
    def with_status(self, status, target, *args, **kwargs):
        t0 = time.time()
        self.write(self.textStatus, status, end='')
        ret = target(*args, **kwargs)
        self.write(self.textStatus, 'Done! Took', time.time()-t0, 'seconds.')


    def calculate(self):
        """
        Performs measurements based on the options.

        Parameters
        ----------
        fullFile : str
            The path to the image to operate on.
        """


        if user_input_good(self.varCutoffWaveLength.get(), 'float', boxName='Cutoff Wavelength'):
            cutoff_wl = float(self.varCutoffWaveLength.get())
        else:
            return 0
        if user_input_good(self.varRemoveObjectSize.get(), 'float', boxName='Remove Small Oxide'):
            remove_small = float(self.varRemoveObjectSize.get())
        else:
            return 0

        # get pixel size
        pixel_size = 1
        unit = 'μm'
        if self.varPixelSelect.get() == 'input':
            if user_input_good(self.varPixelSize.get(), 'float', boxName='Pixel Size'):
                pixel_size = self.varPixelSize.get()
            else:
                return 0


        micrograph_files = selectFile(multiple=True)

        # setup excel
        if self.varOutputExcel.get():
            if self.outFold == None:
                folder, _ = os.path.split(micrograph_files[0])
                excelPath = os.path.join(folder, 'All Measurements.xlsx')
            else:
                excelPath = os.path.join(self.outFold, 'All Measurements.xlsx')
            if os.path.isfile(excelPath):
                msg = tk.messagebox.askyesnocancel(title = 'AQUAMI', message='Do you want to overwrite the existing measurements file?')
                if msg:
                    pass
                elif msg is False:
                    try:
                        path = filedialog.asksaveasfile(filetypes=[('Excel File', '*.xlsx')])
                        excelPath = path.name
                        if '.' not in excelPath:
                            excelPath += '.xlsx'
                    except PermissionError:
                        tk.messagebox.showwarning('AQUAMI Warning', 'File is in use by another program.\nPlease close file and try again.')
                        return 0
                elif msg is None:
                    return 0

            writer = pd.ExcelWriter(excelPath)
            df = pd.DataFrame()
            df.to_excel(writer, header=False, index=False, sheet_name='Summary')
            try:
                writer.save()
            except AttributeError:
                try:
                    writer.close()
                except PermissionError:
                    tk.messagebox.showwarning('AQUAMI Warning',
                                              'File is in use by another program.\nPlease close file and try again.')
                    return 0
            except PermissionError:
                tk.messagebox.showwarning('AQUAMI Warning',
                                          'File is in use by another program.\nPlease close file and try again.')
                return 0

        # Make sure the model has loaded
        self.thread_model_load.join()
        model = self.model[0]
        preprocessing_fn = self.preprocessing_fn[0]


        all_features=[]
        crop_area = None
        for file in micrograph_files:
            if self.varPixelAllSame.get() and crop_area is None and not self.varUseFullImage.get():
                if os.path.isfile(file):
                    im0 = inout.load(file)
                    if self.varPixelSelect.get() == 'measure':
                        crop_area, scale = manualSelectImage(im0, set_scale=True, return_crop_area=True)
                        scale = inout.uint8(scale)
                        pixel_size, unit = manualPixelSize(scale)
                    else:
                        crop_area, scale = manualSelectImage(im0, set_scale=False, return_crop_area=True)
                else: #couldn't load file
                    return 0

            if True:
                m = measure.Measured_EBC_Oxide_Micrograph(
                    parent_gui=self,
                    micrograph_path=file,
                    NN_model=model,
                    NN_preprocessing_fn=preprocessing_fn,
                    segmentation_use_full_image=(self.varPixelSelect.get()=='input' and self.varUseFullImage.get() or
                                                 self.varPixelSelect.get()=='input' and self.varAssumeSquare.get()),
                    segmnetation_crop_area=crop_area,
                    segmentation_assume_square=(self.varAssumeSquare.get() and self.varPixelSelect.get()=='input'),
                    half_size=self.varHalfSize.get(),
                    save_segment=self.varSaveBinary.get(),
                    segmentation_measure_scale_bar=(self.varPixelSelect.get()=='measure'),
                    segmentation_resize=False,
                    show_steps=self.varShowSteps.get(),
                    pixel_size=float(pixel_size),
                    unit=unit,
                    output_folder=self.outFold,
                    output_excel=False,
                    save_pdf=self.varSavePDF.get(),
                    phase_names=['background', 'oxide', 'crack'],
                    cutoff_wl=cutoff_wl,
                    remove_small=remove_small
                )
            #m.calculate()
            all_features.append(m.features)



            # save summary
            if self.varOutputExcel.get():
                titles = [[]]
                for f in all_features[0]:
                    if f.title is None:
                        titles[0].extend([''])
                    else:
                        titles[0].extend([f.title, '', ''])
                data = np.array(titles)

                for i, features in enumerate(all_features):
                    if i == 0:
                        heads = [[key for f in features for key in f.values if key != 'all_measurements']]
                        data = np.append(data, heads, axis=0)
                    vals = [[f.values[key] for f in features for key in f.values if key != 'all_measurements']]
                    try:
                        data = np.append(data, vals, axis=0)
                    except ValueError:
                        print(vals)


            # Format the excel file
                df = pd.DataFrame(data)
                df.to_excel(writer, header=False, index=False, sheet_name='Summary')
                try:
                    writer.save()
                except PermissionError:
                    tk.messagebox.showwarning('AQUAMI Warning',
                                              'File is in use by another program.\nPlease close file and press "OK".\n(Existing file will be overwritten)')
                    try:
                        writer.save()
                    except PermissionError:
                        tk.messagebox.showerror('AQUAMI Error',
                                                'File is in use by another program.\nRESULTS FILE NOT SAVED!')
                        return 0

                all_data = {}
                for i, features in enumerate(all_features):
                    for f in features:
                        if f.title is not None:
                            try:
                                entry = [features[0].values['File name']]
                                entry.extend(f.values['all_measurements'])
                                if i == 0:
                                    all_data[f.title] = [entry]
                                elif i == 1:
                                    big_list = []
                                    big_list.append(all_data[f.title][0])
                                    big_list.append(entry)
                                    all_data[f.title] = big_list
                                else:
                                    big_list = []
                                    [big_list.append(l) for l in all_data[f.title]]
                                    big_list.append(entry)
                                    all_data[f.title] = big_list
                            except TypeError:
                                print(f.values['all_measurements'])
                            except KeyError:
                                pass
                for key in all_data:
                    data = all_data[key]
                    try:
                        data = list(map(list, zip_longest(*data))) #transpose the list
                        df = pd.DataFrame(data)
                        df.to_excel(writer, header=False, index=False, sheet_name=key)
                    except TypeError:
                        print('passing')
                        pass

                try:
                    writer.save()
                except PermissionError:
                    tk.messagebox.showwarning('AQUAMI Warning',
                                            'File is in use by another program.\nPlease close file and press "OK".\n(Existing file will be overwritten)')
                    try:
                        writer.save()
                    except PermissionError:
                        tk.messagebox.showerror('AQUAMI Error',
                                                'File is in use by another program.\nRESULTS FILE NOT SAVED!')

        print('done')


if __name__ == "__main__":
    root = tk.Tk()
    myapp = Aquami_Gui(root)
    myapp.mainloop()

def run():
    myapp = Aquami_Gui()
    myapp.mainloop()
