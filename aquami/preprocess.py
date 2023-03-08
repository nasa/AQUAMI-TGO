import warnings
warnings.filterwarnings("ignore")
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.color import gray2rgb
from skimage.transform import resize

import visualize
import inout

def weka_to_label(input_folder, output_folder):
    """
    Converts labels produced by weka to image-segmentation labels
    :param input_folder: (str) Should contain images and masks with imagename_mask.tif
    :param output_folder: (str) Will create train and train_annot folder and put properly formated masks there
    :return: None
    """
    masks = glob.glob(''.join((input_folder, r'\*_mask.tif')))
    for m in masks:
        im_mask = imageio.imread(m)
        rows, cols, _ = im_mask.shape
        o_mask = np.zeros((rows, cols), dtype='uint8')
        o_mask[im_mask[:, :, 0] == 255] = 0  # matrix
        o_mask[im_mask[:, :, 0] == 79] = 1  # precip
        o_mask[im_mask[:, :, 0] == 198] = 2  # matrix
        o_file = m.replace(input_folder, ''.join((output_folder, r'\train_annot')))
        o_file = o_file.replace('_mask', '')
        imageio.imwrite(o_file, o_mask)

        im_file = m.replace('_mask', '')
        im = imageio.imread(im_file)
        o_file = im_file.replace(input_folder, ''.join((output_folder, r'\train')))
        imageio.imwrite(o_file, im)

def gimp_to_label(input_folder, output_folder, patch_size=None):
    masks = glob.glob(''.join((input_folder, r'\*.png')))
    for m in masks:
        print(m)
        f = m.replace('.png', '.tiff')
        if not os.path.exists(f):
            f = m.replace('.png', '.tif')
            if not os.path.exists(f):
                f = m.replace('.png', '.jpg')

        im = imageio.imread(f)  # load image
        im = im[:im.shape[1], :]  # remove scale bar by setting height = width.
        im = inout.uint8(resize(im, (1024, 1024)))  # convert to consistent size
        im = gray2rgb(im)  # convert to color
        #im = gray2rgb(imageio.imread(f)[:1024, :])

        #im_mask = imageio.imread(m)[:1024, :, :3]
        im_mask = imageio.imread(m)  # load image
        im_mask = im_mask[:im_mask.shape[1], :]  # remove scale bar by setting height = width.
        im_mask = inout.uint8(resize(im_mask, (1024, 1024), order=0, anti_aliasing=False))  # convert to consistent size

        rows, cols, _ = im.shape
        o_mask = np.zeros((rows, cols), dtype='uint8')
        o_mask[im_mask[:, :, 0] - im_mask[:,:,1] - im_mask[:,:,2] == 255] = 1  # oxide
        o_mask[im_mask[:, :, 2] - im_mask[:,:,1] - im_mask[:,:,0] == 255] = 2  # crack
        o_mask[im_mask[:, :, 2] + im_mask[:, :, 1] + im_mask[:, :, 0] < 255] = 0 # not sure why needed to remove spurious cracks

        name = os.path.splitext(os.path.basename(f))[0]

        if patch_size is None:
            o_file = output_folder + r'\train_annot\\' + name + '.tif'
            if os.path.exists(o_file):
                print(o_file + ' already exists!')
            else:
                imageio.imwrite(o_file, o_mask)
                o_file = output_folder + r'\train\\' + name + '.tif'
                imageio.imwrite(o_file, im)
        else:
            ims = get_patches(im, patch_size)
            masks = get_patches(o_mask, patch_size)
            for i, (im, o_mask) in enumerate(zip(ims, masks)):
                o_file = output_folder + r'\train_annot\\' + name + '_' + str(i) + '.tif'
                if os.path.exists(o_file):
                    print(o_file + ' already exists!')
                else:
                    imageio.imwrite(o_file, o_mask)
                    o_file = output_folder + r'\train\\' + name + '_' + str(i) + '.tif'
                    imageio.imwrite(o_file, im)



def get_patches(im, patch_size):
    ims = []
    try:
        rows, cols, _ = im.shape
    except ValueError:
        im = gray2rgb(im)
        rows, cols, _ = im.shape
    for i in range(rows // patch_size[0]):
        for j in range(cols // patch_size[1]):
            ims.append(im[i*patch_size[0]:(i+1)*patch_size[0],
                          j*patch_size[1]:(j+1)*patch_size[1],
                         :])
    return ims


def save_patches(image_path, patch_size):
    """
    Generates patches of large images and saves them
    :param image_path: (str) file to get patches from
    :return: None
    """
    _, ext = os.path.splitext(image_path)
    ims = imageio.imread(image_path)
    ims = get_patches(ims, patch_size)
    for i, im in enumerate(ims):
        o = image_path.replace(ext, ''.join(('_', str(i), '.tif')))
        imageio.imwrite(o, im)

def folder_to_patches(in_folder, out_folder, patch_size, ftype='tif', trim=None, size=None):
    """
    Generates patches of large images and saves them
    :param image_path: (str) file to get patches from
    :return: None
    """
    inout.create_folder(out_folder)
    for image_path in inout.files_in_folder(in_folder, ftype):
        _, ext = os.path.splitext(image_path)
        if trim is None:
            ims = imageio.imread(image_path)
        else:
            ims = gray2rgb(imageio.imread(image_path))[trim[0]: trim[1], trim[2]: trim[3], :]
        if size is not None:
            ims = inout.img_as_ubyte(resize(ims, size))
        ims = get_patches(ims, patch_size)
        for i, im in enumerate(ims):
            o = image_path.replace(ext, ''.join(('_', str(i), '.tif'))).replace(in_folder, out_folder)
            imageio.imwrite(o, im)



def generate_training_from_full_data(imfile, maskfile, trainfolder, name):
    ims = imageio.imread(imfile)
    masks = imageio.imread(maskfile)
    ims = get_patches(ims, (512, 512))
    masks = get_patches(masks, (512, 512))
    for i, (im, im_mask) in enumerate(zip(ims, masks)):
        rows, cols, _ = im_mask.shape
        o_mask = np.zeros((rows, cols), dtype='uint8')
        o_mask[im_mask[:, :, 0] == 255] = 0  # matrix
        o_mask[im_mask[:, :, 0] == 79] = 1  # precip
        o_mask[im_mask[:, :, 0] == 198] = 2  # matrix
        o_file = trainfolder + r'\train_annot\\' + name + '_' + str(i) + '.tif'
        imageio.imwrite(o_file, o_mask)

        o_file = trainfolder + r'\train\\' + name + '_' + str(i) + '.tif'
        imageio.imwrite(o_file, im)

def create_train_val_test_data_folders(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    folders = [folder + '\\' + f for f in ['train', 'train_annot', 'val', 'val_annot', 'test', 'test_annot']]
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)

if __name__ == '__main__':
    # f = r'C:\Users\jstuckne\OneDrive - NASA\Documents\CurrentProjects\Scannapieco_David\Image\10AP-Robomet'
    # imfile = f + r'\ISGRCop-10AP-000013_Mosaic_crop.tif'
    # maskfile = f + r'\ISGRCop-10AP-000013_Mosaic_crop_mask.tif'
    #trainfolder = r'C:\Users\jstuckne\OneDrive - NASA\Documents\Code\Jupyter\seg_models_pytorch\data\ISGRCop'

    # name = 'crop13'
    # generate_training_from_full_data(imfile, maskfile, trainfolder, name)

    #im_path = r'C:\Users\jstuckne\OneDrive - NASA\Documents\CurrentProjects\Image Classification\data_v3\SmithT_H111\High resolution BSD bulk image - rgb.tif'
    #save_patches(im_path, (1000, 1000))

    #input_folder = r'C:\Users\jstuckne\OneDrive - NASA\Documents\CurrentProjects\Image Classification\data_v3\BASF'
    #output_folder = r'C:\Users\jstuckne\OneDrive - NASA\Documents\CurrentProjects\Image Classification\data_v3\BASF_split'
    #folder_to_patches(input_folder, output_folder, patch_size=(447,640), ftype='tif', trim=[0, 447, 0, 640], size=(447*2, 640*2))


    # out_folder = r'C:\Users\jstuckne\OneDrive - NASA\Documents\Code\Jupyter\seg_models_pytorch\data\harder512'
#     # create_train_val_test_data_folders(out_folder)
#     # in_folder = r'C:\Users\jstuckne\OneDrive - NASA\Documents\CurrentProjects\Bryan_Harder\labels'
#     # gimp_to_label(in_folder, out_folder, patch_size=(512,512))

    output_folder = r'C:\Users\jstuckne\OneDrive - NASA\Documents\CurrentProjects\Tim_Smith\hand_segmentation'
    input_folder = r'C:\Users\jstuckne\OneDrive - NASA\Documents\CurrentProjects\TimSmith_CV_SuperAlloy\Images\All__Original_images'
    folder_to_patches(input_folder, output_folder, patch_size=(512,512))