"""
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import copy
import cv2
import numpy as np
import os
import torch
from torch.autograd import Variable
from torchvision import models
from torch.optim import SGD
from torch.nn import functional

from class_table import class_table


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im

"""
@author: Colton Hill - github.com/ColtonChill
"""

def get_params(init_class_idx, target_class_idx):
    # strip of the extra labels on the table
    init_class_label = class_table[init_class_idx].split(',')[0]
    target_class_label = class_table[target_class_idx].split(',')[0]
    # make a path to find & save the images
    sub_dir = f'output_imgs/{init_class_label} --> {target_class_label}'
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    files = {
        'init_class_label':init_class_label,
        'target_class_label':target_class_label,
        'init_image_path':os.path.join('input_imgs',f"{init_class_idx}_{init_class_label.replace(' ','_')}.JPEG"),
        'original_image_path':os.path.join(sub_dir,'original.JPEG'),
        'spooffed_image_path':os.path.join(sub_dir,'spooffed.JPEG'),
    }

    # Read image & Process image
    prep_img = preprocess_image(cv2.imread(files['init_image_path'], 1))
    return ( prep_img, files)