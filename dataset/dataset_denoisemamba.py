import numpy as np
import os
from torch.utils.data import Dataset
import torch
import sys
import copy
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob
import re
from scipy.ndimage import zoom


class Dataset_train(Dataset): 
    def __init__(self, rgb_dir, transform='True', patch_size=None, patch_n=None):
        super(Dataset_train, self).__init__()
        
        self.transform = transform
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.input_ = self.sort_files(glob(os.path.join(rgb_dir, 'input' ) + '/*.*'))
        self.target_ = self.sort_files(glob(os.path.join(rgb_dir, 'target' ) + '/*.*'))

        self.tar_size = len(self.target_)
         
    def sort_files(self, file_list): 
        return natsorted(file_list)
    
    def __len__(self):
        return self.tar_size

    
    def __getitem__(self, index): 
        input_img, target_img = self.input_[index], self.target_[index] 
        input_filename = os.path.basename(input_img)
        target_filename = os.path.basename(target_img)
        input_img, target_img = np.load(input_img), np.load(target_img)
        if self.patch_size:
            input_img, target_img = get_patch(input_img, target_img, self.patch_n, self.patch_size)
        
        
        input = torch.from_numpy(input_img).float() 
        label = torch.from_numpy(target_img).float() 

        data = {'label': label, 'input': input, 'input_name': input_filename, 'target_name': target_filename}
        
        return data  

class Dataset_val(Dataset): 
    def __init__(self, rgb_dir, transform='True', patch_size=None, patch_n=None):
        super(Dataset_val, self).__init__()
        
        self.transform = transform
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.input_ = self.sort_files(glob(os.path.join(rgb_dir, 'input' ) + '/*.*'))
        self.target_ = self.sort_files(glob(os.path.join(rgb_dir, 'target' ) + '/*.*'))

        self.tar_size = len(self.target_)  
         
    def sort_files(self, file_list): 
        return natsorted(file_list)
    
    def __len__(self):
        return self.tar_size
    
    def __getitem__(self, index): 
        input_img, target_img = self.input_[index], self.target_[index] 
        input_filename = os.path.basename(input_img)
        target_filename = os.path.basename(target_img)
        input_img, target_img = np.load(input_img), np.load(target_img)
        
        if input_img.ndim == 2:  
            input_img = np.expand_dims(input_img, axis=2)
        if target_img.ndim == 2:  
            target_img = np.expand_dims(target_img, axis=2)
        
        

        input_img = input_img.transpose((2, 0, 1)).astype(np.float16)
        target_img = target_img.transpose((2, 0, 1)).astype(np.float16)
        input = torch.from_numpy(input_img).float() 
        label = torch.from_numpy(target_img).float() 

        data = {'label': label, 'input': input, 'input_name': input_filename, 'target_name': target_filename}
        
        return data  


def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    if patch_size == h:    
        return full_input_img, full_target_img
    for _ in range(patch_n):  
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
        
        
    return np.array(patch_input_imgs), np.array(patch_target_imgs)


def get_training_data(rgb_dir, patch_size, patch_n):  
    assert os.path.exists(rgb_dir)
    return Dataset_train(rgb_dir, patch_size=patch_size, patch_n=patch_n)


def get_validation_data(rgb_dir, patch_size, patch_n):
    assert os.path.exists(rgb_dir)
    return Dataset_val(rgb_dir, patch_size=patch_size, patch_n=patch_n)

