import torch
import argparse  
import re
import os
import numpy as np
import sys
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from collections import OrderedDict

from tqdm import tqdm

from natsort import natsorted
from model import CT_Mamba
from measure import compute_measure

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

def normalize_( image, MIN_B=0.0, MAX_B=4096.0):
        image = (image - MIN_B) / (MAX_B - MIN_B)
        return image

def denormalize(slice_data, min_val=0, max_val=4096.0):
    return slice_data * (max_val - min_val) + min_val

def load_ctmamba(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

# Use this function to load the provided CT-Mamba pretrained model. (module renaming)
def load_ctmamba_pretrained(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            if name.startswith("PRN_in."):
                name = name.replace("PRN_in.", "PFEN.")
            elif name.startswith("PRN_out."):
                name = name.replace("PRN_out.", "PFFN.")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def save_npy(output_folder1, x, y, pred, fig_name):

        np.save(os.path.join(output_folder1, 'result_{}_low.npy'.format(fig_name)), x)
        np.save(os.path.join(output_folder1, 'result_{}_pred.npy'.format(fig_name)), pred)
        np.save(os.path.join(output_folder1, 'result_{}_full.npy'.format(fig_name)), y)

def save_fig(output_folder, x, y, pred, fig_name, original_result, pred_result, trunc_min, trunc_max):  

        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
        ax[0].set_title('Low-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
        ax[1].set_title('CT-Mamba', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        
        f.savefig(os.path.join(output_folder,  'result_{}.png'.format(fig_name)))
        plt.close()

def split_arr(arr,patch_size,stride=32):    ## 512*512 to 32*32
    pad = (16, 16, 16, 16) # pad by (0, 1), (2, 1), and (3, 3)
    arr = nn.functional.pad(arr, pad, "constant", 0)
    _,_,h,w = arr.shape
    num = h//stride - 1
    arrs = torch.zeros(num*num,1,patch_size,patch_size)

    for i in range(num):
        for j in range(num):
            arrs[i*num+j,0] = arr[0,0,i*stride:i*stride+patch_size,j*stride:j*stride+patch_size]
    return arrs

def agg_arr(arrs, size, stride=32):  ## from 32*32 to size 512*512
    arr = torch.zeros(size, size)
    n,_,h,w = arrs.shape
    num = size//stride
    for i in range(num):
        for j in range(num):
            arr[i*stride:(i+1)*stride,j*stride:(j+1)*stride] = arrs[i*num+j,:,16:48,16:48]

    return arr.unsqueeze(0).unsqueeze(1)


def trunc(mat, trunc_min=None, trunc_max=None):  
        mat[mat <= trunc_min] = trunc_min
        mat[mat >= trunc_max] = trunc_max
        return mat

def main(args):
    trunc_min = -160 + 1024
    trunc_max = 240 + 1024
    data_range = 4096.0
    
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu_ids)
        torch.cuda.set_device(args.gpu_ids)
    else:
        device = torch.device("cpu")

    input_folder = args.input_folder
    target_folder = args.target_folder
    output_folder = args.output_folder
    output_folder1 = args.output_folder1
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)

    netG = CT_Mamba().to(device)  
    load_ctmamba_pretrained(netG, args.model_path)
    netG.eval() 

    input_files = natsorted(os.listdir(input_folder))
    target_files = natsorted(os.listdir(target_folder))
    counter = 1 
    ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    for input_file, target_file in tqdm(zip(input_files, target_files), desc="Processing...", unit="files"):
        input_path = os.path.join(input_folder, input_file)
        target_path = os.path.join(target_folder, target_file)

        input = np.load(input_path) 
        input = torch.from_numpy(input)
        input = input.unsqueeze(0).unsqueeze(1)  
        target = np.load(target_path)
        target = torch.from_numpy(target)
        target = target.unsqueeze(0).unsqueeze(1)  
        input = input.to(device) 
        input = input.float()
        target = target.to(device) 
        target = target.float() 
        with torch.no_grad():
            arrs = split_arr(input, 64).cuda()  ## split to image patches for test into 4 patches
            arrs[0:64] = netG(arrs[0:64])
            arrs[64:2*64] = netG(arrs[64:2*64])
            arrs[2*64:3*64] = netG(arrs[2*64:3*64])
            arrs[3*64:4*64] = netG(arrs[3*64:4*64])
            output = agg_arr(arrs, 512).cuda()

        output = denormalize(output).squeeze().cpu().detach().numpy()    
        input = denormalize(input).squeeze().cpu().detach().numpy()  
        target = denormalize(target).squeeze().cpu().detach().numpy()  

        save_npy(output_folder1, input, target, output, counter)

        original_result, pred_result = compute_measure(input, target, output, data_range)
        ori_psnr_avg += original_result[0]
        ori_ssim_avg += original_result[1]
        ori_rmse_avg += original_result[2]
       
        pred_psnr_avg += pred_result[0]
        pred_ssim_avg += pred_result[1]
        pred_rmse_avg += pred_result[2]

        save_fig(output_folder, input, target, output, counter, original_result, pred_result, trunc_min, trunc_max)

        counter = counter + 1
    
    print('\n')
    print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(target_files), 
                                                                                         ori_ssim_avg/len(target_files), 
                                                                                         ori_rmse_avg/len(target_files)
                                                                                            ))
    print('\n')
    print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(target_files), 
                                                                                            pred_ssim_avg/len(target_files), 
                                                                                            pred_rmse_avg/len(target_files)
                                                                                            ))


parser = argparse.ArgumentParser() 
parser.add_argument('--model_path', type=str, default='CT-Mamba_pretrained.pth', dest='model_path') 
parser.add_argument('--gpu_ids', type=int, default='0', dest='gpu_ids') 
parser.add_argument('--input_folder', type=str, default='./test/input', dest='input_folder')
parser.add_argument('--target_folder', type=str, default='./test/target', dest='target_folder')
parser.add_argument('--output_folder', type=str, default='./save/fig', dest='output_folder')
parser.add_argument('--output_folder1', type=str, default='./save/npy', dest='output_folder1')
args = parser.parse_args()

if __name__ == '__main__':
    main(args)
    