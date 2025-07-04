import os
import sys
import matplotlib.pyplot as plt
import warnings
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import datetime
from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
warnings.filterwarnings("ignore")
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'..'))
import argparse
import optionsmamba
import utils
from dataset.dataset_denoisemamba import *
from torch.utils.tensorboard import SummaryWriter
import NPSloss
torch.backends.cudnn.benchmark = True 
######### parser ########### 
opt = optionsmamba.Options().init(argparse.ArgumentParser(description='low dose image denoising')).parse_args()
print(opt)



######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu  









######### Logs dir ###########
log_dir = os.path.join(opt.save_dir, 'denoising')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
train_img_result = os.path.join(log_dir, 'train_img_result')
val_img_result = os.path.join(log_dir, 'val_img_result')
utils.mkdir(result_dir) 
utils.mkdir(model_dir)
utils.mkdir(train_img_result)
utils.mkdir(val_img_result)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Loss ########### 
criterion = NPSloss.NPSloss()
criterion1 = NPSloss.NPSloss1()
criterion_val = nn.L1Loss()

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(
        list(model_restoration.parameters()) + \
            list(criterion.unet_feature.parameters()) + \
                 list(criterion.ss2d.parameters()) + \
                    list(criterion.unet_feature1.parameters()) + \
                        list(criterion.ss2d1.parameters()), 
        lr=opt.lr_initial, 
        betas=(0.9, 0.999),
        eps=1e-8, 
        weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ###########   
model_restoration = torch.nn.DataParallel(model_restoration) 
model_restoration.cuda() 
     

######### Scheduler ###########  
if opt.warmup: 
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Resume ###########  
    
if opt.resume: 
    path_chk_rest = opt.pretrain_weights   
    path_chk_rest_ss2d = opt.pretrain_weights_SS2D
    path_chk_rest_ss2d1 = opt.pretrain_weights_SS2D1
    path_chk_rest_uFeatureNet = opt.pretrain_weights_uFeatureNet
    path_chk_rest_uFeatureNet1 = opt.pretrain_weights_uFeatureNet1
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration, path_chk_rest)
    utils.load_checkpoint(criterion.unet_feature, path_chk_rest_uFeatureNet)
    utils.load_checkpoint(criterion.unet_feature1, path_chk_rest_uFeatureNet1)
    utils.load_checkpoint(criterion.ss2d, path_chk_rest_ss2d)
    utils.load_checkpoint(criterion.ss2d1, path_chk_rest_ss2d1)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    lr = utils.load_optim(optimizer, path_chk_rest) 


    for i in range(1, start_epoch):
        scheduler.step()  
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')



######### DataLoader ########### 
print('===> Loading datasets')

train_dataset = get_training_data(opt.train_dir, patch_size=opt.patch_size, patch_n=opt.patch_n)  
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=False, drop_last=False)
val_dataset = get_validation_data(opt.val_dir, patch_size=opt.patch_size, patch_n=opt.patch_n)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset) 

num_batch_train = int((len_trainset / opt.batch_size) + ((len_trainset / opt.batch_size) != 0))
num_batch_val = int((len_valset / opt.batch_size) + ((len_valset % opt.batch_size) != 0))

writer = SummaryWriter(os.path.join('tensorboard_log', time.strftime("%Y%m%d-%H%M%S")))

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
best_loss = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

def split_arr(arr, patch_size, stride=32):           
    pad = (16, 16, 16, 16) 
    arr = nn.functional.pad(arr, pad, "constant", 0)
    _,_,h,w = arr.shape
    num = h//stride - 1
    arrs = torch.zeros(num*num, 1, patch_size, patch_size)

    for i in range(num):
        for j in range(num):
            arrs[i*num+j,0] = arr[0,0,i*stride:i*stride+patch_size,j*stride:j*stride+patch_size]
    return arrs

def agg_arr(arrs, size, stride=32):  
    arr = torch.zeros(size, size)
    n,_,h,w = arrs.shape
    num = size//stride
    for i in range(num):
        for j in range(num):
            arr[i*stride:(i+1)*stride,j*stride:(j+1)*stride] = arrs[i*num+j,:,16:48,16:48]
    return arr.unsqueeze(0).unsqueeze(1)

loss_scaler = NativeScaler() 
torch.cuda.empty_cache() 
total_index = 0

for epoch in range(start_epoch, opt.nepoch + 1):  
    
    epoch_start_time = time.time()
    epoch_loss = 0

    model_restoration.train() 
    
    for i, (data) in enumerate(tqdm(train_loader), 0): 
        
        input_ = data['input'].cuda()
        target = data['label'].cuda()
        
        if opt.patch_size:
            input_ = input_.view(-1, 1, opt.patch_size, opt.patch_size)  
            target = target.view(-1, 1, opt.patch_size, opt.patch_size)


        pixelSpacing = 1.0 # Adjust according to the specific data.
        restored = model_restoration(input_)
        optimizer.zero_grad()
        if epoch <= 10: 
           loss, loss1, loss2, loss3, loss4 = criterion1(input_, restored, target, pixelSpacing) 
        else:
            loss, loss1, loss2, loss3, loss4 = criterion(input_, restored, target, pixelSpacing) 
        loss_scaler(loss, 
                    optimizer, 
                    parameters = list(model_restoration.parameters()) + 
                                 list(criterion.unet_feature.parameters()) + 
                                 list(criterion.ss2d.parameters()) + 
                                 list(criterion.unet_feature1.parameters()) + 
                                 list(criterion.ss2d1.parameters())
                                 ) 

        epoch_loss += loss.item()
        total_index += 1
        
        if i % 50 == 0:
           
            print(f'loss:{loss.item()}, loss1:{loss1.item()}, loss2:{loss2.item()}, loss3:{loss3.item()}, loss4:{loss4.item()}')
               
            writer.add_scalar("Train/loss", loss.item(), total_index)
            writer.add_scalar("Train/loss1", loss1.item(), total_index)
            writer.add_scalar("Train/loss2", loss2.item(), total_index)
            writer.add_scalar("Train/loss3", loss3.item(), total_index)
            writer.add_scalar("Train/loss4", loss4.item(), total_index)    

        

        #### Evaluation ####

        if (i+1) % eval_now==0 and i > 0:
            with torch.no_grad():
                model_restoration.eval()
                valLoss = 0
                for ii, (data_val) in enumerate(tqdm(val_loader), 0):
                

                    input_ = data_val['input'].cuda()
                    target = data_val['label'].cuda()

                    arrs = split_arr(input_, 64).cuda()  
                    arrs[0:64] = model_restoration(arrs[0:64])
                    arrs[64:2*64] = model_restoration(arrs[64:2*64])
                    arrs[2*64:3*64] = model_restoration(arrs[2*64:3*64])
                    arrs[3*64:4*64] = model_restoration(arrs[3*64:4*64])
                    restored = agg_arr(arrs, 512).cuda()

                    
                    loss = criterion_val( restored , target)  * 100 + 1e-4
                    
                        
                    

                    if (ii % 50)==0:
                        print(f'loss:{loss.item()}')
                        
                    writer.add_scalar("Valid/loss", loss.item(), total_index)
               
                torch.cuda.empty_cache()

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss/len_trainset, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f :
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss/len_trainset, scheduler.get_lr()[0])+'\n')


    if epoch % opt.checkpoint == 0:   
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
        # torch.save({'epoch': epoch, 
        #             'state_dict': criterion.unet_feature.state_dict(),
        #             'optimizer' : optimizer.state_dict()
        #             }, os.path.join(model_dir,"unet_feature_epoch_{}.pth".format(epoch))) 
        # torch.save({'epoch': epoch, 
        #             'state_dict': criterion.unet_feature1.state_dict(),
        #             'optimizer' : optimizer.state_dict()
        #             }, os.path.join(model_dir,"unet_feature1_epoch_{}.pth".format(epoch)))
        # torch.save({'epoch': epoch, 
        #             'state_dict': criterion.ss2d.state_dict(),
        #             'optimizer' : optimizer.state_dict()
        #             }, os.path.join(model_dir,"ss2d_epoch_{}.pth".format(epoch)))
        # torch.save({'epoch': epoch, 
        #             'state_dict': criterion.ss2d1.state_dict(),
        #             'optimizer' : optimizer.state_dict()
        #             }, os.path.join(model_dir,"ss2d1_epoch_{}.pth".format(epoch)))

        
print("Now time is : ",datetime.datetime.now().isoformat())
