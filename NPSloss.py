import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision import models
import numpy as np
import random
# from model import MambaLayer
import mamba
from functools import partial
# import ssim

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def radial_profile(data):

    batch_size, channels, height, width = data.shape
    assert channels == 1

    radial_means = []

    for i in range(batch_size):

        single_data = data[i, 0, :, :]
        center = (single_data.shape[0] // 2, single_data.shape[1] // 2)

        y, x = torch.meshgrid(torch.arange(single_data.shape[0]), torch.arange(single_data.shape[1]), indexing='ij')
        r = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)  

        r = r.int()
        max_radius = min(height, width) // 2
        radial_mean = torch.zeros(max_radius, dtype=torch.float32)
        for radius in range(max_radius):
            mask = (r == radius)
            radial_mean[radius] = single_data[mask].mean()
        
        radial_means.append(radial_mean)

    radial_means = torch.stack(radial_means)
    
    return radial_means



class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, pred, target):

        assert pred.shape == target.shape

        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)

        pred_diff = pred - pred_mean
        target_diff = target - target_mean

        covariance = torch.sum(pred_diff * target_diff)

        pred_var = torch.sqrt(torch.sum(pred_diff ** 2))
        target_var = torch.sqrt(torch.sum(target_diff ** 2))

        if pred_var == 0 or target_var == 0:
            return torch.tensor(1.0)  

        pearson_corr = covariance / (pred_var * target_var)

        loss = 1 - pearson_corr
        
        return loss


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Encoder, self).__init__()
        self.conv_in_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.PReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, is_pool=True, is_conv_in_relu=True): 
        if is_pool:
            x = self.pool(x)
        if is_conv_in_relu:
            x = self.conv_in_relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, transposeconv_in_channels=1, transposeconv_out_channels=1):
        super(Decoder, self).__init__()
        self.conv_in_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )
        self.transposeconv = nn.Sequential(
            nn.ConvTranspose2d(transposeconv_in_channels, transposeconv_out_channels, 
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(transposeconv_out_channels),
            nn.PReLU()
        )
    def forward(self, x, is_conv_in_relu=True, is_transposeconv=True):
        if is_conv_in_relu:
            x = self.conv_in_relu(x)
        if is_transposeconv:
            x = self.transposeconv(x)
        return x

class MLF_FEC(nn.Module):
    def __init__(self, in_chans=64, radius=150) -> None:
        super().__init__()

        self.ConvBranch1 = nn.Sequential(
            nn.ReflectionPad2d(1),  
            nn.Conv2d(in_chans, in_chans, (3, 3), (1, 1), (0, 0)),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_chans, in_chans, (3, 3), (1, 1), (0, 0)),
            )
        self.radius = radius

        self.ConvBranch2 = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
            nn.Conv2d(in_chans, in_chans, (1, 1), (1, 1), (0, 0)),
            )
        
    def forward(self, x):

        y1 = self.ConvBranch1(x)

        y2_fft = torch.fft.fft2(x)

        y2_amp = torch.abs(y2_fft)
        y2_pha = torch.angle(y2_fft)
        y2_amp = torch.fft.fftshift(y2_amp)

        _, _, H, W = x.shape

        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing = 'ij')
        center_y, center_x = H//2, W//2
        distance = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = (distance <= self.radius).cuda().float()

        y2_amp_mask = y2_amp * mask
        y2_amp_mask = self.ConvBranch2(y2_amp_mask)

        y2_amp = torch.fft.ifftshift(y2_amp_mask)

        real = y2_amp * torch.cos(y2_pha)
        imag = y2_amp * torch.sin(y2_pha)
        result = torch.complex(real, imag)
        y2 = torch.fft.ifft2(result)
        y2 = torch.real(y2).to(torch.float32)
        y = y1 + y2

        return y

class u_Feature_Net(nn.Module):  
    def __init__(self) -> None:
        super().__init__()

        self.encoder1 = Encoder(in_channels=1, out_channels=16)
        self.encoder2 = Encoder(in_channels=16, out_channels=32)
        self.decoder1 = Decoder(transposeconv_in_channels=32, transposeconv_out_channels=16)
        self.decoder2 = Decoder(in_channels=32, out_channels=16)
        self.outputer = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

        self.fft_connection1 = MLF_FEC(in_chans=16, radius=20)  
        self.fft_connection2 = MLF_FEC(in_chans=32, radius=20)

    def forward(self, x):
        x1 = self.encoder1(x, is_pool=False, is_conv_in_relu=True)
        x2 = self.encoder2(x1, is_pool=True, is_conv_in_relu=True)
        y2 = self.fft_connection2(x2)
        y1 = self.decoder1(y2, is_conv_in_relu=False, is_transposeconv=True)
        x1 = self.fft_connection1(x1)
        y1 = torch.cat([x1, y1], dim=1)
        y0 = self.decoder2(y1, is_conv_in_relu=True, is_transposeconv=False)
        y = self.outputer(y0)

        return y



class ResNet50FeatureExtractor(nn.Module):

    def __init__(self, blocks=[1, 2, 3, 4], pretrained=False, progress=True, **kwargs):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained, progress, **kwargs)
        del self.model.avgpool
        del self.model.fc
        self.blocks = blocks

    def forward(self, x):
        feats = list()

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if 1 in self.blocks:
            feats.append(x)

        x = self.model.layer2(x)
        if 2 in self.blocks:
            feats.append(x)

        x = self.model.layer3(x)
        if 3 in self.blocks:
            feats.append(x)

        x = self.model.layer4(x)
        if 4 in self.blocks:
            feats.append(x)

        return feats



class NPSloss(_Loss):
    def __init__(self, 
                 blocks=[1, 2, 3, 4],
                 mae_weight=1.0, 
                 feature_weight=0.01, 
                 radial_weight_L1=0.0001,
                 radial_weight_pearson=0.01) -> None:
        super().__init__()

        self.mae_weight = mae_weight                            
        self.feature_weight = feature_weight                    
        self.radial_weight_L1 = radial_weight_L1                
        self.radial_weight_pearson = radial_weight_pearson      

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.pearson_loss = PearsonLoss()

        self.blocks = blocks
        self.model = ResNet50FeatureExtractor(pretrained=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.unet_feature = u_Feature_Net().cuda()  
        self.unet_feature1 = u_Feature_Net().cuda()

        self.ss2d = mamba.Z_SS2D(d_model=1, patch=64).cuda()  
        self.ss2d1 = mamba.Z_SS2D(d_model=1, patch=64).cuda()

    def forward(self, input, output, target, pixelspacing):
        
        output_noise = input - output  
        target_noise = input - target  

        output_noise_f = self.unet_feature(output_noise)
        target_noise_f = self.unet_feature1(target_noise)

        output_noise_f_fft = torch.fft.fft2(output_noise_f)
        target_noise_f_fft = torch.fft.fft2(target_noise_f)

        output_noise_f_amp = torch.abs(output_noise_f_fft)
        target_noise_f_amp = torch.abs(target_noise_f_fft)

        output_noise_f_nps = pixelspacing * pixelspacing * torch.fft.fftshift(output_noise_f_amp ** 2)
        target_noise_f_nps = pixelspacing * pixelspacing * torch.fft.fftshift(target_noise_f_amp ** 2)

        output_noise_f_nps = output_noise_f_nps.permute(0, 2, 3, 1).contiguous()
        target_noise_f_nps = target_noise_f_nps.permute(0, 2, 3, 1).contiguous()

        output_noise_f_result = self.ss2d(output_noise_f_nps)
        target_noise_f_result = self.ss2d1(target_noise_f_nps)

        output_noise_f_result = output_noise_f_result.permute(0, 3, 1, 2).contiguous()
        target_noise_f_result = target_noise_f_result.permute(0, 3, 1, 2).contiguous()

        output_noise_f_result_radial = radial_profile(output_noise_f_result)
        target_noise_f_result_radial = radial_profile(target_noise_f_result)
       

        loss1 = self.mae_weight * self.mae_loss(output, target) * 100 + 1e-4            
        loss2 = self.radial_weight_L1 * self.mae_loss(output_noise_f_result_radial, target_noise_f_result_radial)           
        loss3 = self.radial_weight_pearson * self.pearson_loss(output_noise_f_result_radial, target_noise_f_result_radial)         

        loss_value = 0
        output_feats = self.model(torch.cat([output, output, output], dim=1))
        target_feats = self.model(torch.cat([target, target, target], dim=1))

        feats_num = len(self.blocks)
        for idx in range(feats_num):
            loss_value += self.mse_loss(output_feats[idx], target_feats[idx])
        loss_value /= feats_num

        loss4 = self.feature_weight * loss_value     

        loss = loss1 + loss2 + loss3 + loss4

        return loss, loss1, loss2, loss3, loss4


class NPSloss1(_Loss): 
    def __init__(self, 
                 blocks=[1, 2, 3, 4],
                 mae_weight=1.0, 
                 feature_weight=0.01, 
                 radial_weight_L1=0.0,
                 radial_weight_pearson=0.0) -> None:
        super().__init__()

        self.mae_weight = mae_weight                            
        self.feature_weight = feature_weight                    
        self.radial_weight_L1 = radial_weight_L1                
        self.radial_weight_pearson = radial_weight_pearson      

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.pearson_loss = PearsonLoss()

        self.blocks = blocks
        self.model = ResNet50FeatureExtractor(pretrained=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.unet_feature = u_Feature_Net().cuda()  
        self.unet_feature1 = u_Feature_Net().cuda()

        self.ss2d = mamba.Z_SS2D(d_model=1, patch=64).cuda()  

    def forward(self, input, output, target, pixelspacing):
        
        output_noise = input - output  
        target_noise = input - target 

        output_noise_f = self.unet_feature(output_noise)
        target_noise_f = self.unet_feature1(target_noise)

        output_noise_f_fft = torch.fft.fft2(output_noise_f)
        target_noise_f_fft = torch.fft.fft2(target_noise_f)

        output_noise_f_amp = torch.abs(output_noise_f_fft)
        target_noise_f_amp = torch.abs(target_noise_f_fft)

        output_noise_f_nps = pixelspacing * pixelspacing * torch.fft.fftshift(output_noise_f_amp ** 2)
        target_noise_f_nps = pixelspacing * pixelspacing * torch.fft.fftshift(target_noise_f_amp ** 2)

        output_noise_f_nps = output_noise_f_nps.permute(0, 2, 3, 1).contiguous()
        target_noise_f_nps = target_noise_f_nps.permute(0, 2, 3, 1).contiguous()

        output_noise_f_result = self.ss2d(output_noise_f_nps)
        target_noise_f_result = self.ss2d(target_noise_f_nps)

        output_noise_f_result = output_noise_f_result.permute(0, 3, 1, 2).contiguous()
        target_noise_f_result = target_noise_f_result.permute(0, 3, 1, 2).contiguous()
        
        output_noise_f_result_radial = radial_profile(output_noise_f_result)
        target_noise_f_result_radial = radial_profile(target_noise_f_result)

        loss1 = self.mae_weight * self.mae_loss(output, target) * 100 + 1e-4           
        loss2 = self.radial_weight_L1 * self.mae_loss(output_noise_f_result_radial, target_noise_f_result_radial)         
        loss3 = self.radial_weight_pearson * self.pearson_loss(output_noise_f_result_radial, target_noise_f_result_radial)      

        loss_value = 0
        output_feats = self.model(torch.cat([output, output, output], dim=1))
        target_feats = self.model(torch.cat([target, target, target], dim=1))

        feats_num = len(self.blocks)
        for idx in range(feats_num):
            loss_value += self.mse_loss(output_feats[idx], target_feats[idx])
        loss_value /= feats_num

        loss4 = self.feature_weight * loss_value  

        loss = loss1 + loss2 + loss3 + loss4

        return loss, loss1, loss2, loss3, loss4