import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from HFEN import HFEN
import pywt   
import ptwt
from mamba import *
from resnet import ResNet10
from subnet import Encoder
from SENet import SE_Block


class ResBlock(nn.Module):
    def __init__(
            self,
            channels,
        ):
        super().__init__()
        
        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels),
            nn.PReLU(),     
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels),
            nn.PReLU(),
        )

    def forward(self, x):

        identity = x
        x = self.res(x)
        x = torch.add(x, identity)
        return x


class MSC_Mamba(nn.Module):   # MSC_Mamba
    def __init__(self, 
                 patch_size = 4, 
                 in_chans = 64,
                 dims = 256, 
                 final_dims = 64//2,  
                 dim_scale = 4,
                 drop_rate = 0.1,
                 attn_drop_rate = 0.,
                 norm_layer = nn.LayerNorm,
                 d_state = 16,
                ) -> None:
        super().__init__()

        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.vssblock1 = CZSS(hidden_dim=in_chans, patch=32, drop_path=drop_rate, norm_layer=norm_layer, 
                                 attn_drop_rate=attn_drop_rate,d_state=d_state)

        self.linear1 = nn.Conv2d(in_chans, final_dims, 3, 1, 1)
        
        self.patch_embed2 = PatchEmbed2D(patch_size//2, in_chans=in_chans, embed_dim=dims, norm_layer=nn.LayerNorm)
        self.vssblock2 = CZSS(hidden_dim=dims, patch=16, drop_path=drop_rate, norm_layer=norm_layer, 
                                 attn_drop_rate=attn_drop_rate,d_state=d_state)
        self.Final_PatchExpand2D2 = Final_PatchExpand2D_16(dim=dims, dim_scale=dim_scale//2)

        self.linear2 = nn.Conv2d(in_chans, final_dims, 3, 1, 1)

        self.patch_embed3 = PatchEmbed2D(patch_size, in_chans=in_chans, embed_dim=dims*4, norm_layer=nn.LayerNorm)
        self.vssblock3 = CZSS(hidden_dim=dims*4, patch=8, drop_path=drop_rate, norm_layer=norm_layer, 
                                 attn_drop_rate=attn_drop_rate,d_state=d_state)
        self.Final_PatchExpand2D3 = Final_PatchExpand2D_8(dim=dims*4, dim_scale=dim_scale)
        self.linear3 = nn.Conv2d(in_chans, final_dims, 3, 1, 1)

        self.mambaout = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32*3, final_dims, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(final_dims),
            nn.PReLU(),     
            nn.ReflectionPad2d(1),
            nn.Conv2d(final_dims, final_dims, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(final_dims),
            nn.PReLU(),
        )

        self.vssblock = CZSS(hidden_dim=32, patch=32, drop_path=drop_rate, norm_layer=norm_layer, 
                                 attn_drop_rate=attn_drop_rate,d_state=d_state)
        
        self.resblock = ResBlock(channels=in_chans)

    def forward(self, x):

        x = self.resblock(x)

        x = x.permute(0, 2, 3, 1).contiguous()

        x1 = self.vssblock1(x)

        x1 = x1.permute(0, 3, 1, 2).contiguous()        

        x2 = self.resblock(x1)
        x2 = self.patch_embed2(x2)
        x2 = self.pos_drop(x2)
        x2 = self.vssblock2(x2)
        x2 = self.vssblock2(x2)
        x2 = self.Final_PatchExpand2D2(x2).permute(0, 3, 1, 2).contiguous()       

        x3 = self.resblock(x1)
        x3 = self.patch_embed3(x3)
        x3 = self.pos_drop(x3)
        x3 = self.vssblock3(x3)
        x3 = self.vssblock3(x3)
        x3 = self.vssblock3(x3)
        x3 = self.vssblock3(x3)
        x3 = self.Final_PatchExpand2D3(x3).permute(0, 3, 1, 2).contiguous()       
        
        x1 = self.linear1(x1)          
        x2 = self.linear2(x2)
        x3 = self.linear3(x3)


        x123 = torch.cat((x1, x2, x3), dim=1)
        x123 = self.mambaout(x123).permute(0, 2, 3, 1).contiguous()
 
        x123 = self.vssblock(x123).permute(0, 3, 1, 2).contiguous()    
        
        return x123


class FFTNet(nn.Module):
    def __init__(self, in_chans) -> None:
        super().__init__()
        self.Cconv_bn_lrelu = Encoder(in_channels=in_chans, out_channels=in_chans)
        self.senet = SE_Block(inchannel=in_chans)

    def forward(self, x):
        
        x = torch.fft.fft2(x)
        x = self.Cconv_bn_lrelu(x)

        x_amp = torch.fft.fftshift(torch.abs(x))
        x_amp = self.senet(x_amp)
        x_amp = torch.fft.ifftshift(x_amp)

        x_pha = torch.fft.fftshift(torch.angle(x))
        x_pha = self.senet(x_pha)
        x_pha = torch.fft.ifftshift(x_pha)

        real = x_amp * torch.cos(x_pha)
        imag = x_amp * torch.sin(x_pha)

        complex = torch.complex(real, imag)
        ifft_complex = torch.fft.ifft2(complex)

        x = torch.real(ifft_complex).to(torch.float32)

        return x


class MambaLowLayer(nn.Module):
    def __init__(self, 
                 patch_size=2,
                 in_chans=64,
                 dims=64, 
                 dim_scale=4,
                 drop_rate=0.1,
                 attn_drop_rate = 0.,  
                 norm_layer=nn.LayerNorm,
                 d_state = 16
                 ) -> None:
        super().__init__()

        self.resnet10 = ResNet10()
        self.fftnet = FFTNet(in_chans=in_chans//2)
        self.mambalayer = MSC_Mamba(in_chans=in_chans, dims=dims)

    def forward(self, x):

        B, C, H, W = x.shape

        x1, x2 = torch.split(x, C//2, dim=1)

        x1_out = self.fftnet(x1)

        x2_out = self.resnet10(x2)

        out = torch.cat((x1_out, x2_out), dim=1)


        out = self.mambalayer(out)

        return out

    
class PFEN(nn.Module):
    def __init__(self):
        super(PFEN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),  
            nn.PReLU(),
            nn.Conv2d(4, 8, 3, 1, 1), 
            nn.PReLU(),
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 20, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(20, 24, 3, 1, 1),
            nn.PReLU(),
        )


    def forward(self, input):
        b, c, h, w = input.shape
        x = input
        x = self.conv1(x)

        x_repeat = input.repeat(1, 8, 1, 1)
        x = torch.cat((x, x_repeat), 1)

        x = self.conv2(x)
        x = torch.cat((x, x_repeat), 1)
        
        return x

class PFFN(nn.Module):                          
    def __init__(self):
        super(PFFN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 12, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(12, 8, 3, 1, 1),
            nn.PReLU()
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 8, 1, 1, 0), 
            nn.PReLU()
            )
 
        self.conv = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4, 1, 3, 1, 1),
            nn.PReLU()
            )
        
        
    def forward(self, input):
        b, c, h, w = input.shape
        x1, x2 = torch.split(input, c//2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = x1 + x2
        x = self.conv(x)

        return x


    

class CT_Mamba(nn.Module):  
    def __init__(self, in_channels=1, out_channels=1, pretrained=False, **kwargs):
        super(CT_Mamba, self).__init__()
        
        self.Net_low = MambaLowLayer(in_chans=64//2, dims=64*4//2)
        self.Net_H = MSC_Mamba(in_chans=128//2, dims=128*4//2) 
        self.Net_D = MSC_Mamba(in_chans=192//2, dims=192*4//2) 
        self.Net_V = MSC_Mamba(in_chans=128//2, dims=128*4//2)  
        self.UNet_H = HFEN(in_channels=64//2, out_channels=64//2)
        self.UNet_D = HFEN(in_channels=64//2, out_channels=64//2)
        self.UNet_V = HFEN(in_channels=64//2, out_channels=64//2)
        self.wavelet = pywt.Wavelet('haar') 
        self.framework_outConv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) 
        self.relu_saptial = nn.PReLU()  
        self.PFEN = PFEN()
        self.PFFN = PFFN()

    def forward(self, x):
        residual = x  
        x = self.PFEN(x)
        residual1 = x 
        
        coeffs = ptwt.wavedec2(x, self.wavelet, level=1)    
        cA, (cH, cV, cD) = coeffs
        cA_output = self.Net_low(cA) 
        H_fusion = self.UNet_H(cA)
        V_fusion = self.UNet_V(cV)
        D_fusion = self.UNet_D(cD)

        H_input = torch.cat([cH, D_fusion], dim=1)   
        V_input = torch.cat([cV, D_fusion], dim=1)   
        D_input = torch.cat([cD, V_fusion, H_fusion], dim=1)  


        H_output = self.Net_H(H_input)
        D_output = self.Net_D(D_input)
        V_output = self.Net_V(V_input)

        reconstructed_coeffs = (cA_output, (H_output, V_output, D_output))
        reconstructed_image = ptwt.waverec2(reconstructed_coeffs, self.wavelet)  
        
        residual_sum = residual1 + reconstructed_image   
        residual_sum = self.PFFN(residual_sum)
        output = residual_sum + residual   

        return output