
import torch
import torch.nn as nn

from matplotlib import colors, pyplot as plt

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from IPython.display import clear_output


class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
        
    def forward(self, x):
        # x_real = x[..., 0]
        # x_im = x[..., 1]
        x_real = torch.real(x)
        x_im = torch.imag(x)
        
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        # output = torch.stack([c_real, c_im], dim=-1)
        output = torch.complex(c_real, c_im)
        return output
    

class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        x_real = torch.real(x)
        x_im = torch.imag(x)
        
        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)  
        
        output = torch.complex(n_real, n_im)
        return output
    
class CLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01) -> None:
        super().__init__()

        self.negative_slope = negative_slope
        self.real_activation = nn.LeakyReLU(negative_slope)
        self.imag_activation = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x_real = torch.real(x)
        x_imag = torch.imag(x)

        r_real = self.real_activation(x_real)
        r_imag = self.imag_activation(x_imag)

        output = torch.complex(r_real, r_imag)

        return output


class Encoder(nn.Module):
    
    def __init__(self, filter_size=(3,3), stride_size=(1,1), in_channels=32, out_channels=32, padding=(1,1)):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = CLeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)
        
        return acted
