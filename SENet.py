import torch
import torch.nn as nn
import torch.nn.functional as F



'''-------------一、SE模块-----------------------------'''

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False), 
            nn.Sigmoid()
        )
 
    def forward(self, x):

            b, c, h, w = x.size()
            y = self.gap(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)