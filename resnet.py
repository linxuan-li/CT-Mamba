import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        
        
    def forward(self, x):
        identity = x
        
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  
        out = self.prelu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_chans):
        super(ResNet, self).__init__()
        
        self.in_channels = in_chans

        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chans)

        self.layer1 = self.make_layer(block, in_chans, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, in_chans, num_blocks[1], stride=1)

        self.prelu = nn.PReLU()
        
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:      
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.prelu(self.bn1(self.conv1(x)))

        return out


def ResNet10():
    return ResNet(BasicBlock, [2, 2], 16)