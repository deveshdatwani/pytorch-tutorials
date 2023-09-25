import torch
from torch import nn as nn 
from torch.functional import F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_channels, 3, 1)
        self.conv2 = nn.Conv2d(hid_channels, out_channels, 3, 1)
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = F.relu()
        
    
    def forward(self, x):
        x_identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x += x_identity

        return x