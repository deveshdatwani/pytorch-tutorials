import torch
from torch import nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(hid_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        x_identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x += x_identity

        return x


class fireNET(nn.Module):
    def __init__(self, blocks=8, n_class=2):
        super().__init__()
        self.n_class = n_class
        self.blocks = blocks
        self.block_channels = [64, 64, 64]
        self.first_conv = nn.Conv2d(3, 64, 3, 1, 1)
        self.resnet_layers = self.make_resnet_blocks(blocks)
    

    def make_resnet_blocks(self, blocks):
        layers = []
        layers.append(self.first_conv)
        in_channel = 64
        
        for i in range(blocks):
            out_channel = self.block_channels[i%3]
            layers.append(ResNetBlock(in_channel, out_channel, out_channel))
            in_channel = out_channel
        
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.resnet_layers(x)
        return x
    


if __name__ == '__main__':
    model = fireNET()
    
    x = torch.rand(1,3,476,476)
    y = model(x)

    print(y.shape)