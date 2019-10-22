import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, use_1x1_conv=False, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if use_1x1_conv:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride)
        else:
            self.conv3 = None
    
    def forward(self, x):
        Y = torch.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            x = self.conv3(x)
        return torch.relu(x+Y)