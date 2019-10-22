import torch
import torch.nn as nn
import ResidualBlock.ResidualBlock as ResidualBlock



def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    blocks = []
    cur_in = in_channels ## Needed to adjust the depth of subsequent blocks dynamically
    for i in range(num_residuals):
        if i==0 and not first_block:
            blocks.append(ResidualBlock(cur_in, out_channels, use_1x1_conv=True, stride=2))
        else:
            blocks.append(ResidualBlock(cur_in, out_channels))
        cur_in = out_channels
    return nn.Sequential(*blocks)

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_labels):
        super(ResNet18, self).__init__()

        self.non_residuals = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.residuals = nn.Sequential(
            resnet_block(64, 64, 2, first_block=True),
            resnet_block(64, 128, 2),
            resnet_block(128, 256, 2),
            resnet_block(256, 512, 2),
        )

        self.global_average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(512, 10)

    def forward(self, x):
        x = self.non_residuals(x)
        x = self.residuals(x)
        x = self.global_average_pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x
