import Inception.Inception as Inception
import torch
import torch.nn as nn


class GoogLeNet(nn.Module):
    def __init__(self, in_channel, num_labels):
        super(GoogLeNet, self).__init__()

        self.layer12= nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channel, 64, 7, 2, 3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 192, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1)
            )
        )

        self.layer3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.dense = nn.Linear(1024, num_labels)
    
    def forward(self, x):
        x = self.layer12(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x= self.dense(x)
        return x