
import torch
import torch.nn as nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    layers = [
              nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
              nn.ReLU(inplace=True),
              nn.Conv2d(out_channels, out_channels, 1),
              nn.ReLU(inplace=True),
              nn.Conv2d(out_channels, out_channels, 1),
              nn.ReLU(inplace=True)
              ]
    return nn.Sequential(*layers)


class NetworkInNetwork(nn.Module):
    def __init__(self, in_channels, num_labels):
        super(NetworkInNetwork, self).__init__()

        self.num_labels = num_labels
        self.in_channels = in_channels
        self.classifier = nn.Sequential(
            nin_block(self.in_channels, 96, 11, 4, 0),
            nn.MaxPool2d(3, 2),
            nin_block(96, 256, 5, 1, 2),
            nn.MaxPool2d(3,2),
            nin_block(256, 384, 3, 1, 1),
            nn.MaxPool2d(3,2),
            nn.Dropout(0.5),
            nin_block(384, 10, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1,1)),
        )
    def forward(self, x):
        x = self.classifier(x)
        return x.view(-1,10)