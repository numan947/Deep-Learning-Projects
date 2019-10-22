import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    layers = [
              nn.BatchNorm2d(in_channels),
              nn.ReLU(inplace=True),
              nn.Conv2d(in_channels, out_channels, 3, padding=1)
    ]

    return nn.Sequential(*layers)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(DenseBlock, self).__init__()
        conv_blocks = []
        for i in range(num_convs):
            conv_blocks.append(conv_block(i*out_channels+in_channels, out_channels))
        self.densenet = nn.Sequential(*conv_blocks)
    
    def forward(self, x):
        for layer in self.densenet:
            y = layer(x)
            x = torch.cat((x,y), dim=1)
        return x


def transition_block(in_channels, out_channels):
    layers = [
              nn.BatchNorm2d(in_channels),
              nn.ReLU(inplace=True),
              nn.Conv2d(in_channels, out_channels, 1),
              nn.AvgPool2d(2, 2)
    ]

    return nn.Sequential(*layers)


class DenseNet(nn.Module):
    def __init__(self, in_channels, num_labels, growth_rate, 
                 num_convs_in_dense_blocks=[4, 4, 4, 4]):
        super(DenseNet, self).__init__()

        self.initial_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, 2, 1)
        )

        num_channels = 64
        layers = [] 
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            layers.append(DenseBlock(num_channels, growth_rate, num_convs))
            num_channels+=num_convs*growth_rate

            if i != len(num_convs_in_dense_blocks)-1:
                layers.append(transition_block(num_channels, num_channels//2))
                num_channels//=2

        self.densenetlayers = nn.Sequential(*layers)
        # print(num_channels)
        self.lastlayers = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(num_channels, 10)
        )
    
    def forward(self, x):
        x = self.initial_layers(x)
        # print(x.shape)
        x = self.densenetlayers(x)
        # print(x.shape)
        x = self.lastlayers(x)

        return x