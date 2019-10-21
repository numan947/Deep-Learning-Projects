import torch.nn as nn
import torch



# config example:
VGG11 = [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']

def vgg_block(config, in_channels):
    layers = []
    for x in config:
        if x == 'M':
            layers.append(nn.MaxPool2d(2,2))
        else:
            layers.append(nn.Conv2d(in_channels, x,3,padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = x
    return nn.Sequential(*layers)

class VGGNet(nn.Module):
    def __init__(self, vgg_config, in_channel, num_classes):
        super(VGGNet, self).__init__()
        self.vgg = vgg_block(vgg_config, in_channel)
        self.adjustor = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Linear(512*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.vgg(x)
        x = self.adjustor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x