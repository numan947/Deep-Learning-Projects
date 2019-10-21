import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, channel):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(channel, 96, 11, 4)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.pool2 = nn.MaxPool2d(3,2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384,384, 3, padding=1)
        self.conv5 = nn.Conv2d(384,256,3, padding=1)
        self.pool3 = nn.MaxPool2d(3, 2)
        self.adjustor = nn.AdaptiveAvgPool2d((7,7))
        self.dense1 = nn.Linear(256*7*7, 4096)
        self.dense2 = nn.Linear(4096, 4096)
        self.dense3 = nn.Linear(4096,10)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.pool3(x)
        x = self.adjustor(x)
        x = x.view(-1,256*7*7)
        x = self.dropout(self.dense1(x))
        x = self.dropout(self.dense2(x))
        x = self.dense3(x) 
        return x
        # return nn.functional.log_softmax(x)