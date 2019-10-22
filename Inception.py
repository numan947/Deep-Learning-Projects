import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_channel, out1, out2, out3, out4): ## out1, out4 --> Single number | out2, out3 --> tuples
        super(Inception, self).__init__()
        
        self.path1 = nn.Conv2d(in_channel, out1, 1)
        
        self.path2a = nn.Conv2d(in_channel, out2[0], 1)
        self.path2b = nn.Conv2d(out2[0], out2[1], 3, padding=1)

        self.path3a = nn.Conv2d(in_channel, out3[0], 1)
        self.path3b = nn.Conv2d(out3[0], out3[1], 5, padding=2)

        self.path4a = nn.MaxPool2d(3, 1, 1)
        self.path4b = nn.Conv2d(in_channel, out4, 1)

    def forward(self, x):
        
        path1_out = torch.relu(self.path1(x))

        path2_out = torch.relu(self.path2b(torch.relu(self.path2a(x))))

        path3_out = torch.relu(self.path3b(torch.relu(self.path3a(x))))

        path4_out = torch.relu(self.path4b(self.path4a(x)))

        return torch.cat((path1_out, path2_out, path3_out, path4_out), dim=1)