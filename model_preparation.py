import segmentation_models_pytorch as smp
from torch.optim import Adam
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convLabel1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.convSeg1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)

        self.convLabel2 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.convSeg2 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)

        self.convLabel3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.convSeg3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsamping = nn.Upsample(scale_factor=8, mode="bilinear")
        self.bn = nn.BatchNorm2d(128)
        
    def forward(self, label, seg):
        # x = label.float()
        # y = seg.float()
        x = F.relu(self.convLabel1(label))
        x = self.maxpool(x)
        y = F.relu(self.convSeg1(seg))
        y = self.maxpool(y)

        x = F.relu(self.convLabel2(x))
        x = self.maxpool(x)
        y = F.relu(self.convSeg2(y))
        y = self.maxpool(y)

        x = F.relu(self.convLabel3(x))
        x = self.maxpool(x)
        y = F.relu(self.convSeg3(y))
        y = self.maxpool(y)

        x = torch.cat([x, y], dim=1)
        x = self.bn(x)
        x = self.upsamping(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1) 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x