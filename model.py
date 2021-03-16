import os
import torch
import torch.nn as nn
import torchvision  
from torchvision.transforms import Compose
import numpy as np
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleCNN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(input_channels, 32, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(65536, self.output_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.flat(y)
        y = self.fc(y)
        # print(y.shape)
        return y
