import os
import torch
import torch.nn as nn
import torchvision  
from torchvision.transforms import Compose
import numpy as np
from PIL import Image

class SimpleAffine(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleAffine, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.fc1 = nn.Linear(1024 * input_channels, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(128, self.output_channels)

    def forward(self, x):
        x = self.flat(x)
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        y = self.relu(y)
        y = self.flat(y)
        y = self.fc(y)
        # print(y.shape)
        return y
