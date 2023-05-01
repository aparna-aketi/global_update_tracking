import torch
#from torchinfo import summary

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from copy import deepcopy


__all__ = ['LeNet5']

class LeNet5(nn.Module):
    def __init__(self):
        """
        Build a LeNet5 pytorch Module.
        """
        super(LeNet5, self).__init__()
        # feature extractor
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # classifer
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Define the forward pass of a LeNet5 module.
        """
        # feature extraction
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        # classification
        x = x.view(x.size(0),-1) # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out