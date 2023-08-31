

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class ConvNet(nn.Module):

    """

    Create a pytorch class with this specs:
    • convolutional layer 1: 5 × 5 kernel, 6 output channels
    • convolutional layer 2: 5 × 5 kernel, 12 output channels
    • convolutional layer outputs are subject to 2 × 2 max-pooling

    """

    def __init__(self):
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv_output = nn.Conv2d(12, 1, 2)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.conv_output(x)
        return x
    


if __name__ == "__main__":

    img = np.array(Image.open("circle.png"))
    