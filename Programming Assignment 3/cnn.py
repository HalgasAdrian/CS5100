import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a convolutional neural network. 

Please only complete the model definition and do not include any training code.

The model should be a convolutional neural network, that accepts 28x28 grayscale images as input, and outputs a tensor of size 10.
The number of layers/kernels, kernel sizes and strides are up to you. 

Please refer to the following for more information about convolutions, pooling, and convolutional layers in PyTorch:

    - https://deeplizard.com/learn/video/YRhxdVk_sIs
    - https://deeplizard.com/resource/pavq7noze2
    - https://deeplizard.com/resource/pavq7noze3
    - https://setosa.io/ev/image-kernels/
    - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


Whether you need to normalize outputs using softmax depends on your choice of loss function. PyTorch documentation is available at
https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton requires normalized outputs or not.

'''

class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()

        # Defining the layers of our CNN.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # First convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Second convolutional layer

        # Fully connected layers.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 7x7 comes from pooling on 28x28 -> 14x14 -> 7x7
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Output layer for 10 classes

    def forward(self, x):

        # Forward pass through the network
        x = F.relu(self.conv1(x))  # First convolutional layer + ReLU
        x = self.pool(x)           # Pooling layer
        x = F.relu(self.conv2(x))  # Second convolutional layer + ReLU
        x = self.pool(x)           # Pooling layer
        
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor for the fully connected layers
        x = F.relu(self.fc1(x))     # First fully connected layer + ReLU
        x = F.relu(self.fc2(x))     # Second fully connected layer + ReLU
        x = self.fc3(x)             # Output layer (no activation, handled by loss function)
        
        return x
        
