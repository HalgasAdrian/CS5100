import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a feedforward neural network. 

Please only complete the model definition and do not include any training code.

The model should be a feedforward neural network, that accepts 784 inputs (each image is 28x28, and is flattened for input to the network)
and the output size is 10. Whether you need to normalize outputs using softmax depends on your choice of loss function.

PyTorch documentation is available at https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton 
requires normalized outputs or not.

'''

class FF_Net(nn.Module):
    def __init__(self):
        super(FF_Net, self).__init__()

        # Setting up a simple feedforward network.
        self.fc1 = nn.Linear(784, 256) # First fully connected layer.
        self.fc2 = nn.Linear(256, 128) # Second fully connected layer.
        self.fc3 = nn.Linear(128, 64) # Third fully connected layer.
        self.fc4 = nn.Linear(64, 10) # Output layer for 10 classes.

    def forward(self, x):
        # Forward pass through the network.
        x = F.relu(self.fc1(x)) # Apply ReLU activation.
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation on output layer because it is handled by loss function.

        return x
        
