import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
import matplotlib.pyplot as plt

conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))

# Get the weights of the first convolutional layer of the network

first_conv_weights = conv_net.conv1.weight.data

# Create a plot that is a grid of images, where each image is one kernel from the conv layer.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, 
# the grid might have 4 rows and 8 columns. Finally, normalize the values in the grid to be 
# between 0 and 1 before plotting.

min_val = first_conv_weights.min()
max_val = first_conv_weights.max()
normalized_weights = (first_conv_weights - min_val) / (max_val - min_val)

# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.

num_kernels = normalized_weights.shape[0]
grid_rows = int(np.ceil(num_kernels / 8))  # 8 columns
grid_cols = 8

fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 12))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < num_kernels:
        kernel = normalized_weights[i].squeeze().cpu().numpy()  # Convert to numpy for plotting
        ax.imshow(kernel, cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.savefig('kernel_grid.png')
plt.show()

# Apply the kernel to the provided sample image.

img = cv2.imread('/Users/adrianhalgas/Documents/GitHub/CS5100/Programming Assignment 3/sample_image.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image file not found, check path.")

img = cv2.resize(img, (28, 28))
img = img / 255.0					# Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)

print(img.shape)

# Apply the kernel to the image

with torch.no_grad():
    output = conv_net.conv1(img)  # Apply the first conv layer only

# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
# If not needed for your implementation, you can remove these lines.

output = output.squeeze(0)
output = output.unsqueeze(1)

# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.

output_min = output.min()
output_max = output.max()
normalized_output = (output - output_min) / (output_max - output_min)

# Save the grid to a file named 'image_transform_grid.png'. Add the saved image to the PDF report you submit.

num_channels = normalized_output.shape[0]
grid_rows = int(np.ceil(num_channels / 8))
grid_cols = 8

fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 12))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < num_channels:
        transformed_img = normalized_output[i].squeeze().cpu().numpy()  # Convert to numpy
        ax.imshow(transformed_img, cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.savefig('image_transform_grid.png')
plt.show()















