import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
from ffn import *
import matplotlib.pyplot as plt

'''

In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.

'''


'''

PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted. Please do not change the transforms given below - the autograder assumes these.

'''

transform = transforms.Compose([                            # Use transforms to convert images to tensors and normalize them
    transforms.ToTensor(),                                  # convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])             # Common method for grayscale images
])

batch_size = 64


'''

PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.

'''

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

'''

PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.

'''

feedforward_net = FF_Net()
conv_net = Conv_Net()

'''

PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.

'''

criterion = nn.CrossEntropyLoss()

optimizer_ffn = optim.Adam(feedforward_net.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.001)

'''

PART 5:
Train both your models, one at a time! (You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs, but it is not recommended for this assignment.)

'''

losses_ffn = [] # Tracking FNN loss.

num_epochs_ffn = 10 # Starting with 10, adjust based on results.

for epoch in range(num_epochs_ffn):  # loop over the dataset multiple times
    running_loss_ffn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Flatten inputs for ffn
        inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs to shape (batch_size, 784)

        # zero the parameter gradients
        optimizer_ffn.zero_grad()

        # forward + backward + optimize
        outputs = feedforward_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ffn.step()
        running_loss_ffn += loss.item()
        losses_ffn.append(loss.item())  # Save loss for each batch

    print(f"Training loss: {running_loss_ffn}")

print('Finished Training')

torch.save(feedforward_net.state_dict(), 'ffn.pth')  # Saves model file (upload with submission)

losses_cnn = [] # Tracking CNN loss.

num_epochs_cnn = 10 # Starting with 10, adjust based on our results.

for epoch in range(num_epochs_cnn):  # loop over the dataset multiple times
    running_loss_cnn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer_cnn.zero_grad()

        # forward + backward + optimize
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()
        losses_cnn.append(loss.item())  # Save loss for each batch

    print(f"Training loss: {running_loss_cnn}")

print('Finished Training')

torch.save(conv_net.state_dict(), 'cnn.pth')  # Saves model file (upload with submission)

'''

PART 6:
Evalute your models! Accuracy should be greater or equal to 80% for both models.

Code to load saved weights commented out below - may be useful for debugging.

'''

# feedforward_net.load_state_dict(torch.load('ffn.pth'))
# conv_net.load_state_dict(torch.load('cnn.pth'))

correct_ffn = 0
total_ffn = 0

correct_cnn = 0
total_cnn = 0

with torch.no_grad():           # since we're not training, we don't need to calculate the gradients for our outputs
    for data in testloader:
        inputs, labels = data

        # Evaluating the FFN
        inputs_ffn = inputs.view(inputs.size(0), -1)  # Flatten for FFN
        outputs_ffn = feedforward_net(inputs_ffn)
        _, predicted_ffn = torch.max(outputs_ffn, 1)
        total_ffn += labels.size(0)
        correct_ffn += (predicted_ffn == labels).sum().item()

        # Evaluating the CNN
        outputs_cnn = conv_net(inputs)
        _, predicted_cnn = torch.max(outputs_cnn, 1)
        total_cnn += labels.size(0)
        correct_cnn += (predicted_cnn == labels).sum().item()

print('Accuracy for feedforward network: ', correct_ffn/total_ffn)
print('Accuracy for convolutional network: ', correct_cnn/total_cnn)


'''

PART 7:

Check the instructions PDF. You need to generate some plots. 

'''
def plot_correct_and_incorrect(model, model_name, data_loader, classes, is_ffn=False):
    model.eval()  # Setting our model to evaluation mode.
    correct_found = False
    incorrect_found = False
    correct_image = None
    correct_pred = None
    correct_label = None
    incorrect_image = None
    incorrect_pred = None
    incorrect_label = None

    with torch.no_grad():
        for images, labels in data_loader:
            if is_ffn:
                images_flat = images.view(images.size(0), -1)  # Flatten for FFN
                outputs = model(images_flat)
            else:
                outputs = model(images)
            
            _, predictions = torch.max(outputs, 1)

            for i in range(len(predictions)):
                if predictions[i] == labels[i] and not correct_found:
                    correct_image = images[i]
                    correct_pred = predictions[i].item()
                    correct_label = labels[i].item()
                    correct_found = True
                elif predictions[i] != labels[i] and not incorrect_found:
                    incorrect_image = images[i]
                    incorrect_pred = predictions[i].item()
                    incorrect_label = labels[i].item()
                    incorrect_found = True
                
                if correct_found and incorrect_found:
                     break
            if correct_found and incorrect_found:
                break

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(correct_image.squeeze(), cmap='gray')
    axs[0].set_title(f"Correctly Classified\nTrue: {classes[correct_label]}, Pred: {classes[correct_pred]}")
    axs[0].axis('off')
    
    axs[1].imshow(incorrect_image.squeeze(), cmap='gray')
    axs[1].set_title(f"Incorrectly Classified\nTrue: {classes[incorrect_label]}, Pred: {classes[incorrect_pred]}")
    axs[1].axis('off')

    # Class names for Fashion-MNIST
    classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    # Call function for both networks
    plot_correct_and_incorrect(feedforward_net, "Feedforward Network", testloader, classes, is_ffn=True)  # FFN
    plot_correct_and_incorrect(conv_net, "Convolutional Network", testloader, classes, is_ffn=False)  # CNN


#losses_ffn = []
#losses_cnn = []

# During FFN training
#losses_ffn.append(running_loss_ffn / len(trainloader))

# During CNN training
#losses_cnn.append(running_loss_cnn / len(trainloader))

def plot_loss_curve(losses, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', label=f'{model_name} Loss')
    plt.xlabel('Batch Iterations')  # Number of batches passed
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot for each model
plot_loss_curve(losses_ffn, "Feedforward Network")
plot_loss_curve(losses_cnn, "Convolutional Network")
