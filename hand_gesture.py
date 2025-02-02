import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F # Import F for functional operations

'''
os: Provides functions to interact with the operating system.
numpy: A library for numerical operations.
torchvision.transforms: Provides common image transformations.
torch.utils.data.Dataset, DataLoader: Utilities for handling datasets and creating data loaders.
torch: The main PyTorch library.
torch.nn: Contains neural network layers and functions.
torch.optim: Contains optimization algorithms.
PIL.Image: For image processing.
torch.nn.functional: Provides functional operations for neural networks.
kagglehub: A library to interact with Kaggle datasets'''


class LeapGestRecog(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.x = []
        self.y = []
        
        # Create a dictionary to map folder names to numerical labels
        self.class_to_idx = {}
        folders = os.listdir(root)
        for i, folder in enumerate(folders):
            self.class_to_idx[folder] = i

        for folder in folders:
            for dirpath, dirnames, filenames in os.walk(os.path.join(root, folder)):
                for filename in filenames:
                    self.x.append(os.path.join(dirpath, filename))
                    self.y.append(self.class_to_idx[folder]) # Store numerical label
        self.len = len(self.x)

        '''__init__ method: Initializes the dataset.
self.root: Root directory of the dataset.
self.transform: Transformations to be applied to the images.
self.x: List to store image file paths.
self.y: List to store corresponding labels.
self.class_to_idx: Dictionary to map folder names to numerical labels.
folders: List of folders in the root directory.
Loops: Iterates through folders and files to populate self.x and self.y with image paths and labels.
self.len: Stores the total number of images.
'''

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = Image.open(self.x[index]).convert('L')
        y = self.y[index]
        if self.transform:
            img = self.transform(img)
        return img, y
    '''__getitem__ method: Retrieves an image and its label by index.
img: Opens the image file and converts it to grayscale.
y: Retrieves the corresponding label.
Transforms: Applies transformations if provided.
Return: Returns the transformed image and its label.'''
path = 'leapgestrecog/leapGestRecog'
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), ])
dataset = LeapGestRecog(path, transform=transform)
'''transform: Defines a series of transformations to be applied to the images:
transforms.Resize((128, 128)): Resizes images to 128x128 pixels.
transforms.ToTensor(): Converts images to PyTorch tensors.
dataset: Initializes the LeapGestRecog dataset with the specified transformations.'''

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        # We will calculate the correct input size dynamically later
        self.fc1 = nn.Linear(0, 128) # Placeholder 
        self.fc2 = nn.Linear(128, 10)
        '''Net class: Defines a convolutional neural network.
__init__ method: Initializes the network layers:
self.conv1: First convolutional layer (1 input channel, 32 output channels, 5x5 kernel).
self.conv2: Second convolutional layer (32 input channels, 64 output channels, 5x5 kernel).
self.conv3: Third convolutional layer (64 input channels, 128 output channels, 5x5 kernel).
self.fc1: First fully connected layer (placeholder for input size, 128 output features).
self.fc2: Second fully connected layer (128 input features, 10 output features).'''

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten layer
        
        # Dynamically calculate input size for fc1
        if self.fc1.in_features == 0:  
            self.fc1 = nn.Linear(x.size(1), 128) # Set correct input size
            self.fc1.to(device)  # Make sure this new layer is on correct device
        '''x.view(x.size(0), -1): Flattens the tensor for the fully connected layers.
Dynamic Input Size Calculation:
Checks if fc1's input features are not set (in_features == 0).
Sets fc1 with the correct input size based on the flattened tensor.
Moves the new layer to the correct device.
F.relu(self.fc1(x)): Applies ReLU activation to the output of the first fully connected layer.
self.fc2(x): Applies the second fully connected layer.
F.log_softmax(x, dim=1): Applies log softmax to the output.'''    
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    '''forward method: Defines the forward pass of the network:
F.relu(self.conv1(x)): Applies ReLU activation to the output of the first convolutional layer.
F.max_pool2d(x, 2): Applies max pooling with a 2x2 kernel.
F.relu(self.conv2(x)): Applies ReLU activation to the output of the second convolutional layer.
F.max_pool2d(x, 2): Applies max pooling with a 2x2 kernel.
F.relu(self.conv3(x)): Applies ReLU activation to the output of the third convolutional layer.
x.view(x.size(0), -1): Flattens the tensor for the fully connected layers.
F.relu(self.fc1(x)): Applies ReLU activation to the output of the first fully connected layer.
self.fc2(x): Applies the second fully connected layer.
return x: Returns the final output.'''

model = Net()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Device:", device)
print("CUDA available:", torch.cuda.is_available())

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Move data and target to the device
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target) # Calculate loss
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
'''train function: Trains the model for one epoch.
model.train(): Sets the model to training mode.
Data Loop: Iterates over batches of data.
Moves data and target to the device.
Resets the gradients.
Performs a forward pass.
Calculates the loss.
Performs backpropagation.
Updates the model parameters.
Prints the training progress and loss'''

for epoch in range(1, 11):
    train(epoch)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
'''test function: Evaluates the model on the test dataset.
model.eval(): Sets the model to evaluation mode.
No Gradient Calculation: Disables gradient calculation for efficiency.
Data Loop: Iterates over batches of test data.
Moves data and target to the device.
Performs a forward pass.
Accumulates the test loss.
Calculates the number of correct predictions.
Average Loss: Computes the average test loss.
Accuracy: Computes the accuracy of the model.
Prints the test loss and accuracy.'''
test()
# test 1 using local GPU (NVIDIA GeForce GTX 1650) has 100% accuracy, highly likely tahat the model is overfitting