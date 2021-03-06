# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:47:07 2019

@author: Wook
"""


import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #


# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 



# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #


# Create tensors.
x = torch.tensor(1.,requires_grad = True)
w = torch.tensor(2.,requires_grad = True)
b = torch.tensor(3.,requires_grad = True)

# Build a computational graph
y = w * x + b

# Compute gradients
y.backward()

# Print out the gradients
print(x.grad)       # dy/dx = w = 2
print(w.grad)       # dy/dw = x = 1
print(b.grad)       # dy/db = 0+1 = 1

# Get the value (python number)
print(x.item())
print(w.item())
print(b.item())


# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #


# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(4,3);print(x,x.shape)
y = torch.randn(4,2);print(y)

# Build a fully connected layer.
linear = nn.Linear(3,2)
print('w:',linear.weight, linear.weight.shape)
print('b:',linear.bias, linear.bias.shape)

# Build loss function and optimizer
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(),lr=0.01)


# Forward pass
output = linear(x)
print('output:', output,output.shape)

# Compute loss
loss = loss_func(output,y)
print('loss:',loss.item())

# Backward pass
loss.backward()

# Print out the gradients (L : loss)
print('dL/dw:', linear.weight.grad)
print('dL/db:', linear.bias.grad)


# 1-step gradient descent
optimizer.step()

# Print out the loss after 1-step gradient descent
output = linear(x)
loss = loss_func(output,y)
print('loss after 1 step optimization:', loss.item())


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #


# Create a numpy array
x = np.array([[1,2],[3,4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x) ; print(y)

# Conver the torch tensor to a numpy array.

z = y.numpy()



# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #


# Download and construct CIFAR-10 datasets.
train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True,transform=transforms.ToTensor(),download=True)



# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print(image.size())     # [3,32,32] / RGB x width x height
print(label)            # 6


# Data loader (this provides queues and treads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=64, shuffle=True)

# When iteration starts, queue and thread start to load data from files
data_iter = iter(train_loader)


# Mini-batch images and labels
images, labels = data_iter.next()


# Actual usage of the data loader is as below
for images, labels in train_loader:
    # Training code should be written here.
    pass



# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #

# You should build your custom dataset as below

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file path or a list of file names.
        pass
    def __getitem__(self,index):
        # TODO
        # 1. Read one data from file (e.g using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (e.g torchvision.Transform)
        # 3. Return a data pair (e.g image and label)
        pass
    
    def __len__(self):
        # You should chnage 0 to the total size of your dataset
        return 0
    
    
# You can then use the prebuilt data loader.
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,batch_size=64, shuffle= True)

