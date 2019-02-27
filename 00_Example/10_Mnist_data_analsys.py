# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:18:54 2019

@author: Wook
"""

# Load the libraries
import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set the parameters
num_workers = 0
batch_size = 20

# Converting the Images to tensors using Transforms
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data',train=True,
                            download=True,transform=transform)
test_data = datasets.MNIST(root='data',train=False,
                           download=True,transform=transform)

# Loading the Data
train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=num_workers)
test_loader = DataLoader(test_data,batch_size=batch_size,num_workers=num_workers)

import matplotlib.pyplot as plt

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()


# Peeking into dataset
fig = plt.figure(figsize=(25,4))
for image in np.arange(20):
    ax = fig.add_subplot(2,20/2,image+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(images[image]),cmap='gray')
    ax.set_title(str(labels[image].item()))
    


img = np.squeeze(images[7])
fig = plt.figure(figsize=(11,11))
ax = fig.add_subplot(111)
ax.imshow(img,cmap='gray')
width,height = img.shape
thresh = img.max()/2.5

for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y]!=0 else 0
        ax.annotate(str(val),xy=(y,x),
                    color='white' if img[x][y]<thresh else 'black')
        
        
        
        