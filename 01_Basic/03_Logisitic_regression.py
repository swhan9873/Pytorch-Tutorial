# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:32:33 2019

@author: Wook
"""



import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Hyper-parameter
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

# Logistic regression model
model = nn.Linear(input_size,num_classes)
#model = nn.Sequential(
#            nn.Linear(input_size,num_classes),
#            nn.ReLU()
#        )

# Loss and optimizer
# CrossentorpyLoss computes softmax internally
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for batch_idx,(images,labels) in enumerate(train_loader):
        # Reshape images to (batcH_size,input_size)
        # images = images.view(-1,28*28)  # [100,784]
        images = images.view(images.size(0),-1) # flatten the input data to fit the linear model dimension
        
        # Forward pass
        outputs = model(images)
        loss = loss_func(outputs,labels)
        
        
        # Backward and opimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx +1) % 100 ==0:
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'
                  .format(epoch+1,num_epochs,batch_idx+1,total_step,loss.item()))
            

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

with torch.no_grad():
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images = images.view(images.size(0),-1)
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum()
    print('Accuracy of the model on the 10000 test images: {} %'.format(100*correct/total))    
        
            
            