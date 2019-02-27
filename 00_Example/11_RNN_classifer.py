# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:17:52 2019

@author: Wook
"""


import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_GPU else "cpu");print(device)

# Set the hyper-parmeters
EPOCH = 5                   # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64                      
TIME_STEP = 28              # RNN time step / image height
INPUT_SIZE = 28             # RNN input size / image width
LEARNING_RATE = 0.01        # learning rate
DOWNLOAD_MNIST = True       # set to True if havent't download the data

train_data = MNIST(
        root='./mnist/',
        train=True,                                     # this is training data
        transform=transforms.ToTensor(),    # Converts a PLT.image or numpy.ndarry to
                                                        # torch.FloatTensor of shape (C x H x W) and normailize in the range([0.0,1.0])
        download=DOWNLOAD_MNIST
)

# plot one example
print(train_data.train_data.size())     # [60000, 28, 28]
print(train_data.train_labels.size())   # [60000]
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i'% train_data.train_labels[0])
plt.show()

# Data loader for easy mini-bacth return in training, the image batch shape will be (50,1,28,28)
train_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
print(train_loader.__len__()) # 60000 / bacth size = 1200 

# convert test data into Variable , pick 2000 samples to speed up testing
test_data = MNIST(root='./mnist',train=False,transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy()[:2000] # (2000, 1, 28, 28) # convert to numpy array

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        
        self.rnn = nn.LSTM(             # if use nn.RNN()), it hardly learns
            input_size = INPUT_SIZE,
            hidden_size = 64,           # RNN hidden unit
            num_layers = 1,             # number of RNN layer
            batch_first = True          # input & output will has batch size as 1s dimenstion. e.g (batch,time_step,input_size)
        )
        
        self.fc= nn.Linear(64,10)

    def forward(self, x):
        r_out,(h_n,h_c) = self.rnn(x,None) # None represents zero initial hidden state
        
        # choose r_out at the last time step
        out= self.fc(r_out[:,-1,:])
        return out
    
    
model = RNN().to(device)
print(model)

optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)     # optimize all CNN parameter
loss_func = nn.CrossEntropyLoss()                               # the target label is not one-hotted

train_losses = []
# training and testing
for epoch in range(EPOCH):
    for step, (b_x,b_y) in enumerate(train_loader):             # gives batch data
        
        b_x = b_x.view(-1,28,28)
        output = model(b_x)                                     # RNN output
        loss = loss_func(output,b_y)                            # cross entropy loss
        optimizer.zero_grad()                                   # clear gradients for this training step
        loss.backward()                                         # backpropagation, compute gradients
        optimizer.step()                                        # apply gradients
        
        train_losses.append(loss.item())
        
        if step % 50 ==0:
            test_output = model(test_x)                         # (samples, time_step, input_size)
            pred_y = torch.max(test_output,1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
        

# print 10 predictions from test data
test_output = model(test_x[:10].view(-1,28,28))
pred_y = torch.max(test_output,1)[1].data.numpy()
print(pred_y,'prediction number')
print(test_y[:10],'real number')

# show play list
plt.plot(train_losses,'g-',lw=3)
plt.title('RNN train loss')
plt.ylabel('loss')
plt.show()
        
        
        
        