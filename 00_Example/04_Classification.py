# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:53:17 2019

@author: Wook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1) 

num_epochs=200

# make the sample data

n_data = torch.ones(100,2)              # tensor([100,2])
x0 = torch.normal(2*n_data,1)           # Gaussian distribution mean=2*n_data, std=1 , tensor([100,2])
y0 = torch.zeros(100)                   # class 0 y data tensor([100])
x1 = torch.normal(-2*n_data,1)          # class 1 x data tensor([100,2])
y1 = torch.ones(100)                    # class 1 y data tensor([100])

x = torch.cat((x0,x1),0).type(torch.FloatTensor)        # torch.size([200,2])
y = torch.cat((y0,y1), ).type(torch.LongTensor)         # torch.size([200])

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=50, lw=0, cmap='RdYlGn')
plt.show()


class MyNet(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(MyNet,self).__init__()
        
        self.hidden = nn.Linear(n_feature,n_hidden)         # hidden layer
        self.out    = nn.Linear(n_hidden,n_output)          # output layer
        
    def forward(self, x):
        x = F.relu(self.hidden(x))                          # activation function for hidden layer
        x = self.out(x)
        return x


model = MyNet(n_feature=2,n_hidden=10,n_output=2)          # define the network
print(model)

optimizer = optim.SGD(model.parameters(),lr=0.095)
loss_func = nn.CrossEntropyLoss()                       # the target label is NOT an one-hotted

#
train_loss_list = []

plt.ion()

for epoch in range(num_epochs):
    
    out = model(x)                                      # input x and predict based on x
    # print(out.shape) [200,2]
    loss = loss_func(out, y)                            # must be (model output, target)
    
    # clear gradients for next train
    optimizer.zero_grad()
    # backpropagation, compute gradients
    loss.backward()
    # apply gradient
    optimizer.step()
    
    train_loss_list.append(loss.item())
    
    
    if epoch %2 ==0:
        print('loss: %.4f ' % loss.item())
        plt.cla()
        prediction = torch.max(out,1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c = pred_y, s=50, lw=0, cmap='RdYlGn')
        accuracy = (pred_y == target_y).sum().item() / target_y.size 
        plt.text(1.5,-4,'Accuracy=%.2f'%accuracy, fontdict={'size':15,'color':'red'})
        plt.pause(0.1)
plt.ioff()

plt.plot(np.arange(num_epochs),train_loss_list,'g-',lw=3)
plt.title('train loss')
plt.ylabel('loss')

plt.show()

        
        
        
        
        
        
        
        
        
        
        
        
        


