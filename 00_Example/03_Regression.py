# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:12:11 2019

@author: Wook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

num_epochs = 300

# unsqueeze 가 무슨 lib 지?
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1) # torch.size([100]) ---> torch.size([100,1(dim)])
print(x,x.size()) # torch.size([100, 1])

noisy = 0.1 * torch.rand(x.size())

y = x.pow(3) + noisy

print(y)

class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)     # hidden layer
        self.predict = nn.Linear(n_hidden,n_output)     # output layer
    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


# define the network
model = Net(n_feature=1,n_hidden=5,n_output=1)
print(model)

# lr 큰값으로 바꾸면 발산함. lr 을 작은 값으로하면 학습이 너무 안됨 
optimizer = optim.SGD(model.parameters(),lr=0.25)
loss_func = nn.MSELoss() # this is for regression mean squared loss (MSE)

plt.ion() # somthing about plotting

train_loss_list = []
for epoch in range(num_epochs):
    prediction = model(x) # input x and prediction based on x
    
    loss = loss_func(prediction,y)  # must be (nn.output, target)
    optimizer.zero_grad()           # claer gradients for next train 
    loss.backward()                 # backpropagatioin, compute gradients
    optimizer.step()                # apply gradients
    
    train_loss_list.append(loss.item())
    if epoch %5 ==0:
        # plot and show learning process
        print('loss: %.4f ' % loss.item())
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=4)
        # loss.item() = loss.data.numpy()
        plt.text(0.5,0,'Loss=%.4f' % loss.item(),fontdict={'size':14,'color':'red'})
        plt.pause(0.1)
plt.ioff()

# show the loss 

plt.plot(np.arange(num_epochs),train_loss_list,'g-',lw=3)
plt.title('train loss')
plt.ylabel('loss')
plt.show()




        
        
        
        
        