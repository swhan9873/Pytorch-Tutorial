# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:10:22 2019

@author: Wook
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# make the sample data
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
print(x, x.shape)
y = x.pow(2) + 0.2*torch.rand(x.size())


def save():
    
    num_epochs = 100
    # save net1
    
    net1 = nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,1)
    )
    
    optimizer = optim.SGD(net1.parameters(),lr=0.5)
    loss_func = nn.MSELoss()
    
    
    for epoch in range(num_epochs):
        prediction = net1(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    # plot result
    plt.figure(1, figsize=(13,5))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
    
    # 2 ways to save the net
    torch.save(net1,'net.pkl') # save entire net
    torch.save(net1.state_dict(),'net_params.pkl') # save only the parameters
    
def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    
    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'g-',lw=5)
    

def restore_params():
    # restore only the parameters in net1 to net3
    
    net3 = nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,1)
    )
    
    # copy net1's paramters into net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    
    # plot result
    plt.subplot(133)
    plt.title('Net2')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'y-',lw=5)

# save net1
save()

# restore entire net (may slow)
restore_net()

# restore only the net parameters
restore_params()


