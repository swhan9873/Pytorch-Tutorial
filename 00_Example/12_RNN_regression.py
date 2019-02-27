# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:38:44 2019

@author: Wook

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# set the hyper parameter

TIME_STEP = 10  # RNN time step
INPUT_SIZE = 1  # RNN input size
LR = 0.02       # learning rate

# show data
steps = np.linspace(0,np.pi*2,100,dtype=np.float32) # float32 for converting torch Floattensor
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps,y_np,'r-',label='target (cos)')
plt.plot(steps,x_np,'b-',label='input (sin)')
plt.legend(loc='best')
plt.show()



class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        
        self.rnn = nn.RNN(
             input_size = INPUT_SIZE,
             hidden_size = 32,      # RNN hidden unit
             num_layers = 1,         # number of RNN layer
             batch_first = True,    # input & output will has batch size as 1s dimension. e.g (batch,time_step,input_size)
        )
        
        self.out = nn.Linear(32,1)
        
    def forward(self,x, h_state):
        
        r_out, h_state = self.rnn(x,h_state)
        outs = []
        for time_step in range(r_out.size(1)): # calculate output for each time step
            outs.append(self.out(r_out[:,time_step,:]))
        return torch.stack(outs,dim=1),h_state



model = RNN()
print(model)

optimizer = optim.Adam(model.parameters(),lr=LR)
loss_func = nn.MSELoss()

h_state = None # for initial hidden state

plt.figure(1,figsize=(12,5))
plt.ion()       # continuously plot

for step in range(100):
    start, end = step*np.pi, (step+1)*np.pi     # time range
    steps = np.linspace(start,end,TIME_STEP,dtype=np.float32,endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    
    
    x = torch.from_numpy(x_np[np.newaxis,:,np.newaxis])     # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis,:,np.newaxis])
    
    prediction, h_state = model(x, h_state)                 
    # next stop is important
    h_state = h_state.data      # repack the hidden state, break the connection from last iteration
    
    
    loss = loss_func(prediction,y)              # calculate loss
    optimizer.zero_grad()                       # clear gradients for this training step
    loss.backward()
    optimizer.step()
    
    
    # plotting
    plt.plot(steps,y_np.flatten(),'r-')
    plt.plot(steps,prediction.data.numpy().flatten(),'b-')
    plt.draw();plt.pause(0.05)

plt.ioff()
plt.show()
