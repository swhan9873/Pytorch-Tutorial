# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:08:11 2019

@author: Wook
"""

import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

LR =0.01
BATCH_SIZE= 32
EPOCH = 12


# make the sample data
x = torch.unsqueeze(torch.linspace(-1,1,1000),dim=1);print(x.size())
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()));print(y.size())

# plot dataset

plt.scatter(x.numpy(),y.numpy())
plt.show()

# put dataset into torch dataset
torch_dataset = Data.TensorDataset(x,y)
loader= Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)


# default network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden = nn.Linear(1,20)   # hidden layer
        self.predict = nn.Linear(20,1)  # output layer
        
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
    
if __name__ == '__main__':
    
    # differents net
    net_SGD         = Net()
    net_Momentum    = Net()
    net_RMSprop     = Net()
    net_Adam        = Net()
    nets = [net_SGD,net_Momentum, net_RMSprop, net_Adam]
    
    # differents optimizer
    opt_SGD         = optim.SGD(net_SGD.parameters(),lr=LR)
    opt_Mometum     = optim.SGD(net_SGD.parameters(),lr=LR,momentum=0.8)
    opt_RMSprop     = optim.RMSprop(net_RMSprop.parameters(),lr=LR, alpha=0.9)
    opt_Adam        = optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
    optimzers = [opt_SGD, opt_Mometum, opt_RMSprop, opt_Adam]
    
    loss_func = nn.MSELoss()
    losses_his = [[],[],[],[]]
    
    
    for epoch in range(EPOCH):
        print('Epoch: ',epoch)
        
        for batch_idx, (x,y) in enumerate(loader):
            # zip lib 는 각기 다른 배열을 하나의 튜플 형식으로 만들어줌.
            for net,opt,l_his in zip(nets,optimzers,losses_his):
                output = net(x)
                loss = loss_func(output,y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                # l_his.append(loss.data.numpy())
                l_his.append(loss.item()) # loss recoder
                
                
    labels = ['SGD','Momentum','RMSprop','Adam'] 
    for i , l_his in enumerate(losses_his):
        plt.plot(l_his,label=labels[i])
    plt.figure(1, figsize=(20,9))
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim((0,0.3)) # y 축 값의 범위를 지정해주는 함수 
    plt.show()
    
    
    
    
    
    
    
    
    
    
    