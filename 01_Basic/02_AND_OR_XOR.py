# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:47:49 2019

@author: Wook
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def main():
    
    # OR
    train_x = np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]]
    )
    train_y = np.array([[0],[1],[1],[1]])
    
    # Convert the numpy array to a torch Tensor
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.FloatTensor)

    # Create the Linear model 
    net = nn.Linear(2,1)

    loss_func = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),lr=0.1)
    
    num_epochs = 50
    
    # Train
    for epoch in range(num_epochs):
        for x,y in zip(train_x,train_y):
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(x)
            loss = loss_func(output,y)
            loss.backward()
            optimizer.step()
            
        print('Epoch [{}/{}], Loss : {:.4f}'.format(epoch+1,num_epochs,loss.item()))

    print('\n')
    print('Finish result')
    
    test_x = torch.randn([30,2]) # 30 개의 랜덤한 데이터를 가지고 확인해보자
    epoch = test_x.size(0)
    threshold = 0.3
    
    
    y = []
    for i in range(epoch):
        
        output = net(test_x[i])
        # 특정 값보다 크면 1로 설정
        pred = 1 if output.item() > threshold else 0
        y.append(pred)
        

    
#    predicted = net(test_x).detach().numpy()
#    print(predicted)
#    
    plt.scatter(test_x.data.numpy()[:,0], test_x.data.numpy()[:,1],c=y, s=50, lw=1, cmap='RdYlGn')
    plt.legend()
    plt.show()
        

    
if __name__ =='__main__':
    main()

