# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:54:19 2019

@author: Wook
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt


torch.manual_seed(1)

BATCH_SIZE = 5


x = torch.linspace(1,10,10) # 1~10 까지 10개로 나누기
y = torch.linspace(10,1,10) # 10~1 까지 10개로 나누기 


torch_dataset = Data.TensorDataset(x,y)

loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
)

def show_batch():
    for epoch in range(3): # train entire dataset 3 times
        for step , (batch_x,batch_y) in enumerate(loader): # for each trianing step
            # train your data...
            print('Epoch: ', epoch,'| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())

if __name__ == '__main__':
    show_batch()