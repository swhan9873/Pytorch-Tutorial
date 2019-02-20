# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:08:54 2019

@author: Wook
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# replace following class code with an easy sequential network


x = torch.randn([20,1])

print(x,x.size(),x.shape)
class Net1(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net1,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden) # hidden layer
        self.predict = nn.Linear(n_hidden,n_output) # output layer
        
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
    
    
model1=Net1(1,10,1)
output1 = model1(x)
print(output1)
print(model1 )

"""
Net1 (
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=1, bias=True)
)

"""


class Net2(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net2,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )
    def forward(self,x):
        x = self.fc(x)
        return x
"""
Net2 (
  (fc): Sequential(
    (0): Linear(in_features=1, out_features=10, bias=True)
    (1): ReLU()
    (2): Linear(in_features=10, out_features=1, bias=True)
  )
)
"""
    
model2 = Net2(1,10,1)
output2 = model2(x)
print(model2)
print(output2)



