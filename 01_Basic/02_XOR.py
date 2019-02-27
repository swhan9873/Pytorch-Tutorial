# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:45:08 2019

@author: Wook
"""

import torch
import torch.nn as nn
import numpy as np

def main():
    
    x = np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]])
    y = np.array([0,0,0,1])

    x = torch.from_numpy(x)
    print(x,x.shape)
    
    y = torch.from_numpy(y)
    print(y, y.shape)
    
    
    net = nn.Linear(2,)
    
if __name__ =='__main__':
    main()

