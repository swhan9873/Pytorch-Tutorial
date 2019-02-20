# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:40:12 2019

@author: Wook

"""

import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()           # 다시 numpy 로 복귀 하는 코드

print(
      '\n numpy array:',     np_data,        # [[0 1 2],[3 4 5]]
      '\n torch tensord:',   torch_data,     # tensor([[0 1 2],[3 4 5]],dtype=torch.int32)
      '\n tensor to array:', tensor2array,   # [[0 1 2],[3 4 5]]
)



# abs

data = [-1,-2,1,2]
tensor = torch.FloatTensor(data)

print(
      '\n abs',
      '\n numpy:',np.abs(data),             # [1 2 1 2]
      '\n torch:',torch.abs(tensor)         # tensor([1.,2.,1.,2.])
)

# sin
print(
      '\n sin',
      '\n numpy:',np.sin(data),             # [-0.84147098 -0.90929743 0.84147098 0.90929743]
      '\n torch:',torch.sin(tensor)         # tensor([-0.8415, -0.9093 ,0.8415 , 0.9093])
)

# mean
print(
      '\n sin',
      '\n numpy:',np.mean(data),             # 0.0
      '\n torch:',torch.mean(tensor)         # tensor(0.)
)

# matrix multipication
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)

# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # tensor([[7, 10], [15, 22]])
)