# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:38:47 2019

@author: Wook
"""
import os

import torch
import torch.nn as nn

import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
# device configuration
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_GPU else "cpu");print(device)



# Initialize hyper-parameters
EPOCH = 1
BATCH_SIZE = 50
LEARNING_RATE = 0.001
DOWNLOAD_MNIST = False


# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist'):
    DOWNLOAD_MNIST = True
    
    
train_data = MNIST(
        root='./mnist/',
        train=True,                                     # this is training data
        transform=torchvision.transforms.ToTensor(),    # Converts a PLT.image or numpy.ndarry to
                                                        # torch.FloatTensor of shape (C x H x W) and normailize in the range([0.0,1.0])
        download=DOWNLOAD_MNIST
)

# plot one example
print(train_data.train_data.size())     # [60000, 28, 28]
print(train_data.train_labels.size())   # [60000]
plt.imshow(train_data.train_data[2],cmap='gray')
plt.title('%i'% train_data.train_labels[2])
plt.show()

# Data loader for easy mini-bacth return in training, the image batch shape will be (50,1,28,28)
train_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
print(train_loader.__len__()) # 60000 / bacth size = 1200 

# pick 2000 samples to speed up testing
test_data = MNIST(root='./mnist',train=False)

# shape from (2000, 28, 28) to (2000, 1, 28, 28)
test_x = torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.test_labels[:2000] # (2000, 1, 28, 28)
print(test_data.__len__(),test_x.size())




class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(             # input shape (1, 28 ,28)
                nn.Conv2d(
                    in_channels = 1,            # input height
                    out_channels = 16,          # n_filters
                    kernel_size = 5,            # filter(kernel) size
                    stride = 1,                 # filter movement/step
                    padding =2,                 # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride
                ),                              # output shape (16 ,28, 28)
                nn.ReLU(),                      # activation
                nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(             # input shape (16, 14, 14)
                nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
                nn.ReLU(),                      # activation
                nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.fc = nn.Linear(32*7*7 ,10)         # fully connected layer, output 10 class
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)                # flatten the output of conv2 to | (batch_size, 32 * 7 * 7)
        output = self.fc(x)
        return output, x                        # return x for visualization
                

model = CNN().to(device)

optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE) # optimize all CNN paramter
loss_func = nn.CrossEntropyLoss()                           # the target label is not one-hotted


# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK=True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X,Y = lowDWeights[:,0], lowDWeights[:,1]
    for x,y,s in zip(X,Y,labels):
        c = cm.rainbow(int(255*s/9));plt.text(x,y,s,backgroundcolor=c,fontsize=9)
        plt.xlim(X.min(),X.max()); plt.ylim(Y.min(),Y.max());plt.title('Visualize last layer');plt.show();plt.pause(0.01)
        
        
        
plt.ion()
# training and testing
for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader): # gives batch data, normalize x when iterate train_loader
        
        output,_ = model(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 2000 == 0:
            test_output,last_layer = model(test_x)
            pred_y = torch.max(test_output,1)[1].data.numpy()
            _,predicted_y = torch.max(test_output.data,1)
            print('1',predicted_y)
            print('2',pred_y)
            
            # Same code to calculate accuracy * acc2, acc3 popular *
            acc1 = ((pred_y == test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
            acc2 = predicted_y.eq(test_y.data).sum().item() / test_y.size(0)
            acc3 = (predicted_y == test_y.data).sum().item() / test_y.size(0)
            
            print(acc1,acc2, acc3)
#            # test_y.size(0) = 2000
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % acc3*100)
            
            if HAS_SK:
                # Visualization of trained flatten later (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
            
plt.ioff()

# print 10 predictions from test data
test_output,_ = model(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')


            
            
            
            
            
            
            
            
