# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:08:16 2019

@author: wook
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.resnet import ResNet18,ResNet34
from hparams import HyperParameter as hp

import matplotlib.pyplot as plt
import os
import numpy as np

def train(train_loader, model, loss_func, optimizer, epoch, device):
    
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        
        # data: [128, 1, 32, 32]
        # target:[128] ~~ [5 1 3 2 4 5 0 9 8 4 6 ...] 
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # output: [128,10]
        
        outputs = model(data)
        loss = loss_func(outputs,target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % hp.train.log_interval == 0:
            print('\nTrain epoch: {} [ {}/{} ({:.2f}%) ]\tLoss: {:.3f}'.format(
                    epoch,batch_idx*len(data), len(train_loader.dataset),
                    100.*batch_idx/len(train_loader), loss.item()))
    
    torch.cuda.empty_cache()
    
def test(test_loader,model,loss_func,device):
    
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device),target.to(device)
            outputs = model(data)

            loss = loss_func(outputs,target)
            _,predicted = torch.max(outputs.data,1) # 제일 큰 값의 index 추출.
            
            
            test_loss += loss.item()
            
            correct += predicted.eq(target.data).sum().item()
            
            
        test_loss /= len(test_loader.dataset)
        acc = 100.*correct/len(test_loader.dataset)
        
        print('\nTest: Average Loss: {:.3f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                test_loss,correct,len(test_loader.dataset), 
                100.*correct/len(test_loader.dataset)))
            
    return test_loss,acc
    
if __name__ == '__main__':
    
    print(hp)
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_set = torchvision.datasets.MNIST('../data',train=True,download=False,transform = transform_train)
    train_loader = DataLoader(train_set,batch_size=hp.train.batch_size,shuffle=True,num_workers=0)
    
    print('train_set size:',train_set.__len__())        # 60,000
    print('train_loader size:',train_loader.__len__())  # 469 = 60,000 / 128 (batch_size)
    
    
    test_set = torchvision.datasets.MNIST('../data',train=False,download=False,transform = transform_test)
    test_loader = DataLoader(test_set,batch_size=hp.test.batch_size,shuffle=False,num_workers=0)
    
    
    print('test_set size:',test_set.__len__())        # 10,000
    print('test_loader size:',test_loader.__len__())  # 157= 10000 / 64 (batch_size)
    
    
    # set the model
    if use_gpu:
        model = ResNet18().to(device)
    else:
        mdoel = ResNet18()
    cudnn.benchmark = True
    
    # loss function
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = hp.train.lr,weight_decay = 5e-4)
    
    test_losses = []
    test_acc_list = []
    
    best_acc = 0
    for epoch in range(hp.train.epochs):
        
        print("-"*45);
        
        train(train_loader,model,loss_func,optimizer,epoch,device)
        test_loss,test_acc = test(test_loader,model,loss_func,device)
        
        test_losses.append(test_loss)
        test_acc_list.append(test_acc)
    
        # save the model
        if test_acc > best_acc:
            best_acc = test_acc
            print('obtain the best acc so this model saving...')
            
            state = model.state_dict()
            
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            
            torch.save(state,'./checkpoint/191107.pth')
            
    
    print("finish the training.")
    
    plt.plot(np.arange(hp.train.epochs),test_acc_list)
    plt.xlabel('epoch')
    plt.ylabel('test_acc')
    plt.show()
    
    
    plt.plot(np.arange(hp.train.epochs),test_losses)
    plt.xlabel('epoch')
    plt.ylabel('test_loss')
    plt.show()    
    
    
    
    
    
        
    
    
    
    
    
 