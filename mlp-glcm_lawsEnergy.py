# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 01:15:27 2020

@author: amsa9
"""
import numpy as np


datadir='./glcm_lawsEnergy-11features/'
feature_num=11

X_train=np.load(datadir+'X_train.npy')
y_train=np.load(datadir+'y_train.npy')
X_test=np.load(datadir+'X_test.npy')
y_test=np.load(datadir+'y_test.npy')

print('X_train : ',X_train.shape)
print('y_train : ',y_train.shape)
print('X_test  : ',X_test.shape)
print('y_test  : ',y_test.shape)

classes=['buildings','forest','mountain','sea']

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsummary import summary

class textureDataset(Dataset):
    def __init__(self,features,labels):
        self.features=features
        self.labels=labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature = self.features[idx]
        label = self.labels[idx]
        sample = (feature,label)
        
        return sample

class MLP_GL(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MLP_GL,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100): 
    formatStr = "{0:." + str(decimals) + "f}" 
    percent = formatStr.format(100 * (iteration / float(total))) 
    filledLength = int(round(barLength * iteration / float(total))) 
    bar = '#' * filledLength + '-' * (barLength - filledLength) 
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)), 
    if iteration == total: 
        sys.stdout.write('\n') 
    sys.stdout.flush()
    
    
batch_size = 1000
learning_rate = 0.01
n_epoch = 1000

Train_data = textureDataset(features=X_train,labels=y_train)
Test_data = textureDataset(features = X_test, labels=y_test)

Trainloader = DataLoader(Train_data, batch_size=batch_size,shuffle=True)
Testloader = DataLoader(Test_data,batch_size=batch_size)

net=MLP_GL(feature_num,8,4)
summary(net,(feature_num,))

optimizer = optim.Adam(net.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss()

max_test_acc=0.0
train_losses=[]
train_accs=[]
test_losses=[]
test_accs=[]

# === 학습 ===
for epoch in range(n_epoch):
    train_loss=0.0
    corrects=0.0
    net.train()
    i=0
    for features,labels in Trainloader:
        printProgress(i,len(Trainloader))
        i+=1
        outputs=net(features.to(torch.float))
        labels = labels.long()
        
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _,predicted = torch.max(outputs,1)
        train_loss+=loss.item()*len(features)
        corrects += torch.sum(predicted==labels.data)


    
    train_loss=train_loss/Train_data.__len__()
    train_acc=corrects/Train_data.__len__()
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # === 테스트 ===
    if(epoch+1)%1==0:
        test_loss=0.0
        corrects=0.0
        net.eval()
        for features,labels in Testloader:
            labels=labels.long()
            features = features.to(torch.float)
            outputs=net(features)
            loss=criterion(outputs,labels)
            _,predicted = torch.max(outputs,1)
            test_loss+=loss.item()*len(features)
            corrects+=torch.sum(predicted==labels.data)
    
        test_loss=test_loss/Test_data.__len__()
        test_acc = corrects/Test_data.__len__()
        
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print('\n[%d, %3d] loss: %.4f Accuracy: %.4f val-loss: %.4f val-Accuracy: %.4f'%
              (epoch+1,n_epoch,train_loss,train_acc,test_loss,test_acc))
        if(test_acc>max_test_acc):
            max_test_acc=test_acc
            torch.save(net,datadir+'mlp/%d-%1.2f_%1.2f.pt'%(epoch+1,train_acc,test_acc))
            print('저장')
        
        
import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(range(len(train_losses)),train_losses,label='train loss')
plt.plot(range(len(test_losses)),test_losses,label='test loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(len(train_accs)),train_accs,label='train acc')
plt.plot(range(len(test_accs)),test_accs,label='test acc')
plt.legend()
plt.show()

plt.savefig(datadir+'mlp/graph')
























        
