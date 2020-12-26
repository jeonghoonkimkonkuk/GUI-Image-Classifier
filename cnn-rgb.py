import numpy as np

datadir='./rgb/'


X_train=np.load('./rgb/X_train.npy',allow_pickle=True)
y_train=np.load('./rgb/y_train.npy',allow_pickle=True)
X_test=np.load('./rgb/X_test.npy',allow_pickle=True)
y_test=np.load('./rgb/y_test.npy',allow_pickle=True)


X_train=np.swapaxes(X_train,1,3)
X_test=np.swapaxes(X_test,1,3)

print('X_train : ',X_train.shape)
print('y_train : ',y_train.shape)
print('X_test  : ',X_test.shape)
print('y_test  : ',y_test.shape)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsummary import summary

class Dataset(Dataset):
    def __init__(self,images,labels):
        self.images=images
        self.labels=labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]
        sample = (image,label)
        
        return sample

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=10,out_channels=3,kernel_size=3)
        self.pool2 = nn.AdaptiveMaxPool2d(output_size=3)
        self.fc1 = nn.Linear(27,4)
        self.relu = nn.ReLU6()
    
    def forward(self,x):
        
        out = self.conv1(x) # 10X10X30
        out = self.relu(out) # 10X30X30
        out = self.conv2(out) # 10X28X28
        out = self.relu(out) # 10X28X28
        out = self.pool1(out) # 10X14X14
        out = self.conv3(out) # 10X12X12
        out = self.relu(out) # 10X12X12
        out = self.conv4(out) # 3X10X10
        out = self.relu(out) # 3X10X10
        out = self.pool2(out) # 3X3X3
        out = torch.flatten(out,1) # 27 
        out = self.fc1(out) # 4
        
        return out
    
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#X_train,y_train,X_test,y_test = load_data()
        
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
learning_rate = 0.001
n_epoch = 500

Train_data = Dataset(images=X_train,labels=y_train)
Test_data = Dataset(images = X_test, labels=y_test)

Trainloader = DataLoader(Train_data, batch_size=batch_size,shuffle=True)
Testloader = DataLoader(Test_data,batch_size=batch_size)

net=CNN()
summary(net,(3,100,100))

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
            torch.save(net,datadir+'cnn/%d-%1.2f_%1.2f.pt'%(epoch+1,train_acc,test_acc))
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

plt.savefig(datadir+'cnn/graph.png')
