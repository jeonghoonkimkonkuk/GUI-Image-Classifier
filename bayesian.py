# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:50:05 2020

@author: amsa9
"""
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np

classes=['buildings','forest','mountain','sea']

priors=[]
covariances=[]
means=[]

# === likelihood 계산 함수 ===
def likelihood(x,prior,mean,cov):
    return -0.5*np.linalg.multi_dot([np.transpose(x-mean),np.linalg.inv(cov),(x-mean)])-0.5*np.log(np.linalg.det(cov))+np.log(prior)

datadir='./glcm_lawsEnergy-15features'
X_train=np.load(datadir+'/X_train.npy')
y_train=np.load(datadir+'/y_train.npy')
X_test=np.load(datadir+'/X_test.npy')
y_test=np.load(datadir+'/y_test.npy')

print('X_train : ',X_train.shape)
print('y_train : ',y_train.shape)
print('X_test  : ',X_test.shape)
print('y_test  : ',y_test.shape)
# === Bayesian classifier ===
for i in range(len(classes)): # 각 클래스 마다
    X = X_train[y_train==i] # i번째 클래스 데이터를 x에 저장
    priors.append((len(X)/len(X_train))) # priors에 사전확률 저장
    means.append(np.mean(X,axis=0)) # means에 평균값 저장
    covariances.append(np.cov(np.transpose(X),bias=True)) # covariances에 공분산 값 저장

# === confusion matrix 시각화 ===
def plot_confusion_matrix(cm, target_names=None, labels=True):
    accuracy = np.trace(cm)/float(np.sum(cm))
    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(6,4))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.colorbar()
    thresh = cm.max()/2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
        
    if labels:
        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,'{:,}'.format(cm[i,j]),
                     horizontalalignment='center',
                     color='white' if cm[i,j]>thresh else 'black')
            
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    
y_pred=[] # 에측 데이터를 저장 할 list
for i in range(len(X_test)): # 각 테스트 데이터에 대해
    likelihoods=[] # 모든 클래스에 대한 likelihood를 저장할 list
    for j in range(len(classes)):
        likelihoods.append(likelihood(X_test[i],priors[j],means[j],covariances[j])) # 모든 클래스의 likelihood 저장
    y_pred.append(likelihoods.index(max(likelihoods))) # 그 중 likelihood가 제일 큰 값을 정답으로 y_pred에 추가
acc = accuracy_score(y_test,y_pred) # test라벨과 예측라벨을 비교하여 정확도 계산
print('accuracy: ',acc)

plot_confusion_matrix(confusion_matrix(y_test,y_pred),target_names=classes) # confusion matrix 시각화

priors=np.array(priors)
covariances=np.array(covariances)
means=np.array(means)

