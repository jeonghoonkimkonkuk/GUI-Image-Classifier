import torch
import torch.nn as nn
import sys
from PyQt5.QtWidgets import QLabel,QFileDialog,QVBoxLayout,QPushButton,QApplication,QWidget
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from scipy import signal as sg
from skimage.feature import greycomatrix, greycoprops
from torchsummary import summary
from tensorflow.keras.utils import to_categorical

classes=['buildings','forest','mountain','sea']

def ensemble(p1,p2,p3,p4): #bayesian, mlp_gl,mlp_rgb,cnn
    _,a1 = torch.max(p1,1)
    _,a2 = torch.max(p2,1)
    _,a3 = torch.max(p3,1)
    _,a4 = torch.max(p4,1)
    
    c1 = to_categorical(a1,4)
    c2 = to_categorical(a2,4)
    c3 = to_categorical(a3,4)
    c4 = to_categorical(a4,4)
    
    c=c1*76+c2*79+c3*72+c4*82
    c=torch.Tensor(c)
    return c
    

def select_class(outputs):
    _,predicted = torch.max(outputs,1)
    return classes[predicted]


def likelihood(x,prior,mean,cov):
    return -0.5*np.linalg.multi_dot([np.transpose(x-mean),np.linalg.inv(cov),(x-mean)])-0.5*np.log(np.linalg.det(cov))+np.log(prior)

def laws_texutre(gray_image):
    (rows, cols) = gray_image.shape[:2] # smoothing filter
    smooth_kernel = (1/25)*np.ones((5,5)) # 흑백이미지 smoothing
    gray_smooth =  sg.convolve(gray_image, smooth_kernel,"same") 

    gray_processed = np.abs(gray_image-gray_smooth) # 원본이미지에서 smoothing된 이미지 빼기

    
    filter_vectors = np.array([[-1, 4, 6, 4, 1], # L5
                               [-1, -2, 0, 2, 1], # E5
                               [-1, 0, 2, 0, 1], # R5
                               [1, -4, 6, -4, 1]]) # 16(4X4)개 filter를 저장할 filters
    filters=[]
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(filter_vectors[i][:].reshape(5,1), # 매트릭스 곱하기 연산을 통해 filter값 계산
                                     filter_vectors[j][:].reshape(1,5)))
            
    conv_maps = np.zeros((rows,cols,16)) # 계산된 convolution 결과를 저장할 conv_maps
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed,filters[i],'same') # 전처리된 이미지에 16개 필터 적용
    
    # === 9+1개 중요한 texture map 계산 ===
    texture_maps = list()
    texture_maps.append((conv_maps[:, :, 1]+conv_maps[:, :, 4])//2) # L5E5 / E5L5
    texture_maps.append((conv_maps[:, :, 2]+conv_maps[:, :, 8])//2) # L5S5 / S5L5
    texture_maps.append((conv_maps[:, :, 3]+conv_maps[:, :, 12])//2) # L5R5 / R5L5
    texture_maps.append((conv_maps[:, :, 7]+conv_maps[:, :, 13])//2) # E5R5 / R5E5
    texture_maps.append((conv_maps[:, :, 6]+conv_maps[:, :, 9])//2) # E5S5 / S5E5
    texture_maps.append((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2) # S5R5 / R5S5
    texture_maps.append(conv_maps[:, :, 10]) # S5S5
    texture_maps.append(conv_maps[:, :, 5]) # E5E5
    texture_maps.append(conv_maps[:, :, 15]) # R5R5
    texture_maps.append(conv_maps[:, :, 0]) # L5L5 (use to norm TEM)
            
    # === Law's texture energy 계산 ===
    TEM = list()
    for i in range(9):
        TEM.append(np.abs(texture_maps[i]).sum()/ # TEM계산 및 L5L5 값으로 정규화
                   np.abs(texture_maps[9]).sum()) 
            
    return TEM # 9차원의 TEM feature 추출: list
def bayesian(image):
    feature=[]
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #흑백
    glcm = greycomatrix(image_gray,distances=[1],angles=[0],levels=256,
                        symmetric=False,normed=True) # GLCM co-occurence 계산
    feature.append([greycoprops(glcm,'dissimilarity')[0][0],
                   greycoprops(glcm,'correlation')[0][0]]+ # GLCM dissimilarity, correlation 특징 추가
                  laws_texutre(image_gray)) # laws texture 특징 추가 (9차원)
    
    feature=np.array(feature).reshape(11)
    covariances = np.load('./bayesian/covariances.npy')
    means = np.load('./bayesian/means.npy')
    priors = np.load('./bayesian/priors.npy')
    
    
    likelihoods=[] # 모든 클래스에 대한 likelihood를 저장할 list
    for j in range(len(classes)):
        likelihoods.append(likelihood(feature,priors[j],means[j],covariances[j])) 
    
    likelihoods = np.array([likelihoods])
    likelihoods = torch.Tensor(likelihoods)
    #predict = likelihoods.index(max(likelihoods))
    return likelihoods

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

def mlp_gl(image):
    net = torch.load('./mlp_gl.pt')
    net.eval()
    
    feature=[]
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #흑백
    glcm = greycomatrix(image_gray,distances=[1],angles=[0],levels=256,
                        symmetric=False,normed=True) # GLCM co-occurence 계산
    feature.append([greycoprops(glcm,'dissimilarity')[0][0],
                   greycoprops(glcm,'correlation')[0][0]]+ # GLCM dissimilarity, correlation 특징 추가
                  laws_texutre(image_gray)) # laws texture 특징 추가 (9차원)
    
    feature=np.array(feature)
    feature=torch.Tensor(feature)
    
    outputs=net(feature)
    return outputs


class MLP_RGB(nn.Module):
    def __init__(self,input_dim,hidden_dim1,hidden_dim2,output_dim):
        super(MLP_RGB,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1,hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1,hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2,hidden_dim2)
        self.fc5 = nn.Linear(hidden_dim2,output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = torch.flatten(x,1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc5(out)
        
        return out
    
def mlp_rgb(image):
    net = torch.load('./mlp_rgb.pt')
    net.eval()
    
    feature = cv2.resize(image,(100,100),interpolation=cv2.INTER_LINEAR)
    
    feature = np.array([feature/128-1])
    
    feature=torch.Tensor(feature)

    outputs=net(feature)

    return outputs

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
    
def cnn(image):
    net = torch.load('./cnn74.pt')
    net.eval()
    
    feature = cv2.resize(image,(100,100),interpolation=cv2.INTER_LINEAR)
    
    feature = np.array([feature/128-1])
    feature=np.swapaxes(feature,1,3)
    feature=torch.Tensor(feature)
    
    outputs=net(feature)

    return outputs

class MyApp(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setGeometry(800,200,300,300)
        self.setWindowTitle('201811172 kjh')
        
        self.pushButton = QPushButton('File Open')
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.label = QLabel()
        self.label2 = QLabel()
        self.label3 = QLabel()
        self.label4 = QLabel()
        self.label5 = QLabel()
        self.label6 = QLabel()
        
        layout = QVBoxLayout()
        layout.addWidget(self.pushButton)
        layout.addWidget(self.label)
        layout.addWidget(self.label2)
        layout.addWidget(self.label3)
        layout.addWidget(self.label4)
        layout.addWidget(self.label5)
        layout.addWidget(self.label6)
        
        self.setLayout(layout)
        self.show()
        
    def pushButtonClicked(self):
        fname,_ = QFileDialog.getOpenFileName(self)
        pixmap = QPixmap(fname)
        
        
        ff = np.fromfile(fname,np.uint8)
        image = cv2.imdecode(ff,cv2.IMREAD_UNCHANGED)
        p1=bayesian(image)
        p2=mlp_gl(image)
        p3=mlp_rgb(image)
        p4=cnn(image)
        predict1 = select_class(p1)
        predict2 = select_class(p2)
        predict3 = select_class(p3)
        predict4 = select_class(p4)
        predict5 = select_class(ensemble(p1,p2,p3,p4))
        
        self.label.setPixmap(QPixmap(pixmap))
        self.label2.setText('bayesian : '+predict1)
        self.label3.setText('mlp-gl : '+predict2)
        self.label4.setText('mlp-rgb : '+predict3)
        self.label5.setText('cnn : '+predict4)
        self.label6.setText('* ensemble : '+predict5)
        
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    sys.exit(app.exec_())
