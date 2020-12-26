from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from scipy import signal as sg
import itertools
import numpy as np
import cv2
import os

# === laws texture 계산 함수 ===
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
            

    

# === 이미지 패치에서 특징 추출 ===
train_dir = './archive/seg_train/seg_train' # train data 경로
test_dir = './archive/seg_test/seg_test' # test data 경로
#classes=['brick','grass','ground'] #클래스 이름
classes=['buildings','forest','mountain','sea']


X_train=[] # train 데이터를 저장 할 list
y_train=[] # train 라벨을 저장 할 list

#PATCH_SIZE = 30 # 이미지 패치 사이즈
#PATCH_NUMBER = 5 # 이미지 패치 갯수
np.random.seed(1234)

for idx, texture_name in enumerate(classes): # 각 class 마다
    image_dir = os.path.join(train_dir,texture_name) # class image가 있는 경로
    i=0
    for image_name in os.listdir(image_dir): # 경로에 있는 모든 이미지에 대해
        if i%10==0:
            print('train - %s : %d'%(texture_name,i))
        i+=1
        
        
        """
        image = cv2.imread(os.path.join(image_dir,image_name)) # 이미지 불러오기
        image_s = cv2.resize(image,(100,100),interpolation=cv2.INTER_LINEAR) # 이미지를 100X100으로 축소
        
        for _ in range(0,PATCH_NUMBER): # 이미지에서 random하게 10개 패치 자르기
            rand_i = np.random.randint(100-PATCH_SIZE) # random하게 자를 위치 선정
            rand_j = np.random.randint(100-PATCH_SIZE)
            
            patch = image_s[rand_i:rand_i+PATCH_SIZE,rand_j:rand_j+PATCH_SIZE] # 이미지 패치 자르기
            patch_gray = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY) # 이미지를 흑백으로 변환
            glcm = greycomatrix(patch_gray,distances=[1],angles=[0],levels=256,
                                symmetric=False,normed=True) # GLCM co-occurence 계산
            X_train.append([greycoprops(glcm,'dissimilarity')[0][0],
                            greycoprops(glcm,'correlation')[0][0]]+ # GLCM dissimilarity, correlation 특징 추가
                           laws_texutre(patch_gray)) # laws texture 특징 추가 (9차원)
            y_train.append(idx) # 라벨 추가
        
        """
        
        image = cv2.imread(os.path.join(image_dir,image_name)) #흑백으로 불러오기
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        glcm = greycomatrix(image_gray,distances=[1],angles=[0],levels=256,
                            symmetric=False,normed=True) # GLCM co-occurence 계산
        X_train.append([greycoprops(glcm,'dissimilarity')[0][0],
                       greycoprops(glcm,'correlation')[0][0],
                       greycoprops(glcm,'contrast')[0][0],
                       greycoprops(glcm,'homogeneity')[0][0],
                       greycoprops(glcm,'ASM')[0][0],
                       greycoprops(glcm,'energy')[0][0]]+ # GLCM dissimilarity, correlation 특징 추가
                      laws_texutre(image_gray)) # laws texture 특징 추가 (9차원)
        y_train.append(idx) # 라벨 추가
            
X_train=np.array(X_train) # list를 numpy array로 변경
y_train=np.array(y_train)

np.save('X_train',X_train) #train데이터를 파일로 저장
np.save('y_train',y_train)

        
X_test=[] # test 데이터를 저장 할 list
y_test=[] # test 라벨을 저장 할 list


for idx, texture_name in enumerate(classes): # 각 class 마다
    image_dir = os.path.join(test_dir,texture_name) # class image가 있는 경로
    i=0
    for image_name in os.listdir(image_dir): # 경로에 있는 모든 이미지에 대해 
        if i%10==0:
            print('test - %s : %d'%(texture_name,i))
        i+=1
        image = cv2.imread(os.path.join(image_dir,image_name)) #흑백으로 불러오기
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        glcm = greycomatrix(image_gray,distances=[1],angles=[0],levels=256,
                            symmetric=False,normed=True) # GLCM co-occurence 계산
        X_test.append([greycoprops(glcm,'dissimilarity')[0][0],
                       greycoprops(glcm,'correlation')[0][0],
                       greycoprops(glcm,'contrast')[0][0],
                       greycoprops(glcm,'homogeneity')[0][0],
                       greycoprops(glcm,'ASM')[0][0],
                       greycoprops(glcm,'energy')[0][0]]+ # GLCM dissimilarity, correlation 특징 추가
                      laws_texutre(image_gray)) # laws texture 특징 추가 (9차원)
        y_test.append(idx) # 라벨 추가
            
X_test=np.array(X_test)
y_test=np.array(y_test)

np.save('X_test',X_test) #test데이터를 파일로 저장
np.save('y_test',y_test)