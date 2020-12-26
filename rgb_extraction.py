
import numpy as np
import cv2
import os

def load_data():
    train_dir = './archive/seg_train/seg_train' # train data 경로
    test_dir = './archive/seg_test/seg_test' # test data 경로
    #classes=['brick','grass','ground'] #클래스 이름
    classes=['buildings','forest','mountain','sea']
    
    X_train=[] # train 데이터를 저장 할 list
    y_train=[] # train 라벨을 저장 할 list
    
    #PATCH_SIZE = 32
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
            image = cv2.imread(os.path.join(image_dir,image_name))
            image_s = cv2.resize(image,(100,100),interpolation=cv2.INTER_LINEAR) # 이미지를 100X100으로 축소
            if(image_s.size!=30000):
                print(os.path.join(image_dir,image_name))
            X_train.append(image_s)
            y_train.append(idx) # 라벨 추가
    
    X_train = np.array(X_train)/128 -1        
    y_train = np.array(y_train)
    
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
            image = cv2.imread(os.path.join(image_dir,image_name))
            image_s = cv2.resize(image,(100,100),interpolation=cv2.INTER_LINEAR) # 이미지를 100X100으로 축소
    
            X_test.append(image_s)
            y_test.append(idx) # 라벨 추가
    
    X_test = np.array(X_test)/128 -1        
    y_test = np.array(y_test)
    
    np.save('X_test',X_test) #test데이터를 파일로 저장
    np.save('y_test',y_test)
                
    return X_train,y_train,X_test,y_test

load_data()