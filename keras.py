import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    X_train=np.load('./glcm_lawsEnergy/X_train.npy')
    y_train=np.load('./glcm_lawsEnergy/y_train.npy')
    
    y_train = to_categorical(y_train,4)
    
    X_test=np.load('./glcm_lawsEnergy/X_test.npy')
    y_test=np.load('./glcm_lawsEnergy/y_test.npy')
    
    y_test = to_categorical(y_test,4)
    
    model = Sequential([
        Input(11),
        Dense(8,activation='relu'),
        Dense(8,activation='relu'),
        Dense(8,activation='relu'),
        Dense(4,activation='softmax')
        ])
    
    model.summary()
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    history = model.fit(X_train,y_train,batch_size=100,epochs=100,verbose=2,validation_data=(X_test,y_test))
    
    
    
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],label='train loss')
    plt.plot(history.history['val_loss'],label='test loss')
    plt.legend()
    
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'],label='train acc')
    plt.plot(history.history['val_accuracy'],label = 'test acc')
    plt.legend()
    
    
    plt.show()
