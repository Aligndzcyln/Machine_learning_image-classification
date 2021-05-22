#GEREKLİ İMPORTLAR
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import classification_report,confusion_matrix
import cv2
import os
import numpy as np
#LABELLAR TANITILIYOR
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',]
img_size = 28
#VERİ ÇEKME İŞLEMLERİ
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1]
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train = get_data('./mnist-train')
val = get_data('./mnist-test')

#DATA İŞLEME
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# VERİYİ NORMALİZE ETME
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

#MODELİ İMPORT ETME
from tensorflow.keras.models import load_model
model = load_model("test_model.h5")

#GEREKLİ DEĞİŞKENLER
accuracy = []
accuracy_test = []

s0=0
s1=0
s2=0
s3=0
s4=0
s5=0
s6=0
s7=0
s8=0
s9=0
j=0

#TEST VE TRAİN DATALARININ TAHMİNİ
x_pred = model.predict(x_train)
y_pred = model.predict(x_val)

#GELEN TRAİN TAHMİN MATRİSİNİN VERİLERİNİN DÜZENLENMESİ
for i in range(len(x_pred)):#satır
    for j in range(len(x_pred[i])):#sütun
        if j==0 and i<200 : 
            accuracy.append(x_pred[i][j]) 
        if j==1 and 199<i<400 : 
            accuracy.append(x_pred[i][j])
        if j==2 and 399<i<600 : 
            accuracy.append(x_pred[i][j])
        if j==3 and 599<i<800 : 
            accuracy.append(x_pred[i][j])
        if j==4 and 799<i<1000 : 
            accuracy.append(x_pred[i][j])
        if j==5 and 999<i<1200 : 
            accuracy.append(x_pred[i][j])
        if j==6 and 1199<i<1400 : 
            accuracy.append(x_pred[i][j])
        if j==7 and 1399<i<1600 : 
            accuracy.append(x_pred[i][j])
        if j==8 and 1599<i<1800 : 
            accuracy.append(x_pred[i][j])
        if j==9 and 1799<i<2000 : 
            accuracy.append(x_pred[i][j])

for i in range(len(accuracy)):
    if i<200 : 
        s0 += accuracy[i]
    if 199<i<400 : 
         s1 += accuracy[i]
    if 399<i<600 : 
         s2 += accuracy[i]
    if 599<i<800 : 
         s3 += accuracy[i]
    if 799<i<1000 : 
         s4 += accuracy[i]
    if 999<i<1200 : 
        s5 += accuracy[i]
    if 1199<i<1400 : 
        s6 += accuracy[i]
    if 1399<i<1600 : 
        s7 += accuracy[i]
    if 1599<i<1800 : 
        s8 += accuracy[i]
    if 1799<i<2000 : 
        s9 += accuracy[i]
        
#ORTALAMA ACCURACY DEĞERLERİNİN TRAİN DATASI İÇİN SONUÇLARININ EKLENMESİ
acc_mean_train = []
acc_mean_train.append(s0/200)
acc_mean_train.append(s1/200)
acc_mean_train.append(s2/200)
acc_mean_train.append(s3/200)
acc_mean_train.append(s4/200)
acc_mean_train.append(s5/200)
acc_mean_train.append(s6/200)
acc_mean_train.append(s7/200)
acc_mean_train.append(s8/200)
acc_mean_train.append(s9/200)

#GELEN TEST TAHMİN MATRİSİNİN VERİLERİNİN DÜZENLENMESİ
for i in range(len(y_pred)):#satır
    for j in range(len(y_pred[i])):#sütun
        if j==0 and i<100 : 
            accuracy_test.append(y_pred[i][j]) 
        if j==1 and 99<i<200 : 
            accuracy_test.append(y_pred[i][j])
        if j==2 and 199<i<300 : 
            accuracy_test.append(y_pred[i][j])
        if j==3 and 299<i<400 : 
            accuracy_test.append(y_pred[i][j])
        if j==4 and 399<i<500 : 
            accuracy_test.append(y_pred[i][j])
        if j==5 and 499<i<600 : 
            accuracy_test.append(y_pred[i][j])
        if j==6 and 599<i<700 : 
            accuracy_test.append(y_pred[i][j])
        if j==7 and 699<i<800 : 
            accuracy_test.append(y_pred[i][j])
        if j==8 and 799<i<900 : 
            accuracy_test.append(y_pred[i][j])
        if j==9 and 899<i<1000 : 
            accuracy_test.append(y_pred[i][j])
            
s0t=0
s1t=0
s2t=0
s3t=0
s4t=0
s5t=0
s6t=0
s7t=0
s8t=0
s9t=0
            
for i in range(len(accuracy_test)):
    if i<100 : 
        s0t += accuracy[i]
    if 99<i<200 : 
         s1t += accuracy[i]
    if 199<i<300 : 
         s2t += accuracy[i]
    if 299<i<400 : 
         s3t += accuracy[i]
    if 399<i<500 : 
         s4t += accuracy[i]
    if 499<i<600 : 
        s5t += accuracy[i]
    if 599<i<700 : 
        s6t += accuracy[i]
    if 699<i<800 : 
        s7t += accuracy[i]
    if 799<i<900 : 
        s8t += accuracy[i]
    if 899<i<1000 : 
        s9t += accuracy[i]

#ORTALAMA ACCURACY DEĞERLERİNİN TEST DATASI İÇİN SONUÇLARININ EKLENMESİ
acc_mean_test = []

acc_mean_test.append(s0t/100)
acc_mean_test.append(s1t/100)
acc_mean_test.append(s2t/100)
acc_mean_test.append(s3t/100)
acc_mean_test.append(s4t/100)
acc_mean_test.append(s5t/100)
acc_mean_test.append(s6t/100)
acc_mean_test.append(s7t/100)
acc_mean_test.append(s8t/100)
acc_mean_test.append(s9t/100)

#TEST VE TRAİN DATALARININ MODELE UYGULANINCA ACCURACY ORTALAMA DEĞERLERİ
print(acc_mean_train)
print("\n\n")
print(acc_mean_test)







