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

#ACCURACY HESPLAMA FONKSİYONU
class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives/1000

    def reset_states(self):
        self.true_positives.assign(0.)


#MODEL OLUŞTURULUYOR

model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(28,28,3)))


model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
model.summary()

#SGD GRADYAN YÖNTEMİ UYGULANIYOR
opt = SGD(lr=0.01, momentum=0.99, decay=0.01)
#opt = Adam(lr=0.000001)

#METRİCS KISMINDA KENDİ FONKSİYONUMU KULLANARAK HESAPLADIM ACCURACY'İ
model.compile(optimizer = opt , loss='sparse_categorical_crossentropy', metrics = [Accuracy()])

history = model.fit(x_train,y_train, epochs = 20 , validation_data = (x_val, y_val))

#BURADAKİ ACCURACYLER KÜTÜPHANEDEN GELİYOR ÇÜNKÜ KENDİ EKLEDİĞİM ACCURACY FONKSİYONU İLE ÇİZİM YAPMIYORDU

#SONUÇLARIN ÇİZİMİ
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(20)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#MODELİN KAYDEDİLMESİ
model.save("test_model.h5")














