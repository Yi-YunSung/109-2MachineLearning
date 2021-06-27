import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import glob
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt

#讀訓練檔案
train_image_path = 'C:/Users/User/Downloads/train_image'
train_image_filelist = os.listdir(train_image_path)
train_label_path = 'C:/Users/User/Downloads/train_label'
train_label_filelist = os.listdir(train_label_path)

#讀取一層數據
ori_data = sitk.ReadImage(os.path.join(train_image_path, train_image_filelist[1]))
data1 = sitk.GetArrayFromImage(ori_data)#讀取數據陣列
#打印数据name、shape、某一个位置的元素的值（z,y,x）
print(train_image_filelist[1], data1.shape, data1[38,255,255])
plt.imshow(data1[60,:,:]) # 对第85张slice可视化
print(plt.show())

#讀測試檔案
test_image_path = 'C:/Users/User/Downloads/test_image'
test_image_filelist = os.listdir(test_image_path)



#訓練檔案轉換
def load_train_data(path):
    train_image_dir = sorted(train_image_filelist)
    data = []
    for p in tqdm(train_image_dir):
        data_list = sorted(os.listdir(path))
        img_itk = sitk.ReadImage(path + '/'+ data_list[0])
        flair = sitk.GetArrayFromImage(img_itk)
        data.append([flair])
    data = np.asarray(data, dtype=np.float16)
    return data
def load_train_label_data(path):
    train_label_dir = sorted(train_label_filelist)
    gt = []
    for p in tqdm(train_label_dir):
        data_list = sorted(os.listdir(path ))
        img_itk = sitk.ReadImage(path  + '/'+ data_list[0])
        seg = sitk.GetArrayFromImage(img_itk)
        gt.append(seg)
    gt = np.asarray(gt, dtype=np.float16)
    return  gt
data1 = load_train_data(train_image_path)
gt1 = load_train_label_data(train_label_path)
np.save('data1', data1)
np.save('gt1',gt1)

#測次檔案轉換
def load_test_data(path):
    test_image_dir = sorted(test_image_filelist)
    data = []
    for p in tqdm(test_image_dir):
        data_list = sorted(os.listdir(path))
        img_itk = sitk.ReadImage(path + '/'+ data_list[0])
        flair = sitk.GetArrayFromImage(img_itk)
        data.append([flair])
    data = np.asarray(data, dtype=np.float16)
    return data
#data2 = load_test_data(test_image_path)
#np.save('data2', data2)

#load npy
data = np.load('C:/Users/User/PycharmProjects/shopee/data1.npy')
data = np.transpose(data,(0,2,3,4,1))
X_train = data[:,30:60,30:150,30:150,:].reshape([-1,200,200,1])
print(X_train.shape)
data = np.load('C:/Users/User/PycharmProjects/shopee/gt1.npy')
y_train = data[:,30:60,30:150,30:150].reshape([-1,200,200,1])
print(y_train.shape)
data = np.load('C:/Users/User/PycharmProjects/shopee/data2.npy')
data = np.transpose(data,(0,2,3,4,1))
X_test = data[:,30:50,30:85,30:85,:].reshape([-1,200,200,1])
print(X_test.shape)

#正規化
from keras.utils import to_categorical
y_train = to_categorical(y_train)
X_train = (X_train-np.mean(X_train))/np.max(X_train)
X_test = (X_test-np.mean(X_test))/np.max(X_test)

#建構模型
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
def unet(pretrained_weights=None, input_size=(200, 200, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
#預測

#
