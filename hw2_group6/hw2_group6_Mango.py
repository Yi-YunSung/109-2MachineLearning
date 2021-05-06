#匯入模組
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

print(os.listdir('/Users/User/Desktop/Data_Mango'))
print(os.listdir('/Users/User/Desktop/Data_Mango/Train_Image'))

#讀取資料
trainPath = '/Users/User/Desktop/Data_Mango/Train_Image/'
testPath = '/Users/User/Desktop/Data_Mango/Test_Image/'

trainCSV = '/Users/User/Desktop/Data_Mango/train.csv'
testCSV   = '/Users/User/Desktop/Data_Mango/Test.csv'

trainDF = pd.read_csv(trainCSV)
print(trainDF)
trainFiles = trainDF['image_id'].tolist()
trainClasses = trainDF['label'].tolist()

testDF = pd.read_csv(testCSV)
print(testDF)
testFiles = testDF['image_id'].tolist()
testClasses = testDF['label'].tolist()

labels = ['A', 'B', 'C']

#圖片前處理
def plot_equilibre(equilibre, labels, title):
    plt.figure(figsize=(5,5))
    my_circle=plt.Circle( (0,0), 0.5, color='white')
    plt.pie(equilibre, labels=labels, colors=['red','green','blue'],autopct='%1.1f%%')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.title(title)
    plt.show()

equilibreTrain = []
[equilibreTrain.append(trainClasses.count(label)) for label in labels]
#print(equilibreTrain)
plot_equilibre(equilibreTrain, labels, 'Train Data')
del equilibreTrain

equilibreTest = []
[equilibreTest.append(testClasses.count(label)) for label in labels]
#print(equilibreTest)
plot_equilibre(equilibreTest, labels, 'Test Data')
del equilibreTest

TargetSize = (192, 144) # image ratio = 4:3
def prepare_image(filepath):
    img = cv2.imread(filepath)
    # get image height, width
    (h, w) = img.shape[:2]
    if (w<h): # rotate270
        # calculate the center of the image
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 270, 1.0)
        img = cv2.warpAffine(img, M, (h, w))
    img_resized = cv2.resize(img, TargetSize, interpolation=cv2.INTER_CUBIC)
    img_result  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_result

#plt.imshow(prepare_image(trainPath + trainFiles[1]))
#plt.imshow(prepare_image(testPath + '/'+ testFiles[1]))


'''
training data
'''
#訓練資料
trainX = []
[trainX.append(prepare_image(trainPath+file)) for file in trainFiles]
trainX = np.asarray(trainX)
print(trainX.shape)

# data normalisation
trainX = trainX / 255.0

# Convert Y_data from {'A','B','C'} to {0,1,2}
trainY = []
[trainY.append(ord(trainClass) - 65) for trainClass in trainClasses]
#print(trainY)

# one-hot encoding
trainY = to_categorical(trainY)


'''
testing data
'''
testX = []
[testX.append(prepare_image(testPath+file)) for file in testFiles]
testX = np.asarray(testX)
print(testX.shape)

# data normalisation
testX = testX / 255.0

# Convert Y_data from char to integer
testY = []
[testY.append(ord(testClass) - 65) for testClass in testClasses]
#print(testY)

# one-hot encoding
testY = to_categorical(testY)

trainX, trainY = shuffle(trainX, trainY, random_state=42)
num_classes = 3

'''
build model
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, BatchNormalization, Activation, LeakyReLU, Concatenate
from sklearn.metrics import classification_report, confusion_matrix

input_shape = trainX.shape[1:]
print(trainX.shape[1:])

# Build Model

input_image = Input(shape=input_shape)
# 1st Conv layer
model = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape)(input_image)
model = MaxPooling2D((2, 2),padding='same')(model)
# 2nd Conv layer
model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)
model = MaxPooling2D((2, 2),padding='same')(model)
# 3rd Conv layer
model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
model = MaxPooling2D((2, 2),padding='same')(model)
# 4th Conv layer
model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
model = MaxPooling2D((2, 2),padding='same')(model)
# 5th Conv layer
model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)
model = MaxPooling2D((2, 2),padding='same')(model)
# FC layers
model = Flatten()(model)

#model = Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(model)
model = Dense(1024)(model)
#model = Dropout(0.2)(model)

#model = Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(model)
model = Dense(64)(model)
#model = Dropout(0.2)(model)

output= Dense(num_classes, activation='softmax')(model)

model = Model(inputs=[input_image], outputs=[output])

print(model.summary())

# Compile Model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

'''
train model
'''

batch_size = 256
num_epochs = 20

# Train Model
history = model.fit(trainX,trainY,batch_size=batch_size,epochs=num_epochs) #, callbacks=[checkpoint])

predY = model.predict(testX)
y_pred = np.argmax(predY,axis=1)
y_actual = np.argmax(testY,axis=1)
#y_label= [labels[k] for k in y_pred]
cm = confusion_matrix(y_actual, y_pred)
print(cm)

'''
confusion matrix
'''

import itertools


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

plot_confusion_matrix(cm,
                      normalize=False,
                      target_names = labels,
                      title="Confusion Matrix, not Normalized")

print(model.evaluate(trainX, trainY))

print(classification_report(y_actual, y_pred, target_names=labels))