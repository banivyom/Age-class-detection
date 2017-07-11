# -*- coding: utf-8

import pandas as pd
import keras
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense,Dropout,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.utils import np_utils
from pathlib import Path
from keras.preprocessing.image import load_img,img_to_array
x = load_img('train/Train/0.jpg',target_size=(64,64))
x = img_to_array(x)
plt.imshow(x)
x = x.reshape(1,64,64,3)
train = x

for i in range(1,26542):
    print(i)
    my_file = Path('train/Train/' + str(i) + '.jpg')
    if  my_file.is_file():
        x = load_img('train/Train/' + str(i) + '.jpg', target_size = (64,64))
        x = img_to_array(x)
        x = x.reshape(1,64,64,3)
        train = np.concatenate((train,x),axis=0)
y_train = pd.read_csv('train/train.csv')

d2 = pd.DataFrame(y_train["ID"].str.split('.').tolist(), columns="ID ext".split())
I = d2["ID"].astype(int).tolist()
y_train.drop(["ID"],inplace=True, axis=1)
y_train["ID2"] = I
y_train = y_train[["ID2", "Class"]] 
y_train["Class"][y_train["Class"]=="YOUNG"] = 0
y_train["Class"][y_train["Class"]=="MIDDLE"] = 1
y_train["Class"][y_train["Class"]=="OLD"] = 2
y_train = y_train.sort_values(by="ID2")
y_train = np_utils.to_categorical(y_train["Class"], 3)

X_train = train.astype('float32')
X_train = train / 255

model = Sequential()
 
model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(64,64, 3)))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))


model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 

model.fit(X_train, y_train, 
          batch_size=128, epochs=70, verbose=1)


x = load_img('test/Test/2.jpg',target_size=(64,64))
x = img_to_array(x)
x = x.reshape(1,64,64,3)
test = x
for i in range(3,26541):
    print(i)
    my_file = Path('test/Test/' + str(i) + '.jpg')
    if  my_file.is_file():
        x = load_img('test/Test/' + str(i) + '.jpg', target_size = (64,64))
        x = img_to_array(x)
        x = x.reshape(1,64,64,3)
        test = np.append(test,x,axis=0)

test = test.astype('float32')
X_test = test/255
        
y = model.predict(X_test)

strs = ["" for x in range(len(test))]
for i in range(len(y)):
    if y[i][0]>y[i][1] and y[i][0]>y[i][2]:
        strs[i] = "YOUNG"
    elif y[i][2]>y[i][1] and y[i][2]>y[i][0]:
        strs[i] = "OLD"
    else:
        strs[i] = "MIDDLE"


y_test = pd.read_csv('test/test.csv')
J = list()
d2 = pd.DataFrame(y_test["ID"].str.split('.').tolist(), columns="ID ext".split())
J = d2["ID"].astype(int).tolist()
J.sort()
y_test["ID2"] = J
J=y_test["ID2"]
K = list(range(6636))
for f in range(6636):
    K[f] = str(J[f]) + '.jpg'
submit = pd.DataFrame({'Class': strs, 'ID': K})
submit.to_csv("final_solution1.csv", index=False)
