# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:08:50 2021

@author: Ahmed Fayed
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import os
import cv2
import random

# Visualing and exploring random data 
Tomato_imgs = os.listdir('E:/Software/Practise Projects/Plant Diseases Prediction/Plant_images_pianalytix/Tomato___Bacterial_spot')

for i in range(25):
    rand_img = random.choice(Tomato_imgs)
    img_path = os.path.join('E:/Software/Practise Projects/Plant Diseases Prediction/Plant_images_pianalytix/Tomato___Bacterial_spot', rand_img)
    img = cv2.imread(img_path)
    plt.imshow(img)
    plt.xlabel(img.shape[1])
    plt.ylabel(img.shape[0])
    plt.show()

# Seems all images have the same size (256, 256)


# Now preparing data and labels
# first: making list of images_data
images_data = []
images_labels = []
classes_path = 'E:/Software/Practise Projects/Plant Diseases Prediction/Plant_images_pianalytix/'
classes_list = os.listdir(classes_path)
class_counter = -1

for class_file in classes_list:
    class_path = os.path.join(classes_path, class_file)
    class_imgs = os.listdir(class_path)
    
    class_counter += 1
    
    for img in class_imgs:
        img_path = os.path.join(class_path, img)
        img = cv2.imread(img_path)
        
        # Normalizing images
        img = img / 255.0
        
        images_data.append(img)
        images_labels.append(class_counter)
    


# splitting data into training and testing data
images_data = np.array(images_data)
x_train, x_test, y_train, y_test = train_test_split(images_data, images_labels, test_size=0.2, shuffle=True)

# splitting training data into training and validation data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)


# Making one hot encoding
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)



# creating model Architecture
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=64, epochs=15, verbose=2, validation_data=(x_val, y_val))
















