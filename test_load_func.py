# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:33:02 2021

@author: ahmed
"""






# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:14:39 2021

@author: ahmed
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

import os
import cv2
import random


# Visualing and exploring random data
tomato_path = "E:/Software/Practise Projects/Plant Diseases Prediction/Plant_images_pianalytix/Tomato___Bacterial_spot"
tomato_imgs = os.listdir(tomato_path)

plt.figure(figsize=(12, 12))

for i in range(1, 9):
    plt.subplot(4, 4, i)
    img_path = os.path.join(tomato_path, random.choice(tomato_imgs))
    img = cv2.imread(img_path)
    plt.imshow(img)
    plt.xlabel(img.shape[1])
    plt.ylabel(img.shape[0])
    plt.show()

# Seems all images have the same size (256, 256)


# Now preparing data and labels
# first: making list of images_data
images = []
labels = []

classes_path = "E:/Software\Practise Projects/Plant Diseases Prediction/Plant_images_pianalytix"

class_counter = -1

for class_name in os.listdir(classes_path):
    class_path = os.path.join(classes_path, class_name)
    
    class_counter += 1
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        
        # Normalizing images
        img = (img / 255.0)
        img = cv2.resize(img, (256, 256))
        
        images.append(img)
        labels.append(class_counter)


# Exploring the number of visualize counts
labels_count = pd.DataFrame(labels).value_counts()
print(labels_count.head())

# We will also observe the number of images under different classes to see if the dataset is balanced or not
print(images[0].shape)



images = np.array(images)
# labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=True)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)




model = tf.keras.models.load_model('Plant_Disease_Detector.h5')
model.load_weights("Plant_Diseases_Weights.h5")

loss, acc = model.evaluate(x_test, y_test)





