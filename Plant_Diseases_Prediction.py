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
label = [1, 2, 3]
class_counter = 0

for class_file in classes_list:
    class_path = os.path.join(classes_path, class_file)
    class_imgs = os.listdir(class_path)
    
    class_counter += 1
    
    for img in class_imgs:
        img_path = os.path.join(class_path, img)
        img = cv2.imread(img_path)
        
        # Normalizing images
        img = img / 255
        
        # Converting images to numpy array
        img = np.array(img)
        
        images_data.append(img)
        images_labels.append(class_counter)
    

















