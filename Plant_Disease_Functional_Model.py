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




# Creating model

input = Input(shape=(256, 256, 3))

x = Conv2D(32, (3, 3), padding="same", activation="relu")(input)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(16, (3, 3), padding="same", activation="relu")(x)
x = AveragePooling2D()(x)

x = Flatten()(x)
x = Dense(8, activation="relu")(x)
# x = Dropout(0.2)(x)
predictions = Dense(3, activation="softmax")(x)

model = Model(inputs = input, outputs = predictions)



class DetectOverfittingCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(DetectOverfittingCallback, self).__init__()
        self.threshold = threshold
        
        
    def on_epoch_end(self, epoch, logs=None):
        ratio = logs["val_loss"] / logs["loss"]
        print(" , Epoch {}, Val/Train ratio: {:.2f}".format(epoch, ratio))
        
        if ratio > self.threshold:
            print("Stop training...")
            self.model.stop_training = True




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=25, validation_data=(x_val, y_val),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True), CSVLogger('training.csv'), DetectOverfittingCallback(threshold=1.3)])




plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], color='g',)
plt.plot(history.history['val_accuracy'], color='b')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['train', 'val'], loc='lower right')
plt.show()



# Calculating model accuracy
loss, accuracy = model.evaluate(x_test, y_test)


model.save("Plant_Disease_Detector.h5")

json_model = model.to_json()

with open("E:/Software/professional practice projects/In progress/Plant_Disease_model.json", 'w') as json_file:
    json_file.write(json_model)

model.save_weights("Plant_Diseases_Weights.h5")


