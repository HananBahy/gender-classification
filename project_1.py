# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 01:40:01 2018

@author: Hanan
"""
import pandas as pd
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.layers import MaxPooling2D
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


###########Data#########
path_training='F:\\MIU\\dataset\\train'
#path_training='F:\\MIU\\dataset\\train'    #write path of traning data
path_test='F:\\MIU\dataset\\test'     #write path of test data
num_of_train=12569     #write number of pictures in traning data
num_of_test= 648     #write number of pictures in test data
training_set = ImageDataGenerator().flow_from_directory(path_training,
                                                 target_size = (224, 224),   #the size of input data to our CNN
                                                 batch_size = 40,
                                                 classes=['male', 'female'])

test_set = ImageDataGenerator().flow_from_directory(path_test,
                                                 target_size = (224, 224),   #the size of input data to our CNN
                                                 batch_size = 20,
                                                 classes=['male', 'female'])


######################Build Fine-tuned VGG16 model#######################3
#################################################################
vgg16_model=keras.applications.vgg16.VGG16()  #make object of exist model
vgg16_model.summary()

classifier=Sequential()  #has no layers
for layer in vgg16_model.layers[:-1]:
    classifier.add(layer)


# model.summary()   
#classifier.pop()   #remove the o/p layer to add the o/p layer that suitable our needs

for layer in vgg16_model.layers[:-3]:
    layer.trainable=False       #to block the retraning       

classifier.add(Dense(2, activation='softmax'))   #o/p layer has 2 nodes for males and females
#model.summary()


##############Train the model##############
     #optimzers
#opt = Adam(lr=2 * 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  #Adam optimizer
opt= SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)   #stochastic gradient desent 

classifier.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
#classifier.compile(Adam(lr=.001),loss='categorical_crossentropy',metrics=['accuracy'])
classifier.fit_generator(training_set,
                         steps_per_epoch =num_of_train,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps =num_of_test)
                         
##########################3
classifier.save('gender_classification.h5')
classifier.save('gender_classification_resumed.h5')                       
                         
                         
                         
                         