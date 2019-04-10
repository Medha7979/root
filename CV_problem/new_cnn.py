# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 06:03:52 2019

@author: MEDHA
"""

# Python program to create 
# Image Classifier using CNN 
  
# Importing the required libraries 
import cv2 
import os 
import numpy as np 
#from random import shuffle 
#from tqdm import tqdm 
#import PIL
import pickle

file = open('C:/Users/MEDHA/Desktop/midas_iiitd/train_image.pkl', 'rb')
f = open('C:/Users/MEDHA/Desktop/midas_iiitd/train_label.pkl', 'rb')
f1=open('C:/Users/MEDHA/Desktop/midas_iiitd/test_image.pkl', 'rb')
image = pickle.load(file)
xx = np.array(image)
im=pickle.load(f)
a=np.array(im)
im1 = pickle.load(f1)
b=np.array(im1)




  
'''Setting up the env'''
  

LR = 1e-3
  
 
  
'''Labelling the dataset'''
def label_img(m): 
    s=a[m]
    return s

'''Creating the training data'''
def create_train_data(): 
    # Creating an empty list where we should the store the training data 
    # after a little preprocessing of the data 
    training_data = [] 
    q=0

    # loading the training data 
    for img in xx: 
        
        # labeling the images 
        label = label_img(q) 
  
        training_data.append([np.array(img), np.array(label)])
        
        q=q+1
  
    # saving our trained data for further uses if required 
    np.save('train_data.npy', training_data) 
    return training_data 
  
'''Processing the given test data'''

def process_test_data(): 
    testing_data = [] 
    j=0
    for img in b: 
        testing_data.append([np.array(img), j]) 
        j=j+1
    
    np.save('test_data.npy', testing_data) 
    return testing_data 
  
'''Running the training and the testing in the dataset for our model'''
train_data = create_train_data() 
test_data = process_test_data() 
  

'''Creating the neural network using tensorflow'''
# Importing the required libraries 
import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
  
import tensorflow as tf 
IMG_SIZE=28
tf.reset_default_graph() 
convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = fully_connected(convnet, 1024, activation ='relu') 
convnet = dropout(convnet, 0.8) 
  
convnet = fully_connected(convnet, 1, activation='linear') 
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
      loss ='categorical_crossentropy', name ='targets') 
  
model = tflearn.DNN(convnet, tensorboard_dir ='log') 
  
# Splitting the testing data and training data 
train = train_data[:-500] 
test = train_data[-500:] 
  
'''Setting up the features and lables'''
# X-Features & Y-Labels 
  
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
Y = np.array([i[1] for i in train]).reshape(-1, 1) 
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
test_y = np.array([i[1] for i in test]).reshape(-1, 1)  

  
'''Fitting the data into our model'''
# epoch = 5 taken 
model.fit({'input': X}, {'targets': Y}, n_epoch = 5,  
    validation_set =({'input': test_x}, {'targets': test_y}),  
    snapshot_step = 500, show_metric = True, run_id = None) 
model.save('me1') 
  
'''Testing the data'''
 


# if you need to create the data: 
#test_data = process_test_data() 
# if you already have some saved: 
test_data = np.load('test_data.npy') 
  

for num, data in enumerate(test_data[:]): 

      
    img_num = data[1] 
    img_data = data[0] 
      
 
    orig = img_data 
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1) 
  

    model_out = model.predict([data])[0] 
    if (model_out[0] == 0): 
        str_label ='0'
    elif(model_out[0] == 2): 
        str_label ='2'
    elif(model_out[0] == 3): 
        str_label ='3'
    elif(model_out[0]==6):
        str_label ='6'
    print(model_out[0])
    
  