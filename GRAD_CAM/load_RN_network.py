from __future__ import print_function
import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Concatenate, Add, Maximum, Average
from keras.layers import Conv2D, Input, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping

import json
from keras.models import model_from_json, load_model


################################################################################
################################ LOAD DATA #####################################
################################################################################


dir_name = 'processed_data/expressive_data/'

density = '05'

# load train data

fname = 'x_train_expressive_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
x_train_mat = sio.loadmat(filename)['x_train_expressive_mat']
x_train_mat = np.float32(x_train_mat)

fname = 'y_train_expressive_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
y_train_mat = sio.loadmat(filename)['y_train_expressive_mat']
y_train_mat = np.float32(y_train_mat)

# load test data
fname = 'x_test_expressive_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
x_test_mat = sio.loadmat(filename)['x_test_expressive_mat']
x_test_mat = np.float32(x_test_mat)

fname = 'y_test_expressive_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
y_test_mat = sio.loadmat(filename)['y_test_expressive_mat']
y_test_mat = np.float32(y_test_mat)

print('x_train_shape:',x_train_mat.shape)
print('x_test_shape:',x_test_mat.shape)
print('y_train_shape:',y_train_mat.shape)
print('y_test_shape:',y_test_mat.shape)


# Form training and testing data

x_train = np.zeros((x_train_mat.shape[2],116,116,1),dtype=np.float32)
y_train = np.zeros((x_train_mat.shape[2],1),dtype=np.float32)

x_test = np.zeros((x_test_mat.shape[2],116,116,1),dtype=np.float32)
y_test = np.zeros((x_test_mat.shape[2],1),dtype=np.float32)


for i in range(x_train_mat.shape[2]):
    x_train[i,:,:,0] = x_train_mat[:,:,i]
    y_train[i,0] = y_train_mat[i,0] # EXPRESSIVE SCORE
        
for i in range(x_test_mat.shape[2]):
    x_test[i,:,:,0] = x_test_mat[:,:,i]
    y_test[i,0] = y_test_mat[i,0]  # EXPRESSIVE SCORE

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


## Load pretrained Model
with open('RN_model_architecture.json', 'r') as f:
    RN_model = model_from_json(f.read())

RN_model.load_weights('RN_model_weights.h5')


# Test Model
y_test_pred = RN_model.predict(x_test)

print(y_test.shape)
print('y_test:', y_test)


print(y_test_pred.shape)
print('y_test_pred:', y_test_pred)


# Print Results
print('mae: ', np.mean(np.abs(y_test - y_test_pred)))
print('sdae: ', np.std(np.abs(y_test - y_test_pred)))

count = 0
for i in range(len(y_test)):
    if (np.abs(y_test[i] - y_test_pred[i])) < 0.125: # prob of mae less than 15 - 15/120 = 0.125
        count += 1
        
prob_error = count / len(y_test)

print('prob of mae less than 0.125: ', prob_error)
