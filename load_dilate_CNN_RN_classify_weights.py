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
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score
from keras.regularizers import l1, l2
from sklearn.metrics import classification_report
from keras.models import model_from_json, load_model


################################################################################
################################ LOAD DATA #####################################
################################################################################


dir_name = '../new_classification_data/'

density = '08'

# load test data
fname = 'x_test_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
x_test_mat = sio.loadmat(filename)['x_test_mat']
x_test_mat = np.float32(x_test_mat)

fname = 'y_test_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
y_test_mat = sio.loadmat(filename)['y_test_mat']
y_test_mat = np.float32(y_test_mat)


# Form testing data

x_test = np.zeros((x_test_mat.shape[2],116,116,1),dtype=np.float32)
y_test = np.zeros((x_test_mat.shape[2],1),dtype=np.float32)

for i in range(x_test_mat.shape[2]):
    x_test[i,:,:,0] = x_test_mat[:,:,i]
    y_test[i,0] = y_test_mat[i,0]  # testing class labels



################################################################################################################################
################################################################################################################################
open_arch_name = 'RN_model_classify_arch_density_' + density + '.json'

with open(open_arch_name, 'r') as f:
    RN_model = model_from_json(f.read())
    

open_weights_name = 'best_RN_model_classify_weights_density_' + density + '.h5'

RN_model.load_weights(open_weights_name)

###################################################################################
# Test Model
y_test_prob = RN_model.predict(x_test)

y_test_pred = (y_test_prob > 0.5).astype(np.int) # for binary classification

y_test = np.reshape(y_test,(y_test_pred.shape[0],))
y_test_pred = np.reshape(y_test_pred,(y_test_pred.shape[0],))
y_test_prob = np.reshape(y_test_prob,(y_test_pred.shape[0],))

# Evaluate Model
#_, test_acc = RN_model.evaluate(x_test, y_test, verbose=0)

#print('Test Acc.: %.3f' % (test_acc))

# Print Results
score = f1_score(y_test, y_test_pred, average=None)
print('F1 score: ', score)

print(classification_report(y_test, y_test_pred))





