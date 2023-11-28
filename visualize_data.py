from __future__ import print_function
import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

################################################################################
################################ LOAD DATA #####################################
################################################################################


dir_name = 'processed_data/expressive_data/'

density = '08'

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
    

################################################################################
################################################################################
    

##################################################################################
############################### DEFINE RN MODEL ##################################
##################################################################################

# Reshape input
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
input_shape = (x_train.shape[1], x_train.shape[2], 1)

plt.figure()
plt.imshow(x_train[15,:,:,0])
#plt.title('EP connectome with density'+' '+density)
plt.colorbar()
plt.show()





