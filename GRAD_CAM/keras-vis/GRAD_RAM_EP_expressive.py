### GRAD-RAM - Gradient based regression activation mapping
## This is for EP connectome expressive scores

## import necessary packages
import numpy as np
import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import vis
import json
from keras.models import model_from_json, load_model
import time

## import gradient visualization functions
from vis.utils import utils
from vis.visualization import visualize_cam, visualize_saliency
import keras.backend as K
from scipy.ndimage.interpolation import zoom


## Read pre-trained model
density = '04'

mode = 'expressive'

open_arch_name = 'leaky_upper_triang_RN_model_' + mode + '_arch_density_' + density + '.json'

with open(open_arch_name, 'r') as f:
    RN_model = model_from_json(f.read())
    

open_weights_name = 'leaky_upper_triang_RN_model_' + mode + '_weights_density_' + density + '.h5'

RN_model.load_weights(open_weights_name)

#RN_model.summary()

## Load Data

# expressive data dir
dir_name = '../../processed_data/expressive_data/'

# load test data

fname = 'x_test_expressive_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
x_test_mat = sio.loadmat(filename)['x_test_expressive_mat']
x_test_mat = np.float32(x_test_mat)

fname = 'y_test_expressive_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
y_test_mat = sio.loadmat(filename)['y_test_expressive_mat']
y_test_mat = np.float32(y_test_mat)

# Form testing data

x_test = np.zeros((x_test_mat.shape[2],116,116,1),dtype=np.float32)
y_test = np.zeros((x_test_mat.shape[2],1),dtype=np.float32)

for i in range(x_test_mat.shape[2]):
    temp = x_test_mat[:,:,i]
    temp = np.triu(temp,1)
    x_test[i,:,:,0] = temp
    y_test[i,0] = y_test_mat[i,0]  # EXPRESSIVE SCORE

x_patient_2 = x_test[21,:,:,0]


plt.figure()
plt.imshow(x_patient_2)
plt.title('EP expressive connectome with density'+' '+density)
plt.show()


# find the index of final linear layer for which gradients are to be calculated
final_layer_idx = utils.find_layer_idx(RN_model,'dense_8')

# find the index of penultimate layer w.r.t which gradients are to be calculated
# These are the final feature maps from the last conv layer

final_fmap_idx = utils.find_layer_idx(RN_model,'conv2d_4')


filter_indices = 0 #since just one regression output

## calculate grad-ram - using combination of guided backprop and grad-ram
# backprop modifier can be 'None' or 'guided' or 'relu'

#########################################################################################################################
################################ Activation Maps for TOP 200 Test Cases ################################################
#########################################################################################################################
'''
y_test_pred = RN_model.predict(x_test)

abs_error_arr = np.zeros((len(y_test),1),dtype=np.float32)

for i in range(len(y_test)):
    abs_error_arr[i] = np.abs(y_test[i] - y_test_pred[i])
    
sorted_idxs = np.argsort(abs_error_arr,axis=0)
sorted_abs_error_arr = np.sort(abs_error_arr,axis=0)


start = time.time()

top = 200

grad_ram_top = np.zeros((x_test.shape[1],x_test.shape[2],top),dtype=np.float32)

for i in range(top):

    # Read image to find activations on
    x_pat = x_test[sorted_idxs[i][0],:,:,0]
    img = x_pat
    img = img.reshape(x_test.shape[1], x_test.shape[2], 1)

    # calculate grad-ram
    seed_input = img # image for which activation maps are to be visualized

    grad_ram = visualize_cam(RN_model, final_layer_idx, filter_indices, seed_input, 
                         penultimate_layer_idx = final_fmap_idx,
                         backprop_modifier = 'guided',
                         grad_modifier = None) 

    grad_ram_t = grad_ram.T
    grad_ram_final = grad_ram + grad_ram_t

    grad_ram_top[:,:,i] = grad_ram_final

    
    x_pat_3 = x_pat[np.newaxis,:,:,np.newaxis]
    y_pred = RN_model.predict(x_pat_3)
    print(i)
    print(sorted_idxs[i][0])
    print("y_actual: ", y_test[sorted_idxs[i][0]])
    print("y_predicted: ", y_pred)



sio.savemat('guided_grad_ram_expressive_matrix_top_200_test_cases.mat', mdict={'guided_grad_ram_expressive_top_200': grad_ram_top})
    
filename = 'guided_grad_ram_expressive_matrix_top_200_test_cases.mat'
grad_ram_test = sio.loadmat(filename)['guided_grad_ram_expressive_top_200']    

end = time.time()  

print("Time taken: ", (end-start))      
    
print(grad_ram_test.shape)

'''
#############################################################################################################################
################################## Activation Maps for TOP 20 Test Cases ####################################################
#############################################################################################################################
'''
y_test_pred = RN_model.predict(x_test)

abs_error_arr = np.zeros((len(y_test),1),dtype=np.float32)

for i in range(len(y_test)):
    abs_error_arr[i] = np.abs(y_test[i] - y_test_pred[i])
    
sorted_idxs = np.argsort(abs_error_arr,axis=0)
sorted_abs_error_arr = np.sort(abs_error_arr,axis=0)


start = time.time()

top = 20

grad_ram_top = np.zeros((x_test.shape[1],x_test.shape[2],top),dtype=np.float32)

for i in range(top):

    # Read image to find activations on
    x_pat = x_test[sorted_idxs[i][0],:,:,0]
    img = x_pat
    img = img.reshape(x_test.shape[1], x_test.shape[2], 1)

    # calculate grad-ram
    seed_input = img # image for which activation maps are to be visualized

    grad_ram = visualize_cam(RN_model, final_layer_idx, filter_indices, seed_input, 
                         penultimate_layer_idx = final_fmap_idx,
                         backprop_modifier = 'guided',
                         grad_modifier = None) 

    grad_ram_t = grad_ram.T
    grad_ram_final = grad_ram + grad_ram_t

    grad_ram_top[:,:,i] = grad_ram_final

    
    x_pat_3 = x_pat[np.newaxis,:,:,np.newaxis]
    y_pred = RN_model.predict(x_pat_3)
    print(i)
    print(sorted_idxs[i][0])
    print("y_actual: ", y_test[sorted_idxs[i][0]])
    print("y_predicted: ", y_pred)



sio.savemat('guided_grad_ram_expressive_matrix_top_20_test_cases.mat', mdict={'guided_grad_ram_expressive_top_20': grad_ram_top})
    
filename = 'guided_grad_ram_expressive_matrix_top_20_test_cases.mat'
grad_ram_test = sio.loadmat(filename)['guided_grad_ram_expressive_top_20']    

end = time.time()  

print("Time taken: ", (end-start))      
    
print(grad_ram_test.shape)
'''

#########################################################################################################################
################################ Activation Maps for TOP 300 Test Cases ################################################
#########################################################################################################################

y_test_pred = RN_model.predict(x_test)

abs_error_arr = np.zeros((len(y_test),1),dtype=np.float32)

for i in range(len(y_test)):
    abs_error_arr[i] = np.abs(y_test[i] - y_test_pred[i])
    
sorted_idxs = np.argsort(abs_error_arr,axis=0)
sorted_abs_error_arr = np.sort(abs_error_arr,axis=0)


start = time.time()

top = 300

grad_ram_top = np.zeros((x_test.shape[1],x_test.shape[2],top),dtype=np.float32)

for i in range(top):

    # Read image to find activations on
    x_pat = x_test[sorted_idxs[i][0],:,:,0]
    img = x_pat
    img = img.reshape(x_test.shape[1], x_test.shape[2], 1)

    # calculate grad-ram
    seed_input = img # image for which activation maps are to be visualized

    grad_ram = visualize_cam(RN_model, final_layer_idx, filter_indices, seed_input, 
                         penultimate_layer_idx = final_fmap_idx,
                         backprop_modifier = 'guided',
                         grad_modifier = None) 

    grad_ram_t = grad_ram.T
    grad_ram_final = grad_ram + grad_ram_t

    grad_ram_top[:,:,i] = grad_ram_final

    
    x_pat_3 = x_pat[np.newaxis,:,:,np.newaxis]
    y_pred = RN_model.predict(x_pat_3)
    print(i)
    print(sorted_idxs[i][0])
    print("y_actual: ", y_test[sorted_idxs[i][0]])
    print("y_predicted: ", y_pred)



sio.savemat('guided_grad_ram_expressive_matrix_top_300_test_cases.mat', mdict={'guided_grad_ram_expressive_top_300': grad_ram_top})
    
filename = 'guided_grad_ram_expressive_matrix_top_300_test_cases.mat'
grad_ram_test = sio.loadmat(filename)['guided_grad_ram_expressive_top_300']    

end = time.time()  

print("Time taken: ", (end-start))      
    
print(grad_ram_test.shape)



