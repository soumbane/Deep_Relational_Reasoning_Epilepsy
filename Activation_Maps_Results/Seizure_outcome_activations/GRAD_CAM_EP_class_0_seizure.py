### GRAD-CAM - Gradient based class activation mapping
## This is for EP connectome class 0 for seizure outcome - HAS seizure

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

open_arch_name = 'RN_model_classify_arch_density_' + density + '.json'

with open(open_arch_name, 'r') as f:
    RN_model = model_from_json(f.read())
    

open_weights_name = 'new_best_RN_model_classify_weights_density_' + density + '.h5'

RN_model.load_weights(open_weights_name)

#RN_model.summary()

## Load Data
# data dir
dir_name = '../../new_classification_data/'

# load test data
fname = 'x_test_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
x_test_mat = sio.loadmat(filename)['x_test_mat']
x_test_mat = np.float32(x_test_mat)

fname = 'y_test_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
y_test_mat = sio.loadmat(filename)['y_test_mat']
y_test_mat = np.float32(y_test_mat)


x_test = np.zeros((x_test_mat.shape[2],116,116,1),dtype=np.float32)
y_test = np.zeros((x_test_mat.shape[2],),dtype=np.float32)

       
for i in range(x_test_mat.shape[2]):
    temp = x_test_mat[:,:,i]
    temp = np.triu(temp,1)
    x_test[i,:,:,0] = temp
    y_test[i] = y_test_mat[i,0]  # testing class labels


x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_patient_2 = x_test[2,:,:,0]


plt.figure()
plt.imshow(x_patient_2)
plt.title('Seizure outcome connectome with density'+' '+density)
plt.show()

###############################################################################################################################

# find the index of final linear layer for which gradients are to be calculated
final_layer_idx = utils.find_layer_idx(RN_model,'dense_8')

# Swap softmax with linear
RN_model.layers[final_layer_idx].activation = keras.activations.linear
RN_model = utils.apply_modifications(RN_model)

# find the index of penultimate layer w.r.t which gradients are to be calculated
# These are the final feature maps from the last conv layer

final_fmap_idx = utils.find_layer_idx(RN_model,'conv2d_4')


filter_indices = 0 # since class 0 of seizure outcome - HAS seizure

## calculate grad-cam - using combination of guided backprop and grad-cam
# backprop modifier can be 'None' or 'guided' or 'relu'

#############################################################################################################################
################################## Activation Maps for TOP 20 Test Cases ####################################################
#############################################################################################################################

start = time.time()

top = 20

grad_cam_top = np.zeros((x_test.shape[1],x_test.shape[2],top),dtype=np.float32)

for i in range(top):

    # Read image to find activations on
    x_pat = x_test[i,:,:,0]
    img = x_pat
    img = img.reshape(x_test.shape[1], x_test.shape[2], 1)

    # calculate grad-cam
    seed_input = img # image for which activation maps are to be visualized

    grad_cam = visualize_cam(RN_model, final_layer_idx, filter_indices, seed_input, 
                         penultimate_layer_idx = final_fmap_idx,
                         backprop_modifier = 'guided',
                         grad_modifier = None) 

    grad_cam_t = grad_cam.T
    grad_cam_final = grad_cam + grad_cam_t

    grad_cam_top[:,:,i] = grad_cam_final

    print(i)

    
# Save guided CAM    
sio.savemat('guided_grad_cam_class_0_top_20_test_cases.mat', mdict={'guided_grad_cam_class_0_top_20': grad_cam_top})

# Save CAM
#sio.savemat('grad_cam_class_0_top_20_test_cases.mat', mdict={'grad_cam_class_0_top_20': grad_cam_top})
    
#filename = 'guided_grad_cam_class_0_top_20_test_cases.mat'
#grad_cam_test = sio.loadmat(filename)['guided_grad_cam_class_0_top_20']    
#print(grad_cam_test.shape)

end = time.time()  

print("Time taken: ", (end-start))      
    





