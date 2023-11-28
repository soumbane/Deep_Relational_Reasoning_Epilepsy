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


batch_size = 128
epochs = 20

################################################################################
################################ LOAD DATA #####################################
################################################################################


dir_name = 'new_classification_data/'

density = '04'

# load train data

fname = 'x_train_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
x_train_mat = sio.loadmat(filename)['x_train_mat']
x_train_mat = np.float32(x_train_mat)

fname = 'y_train_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
y_train_mat = sio.loadmat(filename)['y_train_mat']
y_train_mat = np.float32(y_train_mat)

# load test data
fname = 'x_test_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
x_test_mat = sio.loadmat(filename)['x_test_mat']
x_test_mat = np.float32(x_test_mat)

fname = 'y_test_density_' + density + '.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
y_test_mat = sio.loadmat(filename)['y_test_mat']
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
    y_train[i,0] = y_train_mat[i,0] # training class labels
        
for i in range(x_test_mat.shape[2]):
    x_test[i,:,:,0] = x_test_mat[:,:,i]
    y_test[i,0] = y_test_mat[i,0]  # testing class labels

  

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
plt.imshow(x_train[2,:,:,0])
plt.title('EP connectome with density'+' '+density)
plt.show()

d_rate = 2

# Define 4 convolutional layers
def ConvolutionNetworks(no_filters=32, kernel_size=3, stride_size=1):
    def conv(model):
        model = Conv2D(no_filters, (3,3), strides=(stride_size,stride_size), activation='relu', input_shape=input_shape, data_format='channels_last')(model)
        model = MaxPooling2D()(model)
        model = BatchNormalization()(model)
        
        model = Conv2D(no_filters, (3,3), strides=(stride_size,stride_size), activation='relu')(model)
        model = MaxPooling2D()(model)
        model = BatchNormalization()(model)
        
        model = Conv2D(no_filters, (3,3), strides=(stride_size,stride_size), dilation_rate=(d_rate,d_rate), activation='relu')(model)
        model = MaxPooling2D()(model)
        model = BatchNormalization()(model)
        
        model = Conv2D(no_filters, (3,3), strides=(stride_size,stride_size), dilation_rate=(d_rate,d_rate), activation='relu')(model) # d=3
        #model = Conv2D(no_filters, (3,3), strides=(stride_size,stride_size), activation='relu')(model) # d=4
        model = MaxPooling2D()(model)
        model = BatchNormalization()(model)
        
        return model
    return conv


# Define function to compute relations from objects - the following uses just 4 Lambda layers - O(n^2) time complexity

def compute_relations(objects):
    
    def get_top_dim_1(t):
        return t[:,0,:,:]
    
    def get_all_but_top_dim_1(t):
        return t[:,1:,:,:]
    
    def get_top_dim_2(t):
        return t[:,0,:]
    
    def get_all_but_top_dim_2(t):
        return t[:,1:,:]
    
    slice_top_dim_1 = Lambda(get_top_dim_1)
    slice_all_but_top_dim_1 = Lambda(get_all_but_top_dim_1)
    slice_top_dim_2 = Lambda(get_top_dim_2)
    slice_all_but_top_dim_2 = Lambda(get_all_but_top_dim_2)
    
    d = K.int_shape(objects)[2]
    print('d = ', d)
    features = []
    
    for i in range(d):
        features1 = slice_top_dim_1(objects)
        objects = slice_all_but_top_dim_1(objects)
        
        for j in range(d):
            features2 = slice_top_dim_2(features1)
            features1 = slice_all_but_top_dim_2(features1)
            features.append(features2)
            
    relations = []
    concat = Concatenate()
    for feature1 in features:
        for feature2 in features:
            if (features.index(feature1) < features.index(feature2)):
                relations.append(concat([feature1,feature2]))
                    
            
    return relations



# Baseline model
def f_theta():
    def f(model):
        model = Dense(512)(model)
        model = Activation('relu')(model)
        model = Dropout(0.5)(model)
        
        model = Dense(512)(model)
        model = Activation('relu')(model)
        model = Dropout(0.5)(model)
        
        model = Dense(512)(model)
        model = Activation('relu')(model)
        
        #model = Dense(100)(model)
        #model = Activation('relu')(model)
        
        return model
    return f


# Define the Relation Networks

def g_th(layers):
    def f(model):
        for n in range(len(layers)):
            model = layers[n](model)
        return model
    return f

def stack_layer(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x
    return f

def g_theta(units=512, layers=4):
    r = []
    for k in range(layers):
        r.append(Dense(units))
        r.append(Activation('relu'))
        #r.append(kernel_regularizer=l2(0.01))
        #r.append(activity_regularizer=l1(0.01))
    return g_th(r)

def get_MLP():
    return g_th()


# Define the main RN
    
def RelationNetworks(objects):
    g_t = g_theta()
    relations = compute_relations(objects)
    print('No of Relations:', len(relations))
    #print(relations)
    
    g_all = []
    
    for i, r in enumerate(relations):
        g_all.append(g_t(r))
        
    # combine (pool) to make the network combinatorially generalizable
    #combined_relation = Add()(g_all)
    #combined_relation = Maximum()(g_all)
    combined_relation = Average()(g_all)
    
    f_out = f_theta()(combined_relation)
    return f_out

def build_tag(conv):
    d = K.int_shape(conv)[2]
    tag = np.zeros((d,d,2))
    
    for i in range(d):
        for j in range(d):
            tag[i,j,0] = float(int(i%d))/(d-1)*2-1
            tag[i,j,1] = float(int(j%d))/(d-1)*2-1
            
    tag = K.variable(tag)
    tag = K.expand_dims(tag, axis=0)
    batch_size = K.shape(conv)[0]
    tag = K.tile(tag, [batch_size,1,1,1])
    
    return Input(tensor=tag)


input_img = Input((x_train.shape[1], x_train.shape[2], 1))  
img_after_conv = ConvolutionNetworks()(input_img) 
#tag = build_tag(img_after_conv)
#img_after_conv = Concatenate()([tag, img_after_conv])

img_after_RN = RelationNetworks(img_after_conv)

img_out = Dense(1, activation='sigmoid')(img_after_RN)

#RN_model = Model(inputs=[input_img, tag], outputs=img_out)
RN_model = Model(inputs=[input_img], outputs=img_out)

#ababa

###################################################################################
############################# MODEL COMPILATION ###################################
###################################################################################

#compile model
lr = 0.0001
decay_rate = lr/epochs
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
RN_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = RN_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)


# Test Model
y_test_prob = RN_model.predict(x_test)
#y_test_prob = saved_model.predict(np.array(x_test))

y_test_pred = (y_test_prob > 0.5).astype(np.int) # for binary classification

y_test = np.reshape(y_test,(y_test_pred.shape[0],))
y_test_pred = np.reshape(y_test_pred,(y_test_pred.shape[0],))
y_test_prob = np.reshape(y_test_prob,(y_test_pred.shape[0],))

# Evaluate Model
_, train_acc = RN_model.evaluate(x_train, y_train, verbose=0)
_, test_acc = RN_model.evaluate(x_test, y_test, verbose=0)

print('Train Acc.: %.3f, Test Acc.: %.3f' % (train_acc, test_acc))

# Print Results
score = f1_score(y_test, y_test_pred, average=None)
print('F1 score: ', score)

print(classification_report(y_test, y_test_pred))


#train_loss = history.history['loss']
#np.save('train_loss_dilate_CNN_RN',train_loss)

## plot training history - summarize history for training loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'], label='test')
#plt.xlabel('Epochs')
#plt.ylabel('Training Loss')
#plt.title('Loss variation with epochs for dilate CNN RN')
#plt.legend()
#plt.show()




