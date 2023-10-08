#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:53:20 2020

@author: farzaneh
"""

#%%
'''
subSampNeurons = 0 # if 1, for model training, subsample neurons to the minumum neuron number across planes    
iters_neur_subsamp = 50 # number of times to subsample neurons

doZscore = 1 # z score each neuron (mean and sd across trials)
#norm_to_max_svm = 0 # normalize each neuron trace by its max    
#zEachFrame = 0 # if 0, normalize all frames to the same value. if 1, normalize each frame separately. # makes sense to set it to 0. 
do_pca_data = 0 # if 1, pca will be done on the data before training the model.

# NOTE: you did soft normalization when using inferred spikes ... i'm not sure if it makes sense here (using df/f trace); but we should take care of non-active neurons. 
softNorm = 0 #1 # soft normalziation : neurons with sd<thAct wont have too large values after normalization
#thAct = 5e-4 # will be used for soft normalization #1e-5 
kfold = 10 # assign 10% random samps as testing dataset
'''

#%%
# from def_funs import *

#%%
from keras.layers import Input, Dense, Dropout, Lambda, BatchNormalization, Conv1D, MaxPooling1D, UpSampling1D, Flatten
from keras.models import Model, Sequential, load_model
from keras import regularizers, callbacks, optimizers
import keras.backend as K
from sklearn.decomposition import PCA
import tensorflow as tf
# from sklearn.preprocessing import StandardScaler

print('--------------------------------')
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

##%% Make sure Tensorflow is using GPU

print('is_gpu_available:', tf.test.is_gpu_available())
print('--------------------------------')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
 
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

# Note, if the above gives the following error: AttributeError: module 'tensorflow_core._api.v2.config' has no attribute 'experimental_list_devices'
# then run below (ref: https://github.com/keras-team/keras/issues/13684)
'''
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

print(tfback._get_available_gpus())
'''


#%% Save losses over each batch during training

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
#        self.val_losses = []
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        # note: below didn't work, because val_loss is calculated at the end of each epoch and not each batch!
#        self.val_losses.append(logs.get('val_loss'))        

# Save keras model at specific epochs : 
# https://stackoverflow.com/questions/54323960/save-keras-model-at-specific-epochs
'''
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch == 2:  # or save after some epoch, each k-th epoch etc.
            self.model.save("model_{}.hd5".format(epoch))
'''

            
