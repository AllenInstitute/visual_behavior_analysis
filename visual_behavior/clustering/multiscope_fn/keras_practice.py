#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aa = autoencoder.layers[5].get_weights(); 
np.shape(aa)


Keras practice, using:
https://blog.keras.io/building-autoencoders-in-keras.html
        
Created on Wed Dec  4 10:04:21 2019
@author: farzaneh
"""
# NOTE: I think this link is better:
# https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/

# number of parameters in cnn:
# https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/


#%%
import matplotlib.pyplot as plt

#%% We'll start simple, with a single fully-connected neural layer as encoder and as decoder:

from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)




#%% Alternative model with regularization
#%% Adding a sparsity constraint on the encoded representations

# The outputs (from this autoencoder or the previous one without regularization) look pretty similar to the previous model, the only significant difference being the sparsity of the encoded representations. encoded_imgs.mean() yields a value 3.33 (over our 10,000 test images), whereas with the previous model the same quantity was 7.30. So our new model yields encoded representations that are twice sparser.

from keras import regularizers

encoding_dim = 32

input_img = Input(shape=(784,))
# add a Dense layer with a L1 activity regularizer
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)



#%% Now let's train our autoencoder to reconstruct MNIST digits.

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


#%% Input data

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()


#%% We will normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784.

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


#%% Now let's train our autoencoder for 50 epochs:

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))






#%% Create encoder and decoder models, so later we can use predict to encode and decode xtest.

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))



#%% encode and decode some digits
# note that we take them from the *test* set

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)




# use Matplotlib
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




############################################################
############################################################
#%% Deep autoencoder
# We do not have to limit ourselves to a single layer as encoder or decoder, we could instead use a stack of layers, such as:

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))



#%% Create encoder and decoder models, so later we can use predict to encode and decode xtest.
# (I wrote this part.)

# the right way of doing it is this:
input_img = Input(shape=(input_dim,))
encoder_layer1 = autoencoder.layers[0]
encoder_layer2 = autoencoder.layers[1]
encoder_layer3 = autoencoder.layers[2]
encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))

encoder.summary()


### 
#### get encoded layer 1 
# units_dim0 = 784
l0_units = input_img # encoded_input

l1 = autoencoder.layers[1](l0_units)
encoder1 = Model(l0_units, l1)
# l1 = Dense(128, activation='relu')(input_img) # encoded # same as "l"
# e1 = Model(l0_units, l1)

input_data = x_test # 784 # encoded_imgs0
encoded_imgs1 = encoder1.predict(input_data) #128


##### get encoded layer 2 
units_dim1 = 128 # encoding_dim
l1_units = Input(shape=(units_dim1,))
# l1_units = Input(shape=(autoencoder.layers[2].input_shape[1:]))

l2 = autoencoder.layers[2](l1_units)
encoder2 = Model(l1_units, l2)

encoded_imgs2 = encoder2.predict(encoded_imgs1) # 64


##### get encoded layer 3
units_dim2 = 64
l2_units = Input(shape=(units_dim2,))
# l2_units = Input(shape=(autoencoder.layers[3].input_shape[1:]))

l3 = autoencoder.layers[3](l2_units)
encoder3 = Model(l2_units, l3)

encoded_imgs3 = encoder3.predict(encoded_imgs2) # 32


### decoding
##### get decoded layer 1 (encoded layer 4)
units_dim3 = 32
l3_units = Input(shape=(units_dim3,))
# l3_units = Input(shape=(autoencoder.layers[4].input_shape[1:]))

l4 = autoencoder.layers[4](l3_units)
encoder4 = Model(l3_units, l4) 

encoded_imgs4 = encoder4.predict(encoded_imgs3) # 64


##### get decoded layer 2 (encoded layer 5)
units_dim4 = 64
l4_units = Input(shape=(units_dim4,))
# l4_units = Input(shape=(autoencoder.layers[5].input_shape[1:]))

l5 = autoencoder.layers[5](l4_units)
encoder5 = Model(l4_units, l5) 

encoded_imgs5 = encoder5.predict(encoded_imgs4) # 128


##### get decoded layer 3 (encoded layer 6)
units_dim5 = 128
l5_units = Input(shape=(units_dim5,))
# l5_units = Input(shape=(autoencoder.layers[6].input_shape[1:]))

l6 = autoencoder.layers[6](l5_units)
encoder6 = Model(l5_units, l6) 

encoded_imgs6 = encoder6.predict(encoded_imgs5) # 784



decoded_imgs = encoded_imgs6


#%% Plot decoded images

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




############################################################
############################################################
#%% Convolutional autoencoder
# The encoder will consist in a stack of Conv2D and MaxPooling2D layers (max pooling being used for spatial down-sampling), while the decoder will consist in a stack of Conv2D and UpSampling2D layers.

# input_img = Input(shape=(784,))
# encoded = Dense(128, activation='relu')(input_img)

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # shape=(None, 28, 28, 16)
x = MaxPooling2D((2, 2), padding='same')(x) # shape=(None, 14, 14, 16)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # shape=(None, 14, 14, 8)
x = MaxPooling2D((2, 2), padding='same')(x) # shape=(None, 7, 7, 8)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # shape=(None, 7, 7, 8)
encoded = MaxPooling2D((2, 2), padding='same')(x) # shape=(None, 4, 4, 8)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded) # shape=(None, 4, 4, 8)
x = UpSampling2D((2, 2))(x) # shape=(None, 8, 8, 8)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # shape=(None, 8, 8, 8)
x = UpSampling2D((2, 2))(x) # shape=(None, 16, 16, 8)
x = Conv2D(16, (3, 3), activation='relu')(x) # shape=(None, 14, 14, 16) # HOW??
x = UpSampling2D((2, 2))(x) # shape=(None, 28, 28, 16)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # shape=(None, 28, 28, 1)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


#%% Use MNIST digits to train data

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


#%% Start a TensorBoard server that will read logs stored at /tmp/autoencoder.
#  to visualize the results of a model during training, we will be using the TensorFlow backend and the TensorBoard callback.

tensorboard --logdir=/tmp/autoencoder


#%% Train the model

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


#%% Plot the reconstructed images

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


decoded_imgs = autoencoder.predict(x_test)
# (Question: why in the previous model, we had to go through the encoder, and decoder models to get the decoder images? ... when I tried getting the decoded images, directly using the autoencoder model, the output looked weird!!)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#%% Plot the encoded images
# (Question: we didn't define encoded_imgs!!!)

# rerun the part above where you defined encoded_imgs6
encoded_imgs = encoded_imgs6

n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



################################################################################
################################################################################
#%% Application to image denoising

# Let's put our convolutional autoencoder to work on an image denoising problem. It's simple: we will train the autoencoder to map noisy digits images to clean digits images.
# Here's how we will generate synthetic noisy digits: we just apply a gaussian noise matrix and clip the images between 0 and 1.

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


#%% Plot
# x_train_noisy = x_train_noisy.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test_noisy = x_test_noisy.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train_noisy.shape)
# print(x_test_noisy.shape)

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#%% to improve the quality of the reconstructed, we'll use a slightly different model with more filters per layer:

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


#%% Train

autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])



YOU ARE HERE; WAITING FOR THE MODEL TO BE FITTED.
#%% Plot the reconstructed images

# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)
# print(x_test.shape)


decoded_imgs = autoencoder.predict(x_test_noisy)
# (Question: why in the previous model, we had to go through the encoder, and decoder models to get the decoder images? ... when I tried getting the decoded images, directly using the autoencoder model, the output looked weird!!)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


