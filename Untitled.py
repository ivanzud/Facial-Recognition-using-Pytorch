#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import tensorflow as tf
import skimage.color as color
import skimage.io as io
import os


# In[2]:


# pip install -U numpy


# In[3]:


mydir = r'face_images'
images = [files for files in os.listdir(mydir)]
N = 750;
data = np.zeros([N, 256, 256, 3]) # N is number of images for training
for count in range(len(images)):
 img = cv2.resize(io.imread(mydir + '/'+ images[count]), (256, 256))
 data[count,:,:,:] = img
num_train = N
Xtrain = color.rgb2lab(data[:num_train]*1.0/255)
xt = Xtrain[:,:,:,0]
yt = Xtrain[:,:,:,1:]
yt = yt/128
xt = xt.reshape(num_train, 256, 256, 1)
yt = yt.reshape(num_train, 256, 256, 2)


# In[4]:


# import tensorflow as tf 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)
a = tf.constant(5)
b = tf.constant(6)
c = a*b
print(c)


# In[5]:


session = tf.Session(config=config)
x = tf.placeholder(tf.float32, shape = [None, 256, 256, 1], name = 'x')
ytrue = tf.placeholder(tf.float32, shape = [None, 256, 256, 2], name = 'ytrue')


# In[6]:


def create_weights(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def create_bias(size):
  return tf.Variable(tf.constant(0.1, shape = [size]))


# In[7]:


def convolution(inputs, num_channels, filter_size, num_filters):
  weights = create_weights(shape = [filter_size, filter_size, num_channels, num_filters])
  bias = create_bias(num_filters)
  
  ## convolutional layer
  layer = tf.nn.conv2d(input = inputs, filter = weights, strides= [1, 1, 1, 1], padding = 'SAME') + bias
  layer = tf.nn.tanh(layer)
  return layer


# In[8]:


def maxpool(inputs, kernel, stride):
  layer = tf.nn.max_pool(value = inputs, ksize = [1, kernel, kernel, 1], strides = [1, stride, stride, 1], padding = "SAME")
  return layer
def upsampling(inputs):
  layer = tf.image.resize_nearest_neighbor(inputs, (2*inputs.get_shape().as_list()[1], 2*inputs.get_shape().as_list()[2]))
  return layer


# In[9]:


conv1 = convolution(x, 1, 3, 3)
max1 = maxpool(conv1, 2, 2)
conv2 = convolution(max1, 3, 3, 8)
max2 = maxpool(conv2, 2, 2)
conv3 = convolution(max2, 8, 3, 16)
max3 = maxpool(conv3, 2, 2)
conv4 = convolution(max3, 16, 3, 16)
max4 = maxpool(conv4, 2, 2)
conv5 = convolution(max4, 16, 3, 32)
max5 = maxpool(conv5, 2, 2)
conv6 = convolution(max5, 32, 3, 32)
max6 = maxpool(conv6, 2, 2)
conv7 = convolution(max6, 32, 3, 64)
upsample1 = upsampling(conv7)
conv8 = convolution(upsample1, 64, 3, 32)
upsample2 = upsampling(conv8)
conv9 = convolution(upsample2, 32, 3, 32)
upsample3 = upsampling(conv9)
conv10 = convolution(upsample3, 32, 3, 16)
upsample4 = upsampling(conv10)
conv11 = convolution(upsample4, 16, 3, 16)
upsample5 = upsampling(conv11)
conv12 = convolution(upsample5, 16, 3, 8)
upsample6 = upsampling(conv12)
conv13 = convolution(upsample6, 8, 3, 2)


# In[10]:


loss = tf.losses.mean_squared_error(labels = ytrue, predictions = conv13)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)
session.run(tf.global_variables_initializer())


# In[ ]:


num_epochs = 100
for i in range(num_epochs):
    session.run(optimizer, feed_dict = {x: xt, ytrue:yt})
    lossvalue = session.run(cost, feed_dict = {x:xt, ytrue : yt})
    print("epoch: " + str(i) + " loss: " + str(lossvalue))


# In[ ]:




