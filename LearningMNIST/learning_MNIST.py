#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:59:42 2019

@author: john
"""

import gzip
import numpy as np
import neural_network as nn
import matplotlib.pyplot as plt
import pickle

num_training_images = 60000



# %% read in training images
images_file = gzip.open('Data/train-images-idx3-ubyte.gz','r')

magic_num_images = np.frombuffer(images_file.read(4), dtype=np.uint8)

nums = np.frombuffer(images_file.read(4), dtype=np.uint8)
num_images = 16**6*nums[0] + 16**4*nums[1] + 16**2*nums[2] + nums[3]

nums = np.frombuffer(images_file.read(4), dtype=np.uint8)
num_rows = 16**6*nums[0] + 16**4*nums[1] + 16**2*nums[2] + nums[3]

nums = np.frombuffer(images_file.read(4), dtype=np.uint8)
num_cols = 16**6*nums[0] + 16**4*nums[1] + 16**2*nums[2] + nums[3]

image_size = num_rows * num_cols

images_buf = images_file.read(image_size * num_training_images)
images = np.frombuffer(images_buf, dtype=np.uint8).astype(np.float64)
images = images.reshape( num_training_images, image_size, 1)/255

images_file.close()



# %% read in training labels
labels_file = gzip.open('Data/train-labels-idx1-ubyte.gz','r')

magic_num_labels = np.frombuffer(labels_file.read(4), dtype=np.uint8)

nums = np.frombuffer(labels_file.read(4), dtype=np.uint8)
num_labels = 16**6*nums[0] + 16**4*nums[1] + 16**2*nums[2] + nums[3]

labels = np.frombuffer( labels_file.read(num_training_images), dtype=np.uint8 )

labels_file.close()



# %% read in test images
test_images_file = gzip.open('Data/t10k-images-idx3-ubyte.gz','r')

magic_num_test_images = np.frombuffer(test_images_file.read(4), dtype=np.uint8)

nums = np.frombuffer(test_images_file.read(4), dtype=np.uint8)
num_test_images = 16**6*nums[0] + 16**4*nums[1] + 16**2*nums[2] + nums[3]

nums = np.frombuffer(test_images_file.read(4), dtype=np.uint8)
num_test_rows = 16**6*nums[0] + 16**4*nums[1] + 16**2*nums[2] + nums[3]

nums = np.frombuffer(test_images_file.read(4), dtype=np.uint8)
num_test_cols = 16**6*nums[0] + 16**4*nums[1] + 16**2*nums[2] + nums[3]

image_size = num_test_rows * num_test_cols

images_buf = test_images_file.read(image_size * num_test_images)
test_images = np.frombuffer(images_buf, dtype=np.uint8).astype(np.float64)
test_images = test_images.reshape(num_test_images, image_size, 1)/255

test_images_file.close()



# %% read in test labels
test_labels_file = gzip.open('Data/t10k-labels-idx1-ubyte.gz','r')

magic_num_test_labels = np.frombuffer(test_labels_file.read(4), dtype=np.uint8)

nums = np.frombuffer(test_labels_file.read(4), dtype=np.uint8)
num_test_labels = 16**6*nums[0] + 16**4*nums[1] + 16**2*nums[2] + nums[3]

test_labels = np.frombuffer(test_labels_file.read(num_training_images), 
                            dtype=np.uint8 )

test_labels_file.close()



# %% combine images and labels into training_data and test_data arrays

eye = np.eye(10)
labeled_data = [ (x, eye[:,y:y+1]) for x,y in zip(images,labels) ]

testing_data = [ (x, eye[:,y:y+1]) for x,y in zip(test_images, test_labels) ]



# %% create and train network

net = nn.NeuralNetwork([num_rows*num_cols, 30, 10])

results = net.stochastic_gradient_decent(labeled_data, 30, 10, 0.5, 5.0/num_training_images, 
                                         testing_data,
                                         get_training_loss=True,
                                         get_training_acc=True,
                                         get_test_loss=True,
                                         get_test_acc=True)

test_loss, test_acc, training_loss, training_acc = results



# %% Plot results
plt.figure(1)
plt.plot(training_acc, label='Accuracy on training data')
plt.plot(test_acc, label='Accuracy on test data')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(training_loss, label='Loss on training data')
plt.plot(test_loss, label='Loss on test data')
plt.xlabel('Epoch')
plt.legend()
plt.grid()



# %% save weigths and biases to a file

#f = open('pickled_neural_net','wb')
#
#pickle.dump(net.layer_sizes,f)
#pickle.dump(net.biases,f)
#pickle.dump(net.weights,f)
#
#f.close()