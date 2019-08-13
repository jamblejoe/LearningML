#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:59:47 2019

@author: john
"""

import numpy as np
import neural_network as nn
import matplotlib.pyplot as plt



def func(x):
    """The function we want to learn"""
    return x**2



def loss(net, xs, ys):
    """
    The loss function is the average squared Euclidean distance between the network 
    output and the target output of the test data
    """
    dists = [ (y - net.feed_forward(x))**2
              for x,y in zip(xs,ys) ]
    
    return sum(dists)/len(dists)



# %% Generate training and test data
print('Generate training and test data')
num_rand_points = 10000

random_xs = 2*np.random.rand(num_rand_points) - 1
target_ys = func(random_xs)
training_data = [ (x,y) for x,y in zip(random_xs,target_ys) ]

num_test_points = 1000

random_xs = 2*np.random.rand(num_test_points) - 1
target_ys = func(random_xs)
test_data = [ (x,y) for x,y in zip(random_xs,target_ys) ]



# %% Train network
print('Train network')
net = nn.NeuralNetwork([1,2,1])
x = np.arange(-1,1,0.01)
num = 50
y = np.zeros([num+1,len(x)])

y[0,:] = np.asarray( list(map(net.feed_forward, x)) ).reshape(len(x))

for i in range(1,num+1):
    
    net.stochastic_gradient_decent(training_data, 1, 50, 1.0)
    
    y[i,:] = np.asarray( list(map(net.feed_forward, x)) ).reshape(len(x))

weights = net.weights
biases = net.biases



# %% plot network output as it learned    
print('plot network output as it learned')
plt.figure(1)
for i in range(50,51):
    plt.clf()
    plt.plot(random_xs,target_ys,'o', markerSize=1, label='data')
    plt.plot(x,y[i,:],lw=4,label='net')
    plt.title('Epoch {0}'.format(i))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.pause(0.02)
    
    
    
# %% evaluate cost
print('evaluate cost')

from itertools import chain

c = np.zeros( [sum([w.size for w in chain(weights,biases)]) ,len(x)])

k = 0
for param in chain(weights,biases):
    
    n,m = param.shape
    
    for i in range(n):
        for j in range(m):
            
            omega = param[i,j]

            for l,xi in enumerate(x):
                param[i,j] = omega + 10*xi
                c[k,l] = loss(net, random_xs, target_ys)
            
            print('calculated change in loss due to varying omega_{0}'
                  .format(k) )
            param[i,j] = omega
            k += 1



# %% plot the graphs  
plt.figure(2)
for pic in range(c.shape[0]):
    plt.plot(10*x, c[pic,:], label='{0}'.format(pic) )

plt.title(r'Loss of net in vicinity of trained weigths $\omega_i$')
plt.xlabel(r'$\delta \omega_i$')
plt.ylabel(r'Loss$(\omega +\delta \omega_i)$')
plt.legend()