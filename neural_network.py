#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:58:33 2019

@author: john
"""

import numpy as np
import random


class NeuralNetwork:
    """
    This class defines methods which can be used to create a neural network
    and train it using the stochastic gradient method.
    """

    
    
    def __init__(self, layer_sizes):
        """
        Here, the variable layer_sizes is expected to be a list of positive
        integers defining the number of neurons in each layer of the network.
        The length of the list defines the number of layers. The first element
        in the list defines the size of the input layer, the second number
        gives the size of the first hidden layer, etc. and the last number 
        gives the size output layer.
        
        Matrices of network weights and vectors of biases are then created, at
        random, for the network defined by layer_sizes using a Gaussian number
        generator. These are intended to give the network an initial
        configuration and to be optimised by the training algorithm
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes= layer_sizes
        
        self.biases = [ np.random.randn(n,1) for n in layer_sizes[1:] ]
        self.weights= [ np.random.randn(n,m)
                        for n, m in zip(layer_sizes[1:], layer_sizes[:-1]) ]
    
    
    
    def set_weights_biases(self, w, b):
        """
        This method can be used to set the weights and biases to predefined
        values. If, for example, a network was trained and its weights and
        biases were saved to a file, this method can be used to recreate
        the trained network from that file.
        """
        self.weights = w
        self.biases = b
        
    
    
    def feed_forward(self, arg):
        """
        This method returns the output vector of the network after it is
        fed the input vector arg.
        """
        for W, b in zip( self.weights, self.biases ):
            arg = sigmoid( W.dot(arg) + b )
        
        return arg
    
    
    
    def back_propagation(self, a, target):
        """
        This method uses back propagation to calculate the gradient of the 
        Euclidean distance between the output vector of the network, after it 
        has been fed the vector 'a', and the desired output vector 'target'. 
        """
        
        '''
        The input vector 'a' is first fed into the network to compute the
        output. While computing the output, relevant terms for calculating
        the gradient are stored in the following lists:
        '''
        activations = [ a ]
        zeds = []
        
        for W, b in zip( self.weights, self.biases ):
            z = W.dot( a ) + b
            a = sigmoid( z )
            
            activations.append( a )
            zeds.append( z )

        # gradient of Euclidean distance w.r.t. to z for the output layer:
        delta = (a - target) * sigmoid_prime(z)
        
        # gradients w.r.t. weight matrices and bias vectors will be stored here
        nWs = [ np.dot(delta, activations[-2].T) ]
        nbs = [ delta ]
        
        '''
        Here we begin propagating backward, or pulling back, the gradient
        vector calculated above to compute the gradient w.r.t. weights and 
        biases associated with previous layers.
        '''
        for W, z, a in zip( reversed(self.weights[1:]), 
                            reversed(zeds[:-1]),
                            reversed(activations[:-2]) ):
            
            delta = W.T.dot(delta) * sigmoid_prime( z )
            
            nWs.append( np.dot(delta, a.T) )
            nbs.append( delta )
        
        '''
        The gradient for the weights and biases associated with the last
        later was calculated and appended to nWs, nbs first and the gradient
        associated with those for the first layer were calculate last. We
        should reverse these lists to match the ordering of the weights and
        biases lists
        '''
        nWs.reverse()
        nbs.reverse()
        
        return (nWs, nbs)
    
    
    
    def update(self, mini_batch, eta):
        """
        This method will update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The mini_batch is a list of tuples (x, y) x being an input vector
        and y being the desired or target output. Eta is the learning rate.
        """
        nabla_W = [np.zeros(W.shape) for W in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        n = len(mini_batch)
        
        for x, y in mini_batch:
            nWs, nbs = self.back_propagation(x,y)
            
            nabla_W = [nW + nWx for nW, nWx in zip(nabla_W, nWs) ]
            nabla_b = [nb + nbx for nb, nbx in zip(nabla_b, nbs) ]
            
        self.weights = [W - (eta/n)*nW 
                        for W, nW in zip( self.weights, nabla_W)]
        self.biases  = [b - (eta/n)*nb 
                        for b, nb in zip( self.biases, nabla_b)]
        
        
        
    def stochastic_gradient_decent(self, training_data, epochs, 
                                   mini_batch_size, eta, test_data=None):
        """
        This method will train the neural network using mini-batch stochastic
        gradient descent.  The argument training_data is a list of tuples
        (x, y) x being an input vector and y being the desired or target 
        output. 
        
        The variable epochs defines the number of epochs to train the network
        for and mini_batch_size defines the size of the batches the training
        data should be divided into. Eta is the learning rate
        
        If test_data is provided then the network will be evaluated against 
        the test data after each epoch, and progress printed out. 
        This is useful for tracking progress, but slows things down.
        """
        if test_data: 
            n_test = len(test_data)
        
        n = len(training_data)
        
        for j in range(epochs):
            
            random.shuffle(training_data)
            
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)
                
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, 
                      self.evaluate(test_data), n_test) )
            else:
                print( "Epoch {0} complete".format(j) )
                
    
    
    def evaluate(self, test_data):
        """
        This method returns the number of correct ouputs the network returns
        given the labeled data in test_data
        """
        test_results = [int(self.feed_forward(x).argmax() == y)
                        for x, y in test_data]
        return sum(test_results)
    
    

# Functions for computing the sigmoid function and its derivative:
def sigmoid(z):
    return 1/( 1 + np.exp(-z) )

def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig*(1-sig)