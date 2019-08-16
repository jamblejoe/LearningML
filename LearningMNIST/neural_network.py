#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:58:33 2019

@author: john
"""

import numpy as np
import random



#### Define the quadratic and cross-entropy cost functions
class QuadraticLoss(object):

    @staticmethod
    def fn(a, y):
        """
        Return the squared Euclidean distance between the output a and
        the desired output y. Result is divided by two to avoid extra factors
        of 2 in derivative
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(a, y):
        """
        Return the gradient of the Quadractic loss function w.r.t. z^L
        (i.e. the input to the last sigmoid function)
        """
        return (a-y) * a * (1-a)



class CrossEntropyLoss(object):

    @staticmethod
    def fn(a, y):
        """
        Return the cross entropy loss associated with an output a and 
        desired output y.
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y):
        """
        Return the gradient of the cross entropy loss w.r.t. z^L.
        """
        return (a-y)



# Define the neural network class
class NeuralNetwork:
    """
    This class defines methods which can be used to create a neural network
    and train it using the stochastic gradient method.
    """

    
    
    def __init__(self, layer_sizes, lossFn=CrossEntropyLoss):
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
        self.layer_sizes = layer_sizes
        self.lossFn = lossFn
        
        self.biases = [ np.random.randn(n,1) for n in layer_sizes[1:] ]
        
        # The weights are generated using a gaussian with a standard
        # deviation of sqrt(m) where m is the number of coloumns of the
        # weight matrix. This avoids initial weights that saturate a layer
        # which slows down learning
        self.weights= [ np.random.randn(n,m)/np.sqrt(m)
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
        # [ note: d(sig)/dz = sig(z)*(1-sig(z)) ]
        #delta = (a - target) * (a*(1-a))
        delta = self.lossFn.delta(a, target)
        
        # gradients w.r.t. weight matrices and bias vectors will be stored here
        nWs = [ np.dot(delta, activations[-2].T) ]
        nbs = [ delta ]
        
        '''
        Here we begin propagating backward, or pulling back, the gradient
        vector calculated above to compute the gradient w.r.t. weights and 
        biases associated with previous layers.
        '''
        for W, sz, a in zip( reversed(self.weights[1:]), 
                            reversed(activations[:-1]),
                            reversed(activations[:-2]) ):
            
            # d(sig)/dz = sig(z)*(1-sig(z))
            delta = W.T.dot(delta) * (sz*(1-sz))
            
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
    
    
    
    def update(self, mini_batch, eta, lmbda):
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
         
        # The eta*lmbda/n term comes from L2 regularisation, so the algorithm
        # is actually minimising loss+lmbda*sum w**2/2 in an attempt to avoid 
        # overfitting.
        self.weights = [(1-eta*lmbda)*W - (eta/n)*nW 
                        for W, nW in zip( self.weights, nabla_W)]
        self.biases  = [b - (eta/n)*nb 
                        for b, nb in zip( self.biases, nabla_b)]
        
        
        
    def stochastic_gradient_decent(self, training_data, epochs, 
                                   mini_batch_size, eta, lmbda=0.0,
                                   test_data = None,
                                   get_training_loss=False,
                                   get_training_acc=False,
                                   get_test_loss=False,
                                   get_test_acc=False):
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
        training_loss = [] 
        training_acc = []
        test_loss = [] 
        test_acc = [] 
        
        
        for j in range(epochs):
            
            random.shuffle(training_data)
            
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update(mini_batch, eta, lmbda)
                
                
            print("Epoch {0} training complete\n".format(j))
            
            if get_training_loss:
                loss = self.loss_function(training_data, lmbda)
                training_loss.append(loss)
                print("Loss on training data: {}".format(loss) )
                
            if get_training_acc:
                accuracy = self.evaluate(training_data)
                training_acc.append(accuracy/n)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n) )
                
            if get_test_loss:
                loss = self.loss_function(test_data, lmbda)
                test_loss.append(loss)
                print("Loss on test data: {}".format(loss) )
                
            if get_test_acc:
                accuracy = self.evaluate(test_data)
                test_acc.append(accuracy/n_test)
                print("Accuracy on test data: {} / {}".format(
                    accuracy, n_test) )
            
            print('\n')
        return test_loss, test_acc, training_loss, training_acc
                
    
    
    def evaluate(self, test_data):
        """
        This method returns the number of correct ouputs the network returns
        given the labeled data in test_data
        """
        test_results = [int(self.feed_forward(x).argmax() == y.argmax())
                        for x, y in test_data]
        
        return sum(test_results)
    
    
    
    def loss_function(self, data, lmbda):
        """
        This method returns the loss function evaulated using the labeled
        images in the array data
        """
        loss = 0.0
        
        for x, y in data:
            a = self.feed_forward(x)
            loss += self.lossFn.fn(a, y)
        
        if lmbda:
            loss += 0.5*(lmbda)*sum(
                    np.linalg.norm(w)**2 for w in self.weights)
        
        return loss/len(data)
    
    

# Functions for computing the sigmoid function and its derivative:
def sigmoid(z):
    return 1/( 1 + np.exp(-z) )


def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig*(1-sig)