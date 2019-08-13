#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:35:03 2019

@author: john
"""

import gzip
import matplotlib.pyplot as plt
import numpy as np



def plotImages(data, N, p=0, f=1):
    """
    This function plots a gray-scale image of the images stored in data.
    The first image in data is displayed first. Clicking on the image should
    display the next image in data. Right clicking should display the previous
    image
    """
    
    pic = p
    
    def onclick(event):
        nonlocal pic
        
        event.canvas.figure.clear()
        
        if event.button == 1:
            pic += 1
        elif event.button == 3:
            pic -= 1
        
        if pic < 0:
            pic = N-1
        if pic >= N:
            pic = 0
        
        event.canvas.figure.gca().imshow(np.asarray(data[pic]).squeeze(), 
                                         cmap='gray')
        plt.title('i = {0}, (target = {1})'.format(pic, labels[pic]) )
        event.canvas.draw()
    
    
    fig = plt.figure(f)
    fig.canvas.mpl_connect('button_press_event', onclick )

    plt.imshow(np.asarray(data[pic]).squeeze(), cmap='gray')
    plt.title('i = {0}, (target = {1})'.format(pic, labels[pic]) )


# These variables store the size if the images and number of images to display
image_size = 28
num_images = 20

# create a numpy array containing images read from MNIST data set: 
f = gzip.open('Data/train-images-idx3-ubyte.gz','r')
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)
f.close()

# create a numpy array containing labels read from MNIST data set: 
f = gzip.open('Data/train-labels-idx1-ubyte.gz','r')
f.read(8)
labels = np.frombuffer( f.read(num_images), dtype=np.uint8 )
f.close()

# plot the images
plotImages(data, num_images)