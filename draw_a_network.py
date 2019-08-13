#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:56:10 2019

@author: john
"""
import matplotlib.pyplot as plt

# Define the number of and size of the layers of the network 
layer_sizes = [7,4,5,2]

# The radius of neurons to be drawn
neuron_radius = 0.03

# Set up figure and axes
plt.figure(figsize=(8,8))
fig = plt.gcf()
ax = fig.gca()
plt.axis('off')

numl = len(layer_sizes)

# Draw the edges of the network
for i, (numn, numa) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    for j in range(numn):
        
        # The x,y corrinates of a neuron
        xn, yn = (i+1)/(numl+1), (j+1)/(numn+1)
        
        for k in range(numa):
            
            # The x,y corrinates of a connecting neuron in the next layer
            xa, ya = (i+2)/(numl+1), (k+1)/(numa+1)
            
            # Draw the edge
            weight = plt.Line2D([xn,xa], [yn,ya], color='gray', lw=1)
            ax.add_artist(weight)


# Draw the neurons of the network
for i, numn in enumerate(layer_sizes):
    for j in range(numn):
        
        # The x,y corrinates of a neuron
        x, y = (i+1)/(numl+1), (j+1)/(numn+1)
        
        # Draw the neuron
        neuron = plt.Circle((x,y), neuron_radius, color='darkkhaki', 
                            ec='black', fill=True, zorder=2)
        ax.add_artist(neuron)
        