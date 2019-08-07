#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:38:43 2019

@author: john
"""

import tkinter as tk
import numpy as np
import neural_network as nn
import pickle

class ExampleApp(tk.Tk):
    """
    This class defines a g.u.i. which presents the user with a canvas
    where the user can draw an image. The image can then be converted to
    a 28 by 28 numpy array of zeros and ones and fed into a neural network
    trained to recognise digits
    """
    
    
    
    def __init__(self):
        # Initialise canvas
        tk.Tk.__init__(self)
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        
        self.canvas = tk.Canvas(self, width=280, height=280, 
                                bg = "black", cursor="cross")
        
        self.canvas.pack(side="top", fill="both", expand=True)
        
        self.button_print = tk.Button(self, text = "Feed to network", 
                                      command = self.feed_network)
        
        self.button_print.pack(side="top", fill="both", expand=True)
        
        self.button_clear = tk.Button(self, text = "Clear", 
                                      command = self.clear_all)
        
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)
        
        # Load the trained neural network
        f = open('pickled_neural_net', 'rb')
        
        layer_sizes = pickle.load(f)
        biases = pickle.load(f)
        weights = pickle.load(f)
        
        f.close()
        
        # create neural network object with loaded weights and biases
        self.net = nn.NeuralNetwork(layer_sizes)
        self.net.set_weights_biases(weights, biases)
        


    def clear_all(self):
        self.canvas.delete("all")



    def feed_network(self):
        if self.points_recorded:
            self.points_recorded.pop()

        # Create a numpy array containing the drawn image
        image = np.zeros( (28,28) )
        for x,y in self.points_recorded:
            image[ y//10, x//10 ] = 1
            
        # Reshape the array into a vector for the network    
        image = image.reshape(28*28,1)
        
        # Feed the network the image and print the output
        num = self.net.feed_forward(image).argmax()
        print(num)
        
        self.points_recorded[:] = []



    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y



    def draw_from_where_you_are(self, event):

        self.x = event.x
        self.y = event.y
        self.canvas.create_line(self.previous_x, self.previous_y, 
                                self.x, self.y,fill="yellow")
        self.points_recorded.append( (self.x,self.y) )          
        self.previous_x = self.x
        self.previous_y = self.y



if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()