# LearningML

The neural_network.py file defines a NeuralNetwork object which has methods for creating and training a neural network.
This code was written while I was reading Michael Nielsen's online book: http://neuralnetworksanddeeplearning.com/

The learning_MNIST.py file contains code which reads the MNIST data downloaded from http://yann.lecun.com/exdb/mnist/ and
saved in the Data folder. It then creates and trains a network with with 784 input neurons, 30 neurons in a single hidden 
layer and 10 output neurons. The weights and biases of the trained network are then saved to the file pickled_neural_net.

The draw_input script presents the user with a canvas to draw images which are then converted to a 28 by 28 numpy array
and fed into the trained network stored in pickled_neural_net. The output of the neural net is printed to the console.

The iew_data.py file has code for viewing a selected number of images from the MNIST data set.
