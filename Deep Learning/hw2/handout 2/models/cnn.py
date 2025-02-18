# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)​

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN(object):

    """
    A simple convolutional neural network

    Here you build implement the same architecture described in Section 3.3
    You need to specify the detailed architecture in function "get_cnn_model" below
    The returned model architecture should be same as in Section 3.3 Figure 3
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Your code goes here -->
        # self.convolutional_layers (list Conv1d) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)
        # <---------------------

        self.conv1 = Conv1d(in_channels=24, out_channels=56, kernel_size=5, stride=1)
        self.conv2 = Conv1d(in_channels=56, out_channels=28, kernel_size=6, stride=2)
        self.conv3 = Conv1d(in_channels=28, out_channels=14, kernel_size=2, stride=2)

        self.flatten = Flatten()
        in_features = 420
        out_features = 10
        self.linear_layer = Linear(in_features, out_features)

        self.convolutional_layers = [
            self.conv1,
            self.conv2,
            self.conv3,
        ]
        self.layers = [
            self.conv1,
            Tanh(),
            self.conv2,
            ReLU(),
            self.conv3,
            Sigmoid(),
            self.flatten,
            self.linear_layer
        ]

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, num_input_channels, input_width)
        Return:
            Z (np.array): (batch_size, num_linear_neurons)
        """

        # Your code goes here -->
        # Iterate through each layer
        # <---------------------
        
        L = len(self.layers)
        for i in range(L):
            layer = self.layers[i]
            A = layer.forward(A)
            print(f"Forward {layer.__class__.__name__} output shape: {A.shape}")  # Debugging

        # Save output (necessary for error and loss)
        self.Z = A

        return self.Z

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion.forward(self.Z, labels).sum()
        grad = self.criterion.backward()

        # Your code goes here -->
        # Iterate through each layer in reverse order
        # <---------------------

        L = len(self.layers)
        for layer in reversed(self.layers):
            print(f"Backward {layer.__class__.__name__} input shape: {grad.shape}")  # Debugging

            grad = layer.backward(grad)

        return grad

    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.dLdW.fill(0.0)
            self.convolutional_layers[i].conv1d_stride1.dLdb.fill(0.0)

        self.linear_layer.dLdW.fill(0.0)
        self.linear_layer.dLdb.fill(0.0)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.W = (self.convolutional_layers[i].conv1d_stride1.W -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdW)
            self.convolutional_layers[i].conv1d_stride1.b = (self.convolutional_layers[i].conv1d_stride1.b -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdb)

        self.linear_layer.W = (
            self.linear_layer.W -
            self.lr *
            self.linear_layer.dLdW)
        self.linear_layer.b = (
            self.linear_layer.b -
            self.lr *
            self.linear_layer.dLdb)

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False
