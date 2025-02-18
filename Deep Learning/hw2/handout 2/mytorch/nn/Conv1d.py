# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        self.k = self.kernel_size
        self.N, _, self.Win = A.shape
        self.Wout = self.Win - self.k + 1
        Z = np.zeros((self.N, self.out_channels, self.Wout))

        for i in range(self.Wout):
            aSub = A[:, :, i:i+self.k]
            Z[:, :, i] = np.tensordot(aSub, self.W, axes=([1, 2], [1, 2]))

        Z += self.b[:, None]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        self.k = self.kernel_size
        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.k))
        self.dLdb = np.sum(dLdZ, axis=(0, 2))
        dLdA = np.zeros((self.N, self.in_channels, self.Win))

        for i in range(self.Wout):
            self.dLdW += np.tensordot(dLdZ[:, :, i], self.A[:, :, i:i+self.k], axes=(0, 0))
        for i in range(self.Wout):
            dLdA[:, :, i:i+self.k] += np.tensordot(dLdZ[:, :, i], self.W, axes=(1, 0))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() instance
        Cin = in_channels
        Cout = out_channels
        self.k = kernel_size
        w = weight_init_fn
        b = bias_init_fn
        
        self.conv1d_stride1 = Conv1d_stride1(Cin, Cout, kernel_size, w, b)
        # in this case, stride = downsampling_factor
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant', constant_values=0)

        # Call Conv1d_stride1
        zConv1d = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(zConv1d)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZDown = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZDown)

        # Unpad the gradient
        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:-self.pad]
        return dLdA
