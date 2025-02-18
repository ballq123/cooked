import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        self.k = self.kernel_size

        self.N, _, self.Hin, self.Win = A.shape
        self.Hout = self.Hin - self.k + 1
        self.Wout = self.Win - self.k + 1

        Z = np.zeros((self.N, self.out_channels, self.Hout, self.Wout))
        for i in range(self.Hout):
            for j in range(self.Wout):
                aSub = A[:, :, i:i+self.k, j:j+self.k]
                Z[:, :, i, j] = np.tensordot(aSub, self.W, axes=([1, 2, 3], [1, 2, 3]))
        
        Z += self.b[None, :, None, None]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        self.dLdW = np.zeros_like(self.W)
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))
        dLdA = np.zeros_like(self.A)
        self.k = self.kernel_size

        for i in range(self.Hout):
            for j in range(self.Wout):
                aSub = self.A[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                self.dLdW += np.tensordot(dLdZ[:, :, i, j], aSub, axes=(0, 0))
                dLdA[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += np.tensordot(dLdZ[:, :, i, j], self.W, axes=(1, 0))
            
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() instance
        self.Cin = in_channels
        self.Cout = out_channels
        self.k = kernel_size
        self.w = weight_init_fn
        self.b = bias_init_fn
        
        self.conv2d_stride1 = Conv2d_stride1(self.Cin, self.Cout, self.k, self.w, self.b)
        # again, in this case stride = downsampling_factor
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=0)

        # Call Conv2d_stride1
        zConv = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(zConv)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample2d backward
        dLdZDown = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZDown)

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return dLdA
