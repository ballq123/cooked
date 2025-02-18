import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel
        self.k = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        self.N, self.Cin, self.Hin, self.Win = A.shape
        self.Hout = self.Hin - self.k + 1
        self.Wout = self.Win - self.k + 1

        # store max indices as tuples
        self.maxIdxs = np.zeros((self.N, self.Cin, self.Hout, self.Wout, 2))
        Z = np.zeros((self.N, self.Cin, self.Hout, self.Wout))
        for i in range(self.N):  
            for j in range(self.Cin): 
                for k in range(self.Hout): 
                    for l in range(self.Wout): 
                        aSub = self.A[i, j, k:k+self.k, l:l+self.k]
                        maxVal = np.max(aSub)
                        argM = np.argmax(aSub)
                        maxIdx = np.unravel_index(argM, aSub.shape)
                        Z[i, j, k, l] = maxVal
                        self.maxIdxs[i, j, k, l] = (k + maxIdx[0], l + maxIdx[1])

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A)

        for i in range(self.N):  
            for j in range(self.Cin):  
                for k in range(self.Hout):  
                    for l in range(self.Wout):  
                        max_k, max_l = self.maxIdxs[i, j, k, l]
                        # turn into ints because max_k/l are stored as float64s
                        dLdA[i, j, int(max_k), int(max_l)] += dLdZ[i, j, k, l]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel
        self.k = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        self.N, self.Cin, self.Hin, self.Win = A.shape
        self.Hout = self.Hin - self.k + 1
        self.Wout = self.Win - self.k + 1

        Z = np.zeros((self.N, self.Cin, self.Hout, self.Wout))
        for i in range(self.N):  
            for j in range(self.Cin):  
                for k in range(self.Hout):  
                    for l in range(self.Wout):  
                        aSub = self.A[i, j, k:k+self.k, l:l+self.k]
                        Z[i, j, k, l] = np.mean(aSub)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A)

        for i in range(self.N):  
            for j in range(self.Cin):  
                for k in range(self.Hout):  
                    for l in range(self.Wout):  
                        grad = dLdZ[i, j, k, l] / (self.k ** 2) 
                        dLdA[i, j, k:k+self.k, l:l+self.k] += grad

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1 & Downsample2d
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        aMax = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(aMax)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        zDown = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(zDown)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        aMean = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(aMean)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        zDown = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(zDown)
        return dLdA
