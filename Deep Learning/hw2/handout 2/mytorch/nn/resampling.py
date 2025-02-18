import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.k = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        N, C, Win = np.shape(A)
        Wout = self.k * (Win - 1) + 1

        # add in intermediate corresponding values of A
        # ex: k = 2, A = [1, 2, 3], Z = [0, 0, 0] -> Z = [1, 0, 2, 0, 3]
        Z = np.zeros((N, C, Wout))
        Z[..., ::self.k] = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # remove the extra zeros added in the forward pass
        dLdA = dLdZ[..., ::self.k]
        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.k = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        N, C, self.Win = A.shape
        Z = A[..., ::self.k]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        N, C, _ = dLdZ.shape

        dLdA = np.zeros((N, C, self.Win))
        dLdA[..., ::self.k] = dLdZ
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.k = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        N, C, Hin, Win = A.shape

        Hout = self.k * (Hin - 1) + 1
        Wout = self.k * (Win - 1) + 1

        Z = np.zeros((N, C, Hout, Wout))
        Z[:, :, ::self.k, ::self.k] = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = dLdZ[:, :, ::self.k, ::self.k]
        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.k = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N, C, self.Hin, self.Win = A.shape
        Z = A[:, :, ::self.k, ::self.k]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C, _, _ = dLdZ.shape
        dLdA = np.zeros((N, C, self.Hin, self.Win))
        dLdA[:, :, ::self.k, ::self.k] = dLdZ
        return dLdA
