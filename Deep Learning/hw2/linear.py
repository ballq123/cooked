import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros (Hint: check np.zeros method)
        Read the writeup (Hint: Linear Layer Section Table) to identify the right shapes for `W` and `b`.
        
        Cin = # input features
        Cout = # output features
        W = (Cout, Cin)
        b = (Cout, 1)
        """
        self.debug = debug
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)

        Read the writeup (Hint: Linear Layer Section) for implementation details for `Z`

        A = (N, Cin)
        N = batch size
        """
        self.A = A
        self.N = np.shape(A)[0] # store the batch size parameter of the input A

        # Think how can `self.ones` help in the calculations and uncomment below code snippet.
        self.ones = np.ones((self.N, 1))

        Z = (self.A @ np.transpose(self.W)) + (self.ones @ np.transpose(self.b))
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        Read the writeup (Hint: Linear Layer Section) for implementation details below variables.

        dLdA = (N, Cin)
        dLdW = (Cout, Cin)
        dLdb = (Cout, 1)
        """
        dLdA = dLdZ @ self.W  
        self.dLdW = np.transpose(dLdZ) @ self.A  
        self.dLdb = np.transpose(dLdZ) @ self.ones  

        if self.debug:
            self.dLdA = dLdA

        return dLdA
