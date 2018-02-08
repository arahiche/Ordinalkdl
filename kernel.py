#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#
#   This code computes different Kernel functions.
#
#   K(Xi, Xj) = K(Xi).K(Xj)
#
#   The code has been tested with Ubuntu 16.04 x86_64.
#
#
#
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy
from scipy import sparse



class kernelMatrix(object):
    """
    Class for some kernel functions
    Arguments:
                kernel --> Kernel function
                Param  --> vector with the parameter information


    Returns: Kernel matrix

    """

    def linear(self, X, Y):
        """
        Polynomial Kernel
        """

        kernelMatrix = np.dot( X, Y.T )

        return kernelMatrix


    def polynomial(self, X, Y, bias, power):
        """
        Polynomial Kernel
        """

        kernelMatrix = (np.dot( X, Y.T ) + bias) ** power

        return kernelMatrix


    def gaussian(self, X, Y, gamma):
        """
        Gaussian Kernel between X and Y
        K(X, Y) = exp(-gamma *||X-Y||^2)
        """
        kernelMatrix =np.zeros((X.shape[0], Y.shape[0]))

        for i in range(Y.shape[0]):
            kernelMatrix[:][i] = np.exp(-gamma*np.sum((X - np.dot(Y[:][i], np.ones((1, X.shape[0])))) ** 2))
        return kernelMatrix




    def rbf(self, X, Y, gamma):
        """
        RBF Kernel between X and Y
        K(X, Y) = exp(-gamma *||X-Y||^2)
        """
        # another implementation
        # kernelMatrix = sparse.lil_matrix( (X.shape[0], Y.shape[0]) )
        print "calculate the kernel matrix ", X.shape, Y.shape
        kernelMatrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                kernelMatrix[i, j] = np.exp(-gamma * np.linalg.norm(x - y) ** 2)
        return kernelMatrix













