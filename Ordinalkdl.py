#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#  
#   This program is a python implementation of the Kernel Discriminant Learning for Ordinal Regression method. For more details see [1].  This implementation is
#   inspired by the Matlab implementation written by P.A. Gutiérrez et al [2], which is publicly available here : https://github.com/ayrna/orca.  
#
#   Author: Abderrahmane rahiche <arahiche@yahoo.com>
#   
#   The code has been tested with Ubuntu 16.04 x86_64.
#
#   [1]: B.-Y. Sun, J. Li, D. D. Wu, X.-M. Zhang, and W.-B. Li, “Kernel discriminant learning for ordinal regression,”
#        IEEE Trans. Knowl. Data Eng., vol. 22, no. 6, pp. 906–910, 2010.
#   [2]: P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero, F. Fernández-Navarro and C. Hervás-Martínez (2016),
#	"Ordinal regression methods: survey and experimental study",
#	IEEE Transactions on Knowledge and Data Engineering. Vol. 28. Issue 1
#
#   
#       
"""

from __future__ import division



import numpy as np
import kernel 
from scipy import sparse
import time, math
from cvxopt import matrix, solvers
import quadprog
import sys


class Ordinalkdl(object):
    """
    Ordinalkdl method (Kernel Discriminant Learning for Ordinal Regression)
    """

    def __init__(self, Kernel, Param, OptMethod, C):
        """
        Arguments:
                kernel --> Kernel function
                Param  --> vector with the parameter information
                OptMethod  --> optimization method


       """

        # verification of The kernel value
        # print "np.shape(Param)[0]", np.shape(Param)[0]


        self.kernelParam = Param
        self.KernelType = Kernel
        self.C = C

        if Kernel == "gaussian":
            self.KernelType = "gaussian"
            if np.shape(self.kernelParam)[0] == 2:
                self.gamma = self.kernelParam[0]
		self.u = self.kernelParam[1]
		print self.gamma, self.u
            else:
                print "The Gaussian function takes only one parameter"
        elif Kernel == "polynomial":
            self.KernelType = "polynomial"
            if np.shape(self.kernelParam)[0] == 2:
                self.bias  = self.kernelParam[0]
                self.power = self.kernelParam[1] 
            else:
                print "The Polynomial function takes two parameters !"
        elif Kernel == "linear":
            self.KernelType = "linear"
            if np.shape(self.kernelParam)[0] == 1:
                pass
            else:
                print "No parameters are needed for the linear kernel"
        else:
            print "Invalid value for the kernel. Please use one of these values : Gaussian, Polynomial, Linear "


        # verification of the optimisation method

        if OptMethod not in ["quadprog", "cvx"]:
            print "Invalid value for variable OptMethod."
        else:
            self.OptMethod = OptMethod




    def Learn(self, Train_feat, Train_labels):
        """
        Arguments:
                Train_feat    --> vector of training features
                Train_labels  --> vector of training labels

        Returns: model

        """

        n_samples, n_features = np.shape(Train_feat)
        nbr_classes = len(np.unique(Train_labels))
        # print nbr_classes
        class_means = np.zeros(shape=(nbr_classes, n_samples)) # mean of classes
        # print "class_means", class_means
        kernelMatrix = kernel.kernelMatrix()

        # Matkernel = kernelMatrix(Train_feat, Train_feat, self.KernelType, self.kernelParam)
        # print "Matkernel", Matkernel
        # Calculate the Kernel matrix (Gram matrix)

        if self.KernelType == "gaussian":
            # gamma = self.gamma
            # print gamma, Train_feat
            # print inspect.getargspec(k.RBF)
            Matkernel = kernelMatrix.rbf(Train_feat, Train_feat, self.gamma)

        elif self.KernelType == "polynomial":
            Matkernel = kernelMatrix.polynomial(Train_feat, Train_feat, self.bias, self.power)

        elif self.KernelType == "linear":
            Matkernel = kernelMatrix.linear(Train_feat, Train_feat)

        # print "Matkernel ok" #, 
        # return Matkernel
        ###############################################################################
        # Calculate the H matrix and the mean of classes
        ###############################################################################

        # H = sum_{K=1}^{K}(Matrixkernel_{k}*(I - 1_{NK})*Matrixkernel_{k}.T)
        #
        # count the number of elements for each class
        NK = np.histogram(Train_labels, bins = range(1, nbr_classes + 2)) # NK[0] contains the hist values
        # print "NK ok" #, NK
        # sparse matrix
        H = sparse.csr_matrix((n_samples, n_samples))
        #H = np.zeros(shape=(n_samples, n_samples))
        # print "H = \n", H
        for iclass in range(1, nbr_classes + 1):
            # print iclass
            #
            index = ([i for i, t in enumerate(Train_labels) if t == iclass])
            # print "index = \n", index
            kernelMatrix_K = Matkernel[:, index]
            # print "kernelMatrix_K", kernelMatrix_K.shape, kernelMatrix_K
            # kernelMatrix_K = [[row[i] for row in Matkernel] for i in (index)]
            # print "kernelMatrix_K = \n", kernelMatrix_K.shape, kernelMatrix_K
            class_means[iclass -1][:] = np.mean(kernelMatrix_K, axis=1)
            I_k = np.eye(NK[0][iclass -1])
            eyeNK = np.ones((NK[0][iclass -1], NK[0][iclass -1]))/NK[0][iclass -1] # -1 because python is 0-th indexing
            # print "class_means = \n", class_means.shape, class_means
            #  print "I_k = \n", I_k.shape, I_k
            # print "1NK = \n", eyeNK.shape, eyeNK
            H += np.dot(kernelMatrix_K, (I_k - eyeNK)).dot(np.transpose(kernelMatrix_K)) #*np.transpose(kernelMatrix_K)


        print "kernelMatrix_K = ", kernelMatrix_K.shape, " class_means_k = ", class_means.shape, " I_k = ", I_k.shape, " 1NK = ", eyeNK.shape, " H", H.shape

        print "\n Hessian matrix done ...." # , H
        print "\n class_means_k done .... " # , class_means

        ###############################################################################
        # Calculate the P matrix of the optimization problem
        # To avoid the singularity problem of the matrix H (ill-conditioning problem) we use the regularization method.
        # Here the solution is simply to adding a small bias "u" to the diagonal might fix the problem, so H = H +u*I
        # Other methods like SVD can be used also.

        u = self.u # default = 0.001.  It should be tunned like the other parameters
        H = H + u * np.eye( n_samples )
        H_inv = np.linalg.inv( H )
        # print H_inv

        J = np.zeros((nbr_classes-1, nbr_classes-1))
        # C = np.zeros((nbr_classes-1, 1))
        # A = -1*np.ones( (nbr_classes - 1, nbr_classes - 1) )
        # b = np.zeros( (nbr_classes - 1, 1) )
        # E = np.ones( (1, nbr_classes - 1) )
        sigma = np.zeros((1, n_samples))
        # print "J", J

        for m in range(nbr_classes - 1):
            for n in range(m, nbr_classes -1):
                J[m][n] = np.dot(np.dot( (class_means[m + 1][:] - class_means[m][:]), H_inv ), (class_means[n + 1][:] - class_means[n][:] ))
                # make sure J is symmetric
                J[n][m] = J[m][n]

        # print 'J ok', J.shape #, J

        # solve the following objective function
        # obj = 0.5 * x'*P*x
        # s.t x >= 0
        # and sum(x_k) = C
        #
        # Setting the boundaries and initial conditions
        # lower bound
        lb = matrix(np.zeros((nbr_classes -1, 1)))
        # upper bound
        ub = matrix(np.inf*np.ones((nbr_classes -1, 1)))
        # initial values
        initvals  = matrix(np.zeros((nbr_classes -1, 1)))

        #
        # Define QP parameters

        C = self.C
        P = 2.*matrix( J , tc='d')
        q = matrix( np.zeros( (nbr_classes-1, 1) ) , tc='d')# [0.0] * (nbr_classes-1) )
        # n = J.shape[1]  # This is for readability only
        G = -1 * matrix( np.identity( nbr_classes-1 ) , tc='d')
        #G = np.vstack( [-1 * np.ones( (nbr_classes - 1, nbr_classes - 1) ), -np.eye( n ), np.eye( n )] )
        # h = np.hstack( [np.zeros( (nbr_classes - 1, 1) ), -lb, ub] )
        h = matrix( np.zeros((nbr_classes-1, 1)) , tc='d')
        # A = None
        A = matrix(np.ones( (1, nbr_classes - 1) ), tc='d')
        b = matrix(C, (1,1), tc='d') #*matrix(np.ones( (nbr_classes - 1, 1) ))

        # print sol['x']

        if self.OptMethod == "quadprog":
            Jmin = quadprog.solve_qp(P, [q, G, h, A, b])
        elif self.OptMethod == "cvx":
            Jmin = solvers.qp(P, q, G, h, A, b, initvals) #(P, matrix(C), E, d, vlb, vub, initvals)

        # print alpha['x']
        alpha =Jmin['x']
        # print 'alpha ok' #, alpha

        # Claculate W = 0.5 * H^{-1} * Sum_{k=1}^{K-1}(alpha_{k}*(M_{k+1}-M_{k}))
        # We first calculate sigma = Sum_{k=1}^{K-1}(alpha_{k}*(M_{k+1}-M_{k}))
        for iclass in range(1, nbr_classes):
            # print iclass, alpha[iclass -1]
            sigma += np.dot(alpha[iclass -1],  (class_means[iclass][:] - class_means[iclass - 1][:]))
            # np.dot( class_means[m + 1][:]
        # print "sigma", sigma.shape #, sigma

        W = 0.5 * H_inv * (sigma.reshape((sigma.shape[1], 1)))
        # print "W", W.shape #, W

        # Calculate bk
        bk = np.zeros((nbr_classes-1, 1))
        for iclass in range(1, nbr_classes):
            bk[iclass - 1] = 0.5*np.dot(W.reshape((1, W.shape[0])), (class_means[iclass][:] + class_means[iclass - 1][:]))
        print "bk ok" #, bk
        model={}
        model['name'] = 'KDLOR'
        model['infokernel'] = self.KernelType
        model['kernelparam'] = self.kernelParam
        model['infoptmethod'] = self.OptMethod
        model['W'] = W
        model['bk']= bk
        return model


    def Test(self, Train_feat, Test_feat, model):
        kernelMatrix = kernel.kernelMatrix()
        nbr_class = model['bk'].shape[0] + 1
        KernelParam = model['kernelparam']
        KernelType = model['infokernel']
        print 'nbr_class', nbr_class, 'KernelType', KernelType, 'KernelParam', KernelParam

        # Matkernel = kernelMatrix( Test_feat, Test_labels, model.kernelType, model.parameters.k );
        # projected = model.projection'*kernelMatrix2;
        # Calculate the Kernel matrix (Gram matrix)

        if KernelType == "gaussian":
            # gamma = self.gamma
            # print gamma, Train_feat
            # print inspect.getargspec(k.RBF)
            Matkernel2 = kernelMatrix.rbf(Train_feat, Test_feat, KernelParam[0])

        elif KernelType == "polynomial":
            Matkernel2 = kernelMatrix.polynomial(Train_feat, Test_feat, KernelParam[0], KernelParam[1])

        elif KernelType == "linear":
            Matkernel2 = kernelMatrix.linear(Train_feat, Test_feat)

        # print "Matkernel2", Matkernel2.shape #, Matkernel2

        # Calculate the projected data f( x ) = max {W*x - b(k) < 0} or {W*x - b_(K-1) > 0}

        projected = np.dot(model['W'].reshape((1, model['W'].shape[0])), Matkernel2)

        WX = np.tile(projected, [nbr_class -1, 1])
        # print "projected ok", projected.shape #, projected

        WX -= np.dot(model['bk'], np.ones((1, WX.shape[1])))

        # print "res after projection ok", WX.shape #, WX

        WX[np.where( WX[:, :] > 0 )] = np.NaN
        rslt = np.array(WX)
        # print rslt
        # print "WX after", rslt

        valmax = np.nanmax(np.array(rslt)[:, :], axis =0)
        # print "val max size : ", valmax.shape[0]
        predicted = np.zeros((1, valmax.shape[0]))
        # print "predicted size:", predicted.shape

        for i, j in enumerate(np.where(rslt == valmax)[0]):
            #print i, j
            predicted[0, i] = int(j + 1)

        NaN_index = np.where(np.isnan(valmax))
        predicted[0, NaN_index] = nbr_class
        zero_index = np.where(predicted[0, :] == 0)
        print zero_index
        predicted[0, zero_index] = nbr_class
        # print valmax, predicted
        return valmax, predicted
