#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from Ordinalkdl import Ordinalkdl

import pickle
import scipy.io

import time
 










def main():

    # load dataset 
    # BE SURE THAT DATA FIT INTO THE MEMORY OF YOUR MACHINE
    x = np.loadtxt('test/train_toy.0')
    y = np.loadtxt('test/test_toy.0')
    train_feat = x[:, 0:2]
    train_labels = x[:,2]
    test_feat = y[:, 0:2]
    test_labels = y[:,2]

    param =np.zeros((2, 1))
    kernel = "gaussian"
    optmethod = "cvx"
    gamma = 0.1
    C = 10.
    u = 0.001
    param[0] = gamma
    param[1] = u
    #
    kdlor = Ordinalkdl( kernel, param, optmethod, C )
    # print x.shape, y.shape
    
    
    ## For training ##
    # Training the model
    tic = time.time()
    kdlormodel = kdlor.Learn( train_feat, train_labels )
    toc =time.time()
    t_train = (toc - tic)/60.
    print 'trainning time', t_train

    # save the kdlor model
    #modelfile = open( 'data/kdlor_aging_model.pkl', 'wb' )
    #pickle.dump( kdlormodel, modelfile )
    #modelfile.close()
    #print "training time (mn) = ", t_train


    tic = time.time()
    _, predicted = kdlor.Test( train_feat, test_feat, kdlormodel )
    toc =time.time()
    t_test = (toc - tic)/60.
    print 'testing time', t_test
    








if __name__ == "__main__":
    main()
