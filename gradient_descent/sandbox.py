# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 20:39:45 2018

@author: Erik
"""

import decode as dc
import get_data as gd
import numpy as np
import full_gradient as fg
import gradient_calculation as gc

def decode(X, w, t):
    #for 100 letters I need an M matrix of 100  X 26
    M = np.zeros((len(X), 26))
    
    #populates first row
    for j in range(26):
        M[0][j] = np.inner(X[0], w[j])
    
    #go row wise through M matrix, starting at line 2 since first line is populated
    for row in range(1, len(X)):
        #Need to set this letter equal to it's own inner product + max previous + transition
        
        
        for cur_letter in range(26):
            M[row][cur_letter] = np.max(M[row-1] + t.transpose()[cur_letter]) + np.inner(X[row], w[cur_letter])
            
            
            
    return M

params = np.zeros(129*26 + 26 ** 2)
w = gc.w_matrix(params)
t = gc.t_matrix(params)

X_test, y_test = gd.read_data("test_sgd.txt")

for i, X in enumerate(X_test):
    print("Iteration: %d" %i)
    true_decode = dc.decode(X, w, t)
    test_decode = decode(X, w, t)
    print("Same: ", end = '')
    print(np.sum(true_decode == test_decode) == true_decode.size)
    
