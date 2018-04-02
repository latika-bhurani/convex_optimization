# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:12:49 2018

@author: Erik
"""
import get_data as gd
import gradient_calculation as gc
import numpy as np
import full_gradient as fg
import os
import random as rd

iteration = 0
def print_params(params):
    global iteration
    with open("sgd.txt", "a") as text_file:
        text_file.write("%d " %iteration)
        for p in params:
            text_file.write("%f " %p)
        text_file.write("\n")
        text_file.close()
    iteration += 1
    
def sgd_crf(X_train, y_train, params, l, l_rate, n_epoch, gtol):
    epoch = 0
    g_avg = np.ones_like(params)
    old_g_avg = np.ones_like(params)
    while(True):
        #calculate gradient with respect to a randomly selected word
        gradient = fg.grad_func_word(params, X_train, y_train, rd.randint(0, len(X_train) - 1), l)
        
        #revise average gradient
        g_avg = 0.1 * gradient + 0.9 * g_avg
        params = params - l_rate * gradient
            

        if(epoch > n_epoch):
            print("Epoch limit")
            break
        if(epoch % len(X_train) == 0):
            epoch += 1
            print_params(params)
            print("Current : ", end = '')
            print(np.sum(g_avg ** 2))
            print("Old: ", end = '')
            print(np.sum(old_g_avg ** 2))
            if(abs(np.sum(g_avg **2) -  np.sum(old_g_avg **2)) < 0.00001):
                print("small difference")
                break
            else:
                old_g_avg = g_avg
            print_params(params)

    return params
 
def get_sgd_accuracy(X_train, y_train, X_test, y_test):
    file = open("sgd.txt" , 'r')
    print("iter: test_word, test_letter, train_word, train_letter, train_word")
    for line in file:
        split = line.split()
        print(split[0] + ": ", end = '')
        params = np.array(split[1:]).astype(np.float)
        fg.print_accuracies(params, X_train, y_train, X_test, y_test)
    file.close()



if(os.path.isfile("sgd.txt")):
    os.remove("sgd.txt")
    
X_train, y_train = gd.read_data("train_sgd.txt")
X_test, y_test = gd.read_data("test_sgd.txt")
params = np.zeros(129*26 + 26 **2)

opt_params = sgd_crf(X_train, y_train, params, 0.01, 0.0007, 100000, 0.01)

get_sgd_accuracy(X_train, y_train, X_test, y_test)
