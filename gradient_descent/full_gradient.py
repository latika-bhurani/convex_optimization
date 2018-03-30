
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:02:04 2018

@author: Erik
"""

import gradient_calculation as gc
from scipy.optimize import fmin_bfgs
from time import time
import numpy as np
import decode as dc

iteration = 0



def func_to_minimize(params, X_train, y_train, X_test, y_test, l, do_printing):
    num_examples = len(X_train)
    reg = 1/2 * np.sum(params ** 2)
    avg_prob = gc.log_p_y_given_x_avg(params, X_train, y_train, num_examples)

    
    if(do_printing):
        global iteration
        #this function gets called by bfgs, so we need to track accuracy at every  iteration
        w = gc.w_matrix(params)
        t = gc.t_matrix(params)
        predictions = predict(X_train, w, t)
        acc = accuracy(y_train, predictions)
        print(str(iteration) + ": ", end = '')
        print("%.3f, " %acc[0], end = '')
        print("%.3f, " %acc[1], end = '')
        
        predictions = predict(X_test, w, t)
        acc = accuracy(y_test, predictions)
        print("%.3f, " %acc[0], end = '')
        print("%.3f " %acc[1])
        iteration += 1
        
    
    return -avg_prob + l * reg

def grad_func(params, X_train, y_train, X_test, y_test, l, do_printing):
    num_examples = len(X_train)
    grad_avg =  gc.gradient_avg(params, X_train, y_train, num_examples)
    grad_reg = params
    return - grad_avg + l * grad_reg
    

def optimize(params, X_train, y_train, X_test, y_test, l):

    start = time()
    fmin_bfgs(func_to_minimize, params, grad_func, (X_train, y_train, X_test, y_test, l, True))
    print("Total time: ", end = '')
    print(time() - start)
    
def accuracy(y_pred, y_act):
    word_count = 0
    correct_word_count = 0
    letter_count = 0
    correct_letter_count = 0
    for i in range(len(y_pred)):
        word_count += 1
        correct_word_count += np.sum(y_pred[i] == y_act[i]) == len(y_pred[i])
        letter_count += len(y_pred[i])
        correct_letter_count += np.sum(y_pred[i] == y_act[i])
    return correct_word_count/word_count, correct_letter_count/letter_count

def get_optimal_params(name):
    
    file = open('../result/' + name + '.txt', 'r') 
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)

def predict(X, w, t):
    y_pred = []
    for i, x in enumerate(X):
        M = dc.decode(x, w, t)
        y_pred.append(dc.get_solution_from_M(M, x, w, t))
    return y_pred