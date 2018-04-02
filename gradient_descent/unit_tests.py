# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:46:20 2018

@author: Erik
"""
import get_data as gd
import numpy as np
import gradient_calculation as gc
import full_gradient as fg
from scipy.optimize import check_grad
import time

#should just print a bunch of trues

X, y = gd.read_data("train_sgd.txt")
file = open('../data/train_sgd.txt' , 'r') 

print("Testing get_data:")
print(np.sum(y[0] == np.array([0, 10, 4])) == 3)
print(len(X) == 3438)
print(len(X[0]) == 3)
print(len(X[0][0]) == 129)
print(type(X[0][0]) == np.ndarray)
print(type(X[0][0][0]) == np.int32)
line = file.readline()
temp = line.split()
temp = np.array(temp[5:]).astype(np.int)
print(np.sum(X[0][0][1:] == temp) == 128)

print(np.sum(y[1] == np.array([14, 12, 12, 0, 13, 3, 8, 13, 6])) == 9)
print(len(X[1]) == 9)
print()


print("Testing gradient_calculation:")
params = np.zeros(129 * 26 + 26 **2)

X_train = X
y_train = y
X_test, y_test = gd.read_data("test_sgd.txt")

start = time.time()
print("Gradient calculation error on first hundred examples:")
#print(check_grad(fg.func_to_minimize, fg.grad_func, params, X_train[:100], y_train[:100], X_test, y_test, 0.001, False))
#print("Finished in " + str(time.time() - start) + " seconds")

print()
print("Running optimization")
print("train_word_accuracy, train_letter_accuracy, test_word_accuracy, test_letter_accuracy")

fg.optimize(params, X_train, y_train, 0.001)


