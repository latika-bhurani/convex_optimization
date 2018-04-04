# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:59:43 2018

@author: Erik
"""
import numpy as np
import full_gradient as fg
import get_data as gd

def print_function_values(infile, outfile, X_train, y_train, l):
    lines = []
    f_vals = []
    #just to keep the file open for as short a time as possible
    file = open(infile, 'r')
    for line in file:
        lines.append(line)
    file.close()
    for line in lines:
        split = line.split()
        print(split[0] + ": ", end = '')
        params = np.array(split[1:]).astype(np.float)
        f_vals.append(fg.func_to_minimize(params, X_train, y_train))
        
    file = open(outfile, 'w')
    file.writelines(f_vals)
    file.close()
    
X_train, y_train = gd.read_data("train_sgd.txt")
print_function_values("sgd_1e-2.txt", "sgd_1e-2_f_vals.txt", X_train, y_train, 0.01)

X_train, y_train = gd.read_data("train_sgd.txt")
print_function_values("sgd_1e-4.txt", "sgd_1e-4_f_vals.txt", X_train, y_train, 0.0001)

X_train, y_train = gd.read_data("train_sgd.txt")
print_function_values("sgd_1e-6.txt", "sgd_1e-6_f_vals.txt", X_train, y_train, 0.000001)

X_train, y_train = gd.read_data("train_sgd.txt")
print_function_values("sgd_1e-2.txt", "sgd_1e-2_f_vals.txt", X_train, y_train, 0.01)

X_train, y_train = gd.read_data("train_sgd.txt")
print_function_values("sgd_1e-4.txt", "sgd_1e-4_f_vals.txt", X_train, y_train, 0.0001)

X_train, y_train = gd.read_data("train_sgd.txt")
print_function_values("sgd_1e-6.txt", "sgd_1e-6_f_vals.txt", X_train, y_train, 0.000001)