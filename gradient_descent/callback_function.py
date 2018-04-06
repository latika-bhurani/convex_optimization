# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:11:02 2018

@author: Erik
"""
import numpy as np
import full_gradient as fg
import gradient_calculation as gc
import os

class callback_function:
    def __init__(self, iX_train, iy_train, iX_test, iy_test, ifile_name):
        self.X_train = iX_train
        self.y_train = iy_train
        self.X_test = iX_test
        self.y_test = iy_test
        self.file_name = ifile_name
        self.iteration = 0
        self.avg_grad = 0

    def delete_file(self):
        if(os.path.isfile(self.file_name)):
            os.remove(self.file_name)
    
    def zero_iteration(self):
        self.iteration = 0
        
    
    def call_back(self, params, l):
        self.iteration += 1
        self.print_params(params)
        self.print_gradient_average(params, l)
    
    def funct_value(self, params):
        
        print(fg.func_to_minimize(params, self.X_train, self.y_train))
    
    def print_gradient_average(self, params, l):
        print("Average gradient %d:" %self.iteration)
        self.avg_grad = fg.grad_func(params, self.X_train, self.y_train, l)
        print(np.sum(self.avg_grad ** 2))
    
    def print_params(self, params):
        with open(self.file_name, "a") as text_file:
            text_file.write("%d " %self.iteration)
            for p in params:
                text_file.write("%f " %p)
            text_file.write("\n")
            text_file.close()
