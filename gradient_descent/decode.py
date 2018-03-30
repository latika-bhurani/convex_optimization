# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:06:02 2018

@author: Erik
"""

#import common_utilities
import numpy as np
from copy import deepcopy


def decode(X, w, t):
    #for 100 letters I need an M matrix of 100  X 26
    M = np.zeros((len(X), 26))
    
    #populates first row
    for j in range(26):
        M[0][j] = np.inner(X[0], w[j])
    
    #go row wise through M matrix, starting at line 2 since first line is populated
    for row in range(1, len(X)):
        
        #go column wise, populating the best sum of the previous + T[previous letter][
        for cur_letter in range(26):
            #initialize with giant negative number
            best = -99999999999999
            
            #iterate over all values of the previous letter, fixing the current letter
            for prev_letter in range(26):
                temp_product = M[row-1][prev_letter] + np.inner(X[row], w[cur_letter]) + t[prev_letter][cur_letter]
                if(temp_product > best):
                    best = temp_product
            M[row][cur_letter] = best
    return M



def get_solution_from_M(M, X, w, t):
    solution = []
    cur_word_pos = len(M) - 1
    prev_word_pos = cur_word_pos - 1
    cur_letter = np.argmax(M[cur_word_pos])
    cur_val = M[cur_word_pos][cur_letter]
    solution.insert(0, cur_letter)
    
    while(cur_word_pos > 0):
        for prev_letter in range(26):
            if(abs(cur_val - M[prev_word_pos][prev_letter] - t[prev_letter][cur_letter] - np.inner(X[cur_word_pos], w[cur_letter]) ) < 0.00001):
                solution.insert(0, prev_letter)
                cur_letter = prev_letter
                cur_word_pos -=1
                prev_word_pos -=1
                cur_val = M[cur_word_pos][cur_letter]
                break
        
    return np.array(solution)




