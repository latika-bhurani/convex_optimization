import numpy as np
import full_gradient as fg
import random as rd
import math
import get_data as gd
import callback_function as cf

def adam_crf(call_func, lmda, n_epoch, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    epoch = 0
    iteration = 0
    m_t = 0
    v_t = 0
    theta_0 = np.zeros(129 * 26 + 26 ** 2)
    grad = np.zeros_like(theta_0)

    while True:
        epoch += 1
        theta_0_prev = theta_0
        grad = fg.grad_func_word(theta_0, call_func.X_train, call_func.y_train, rd.randint(0, len(X_train) - 1), lmda)
        
        m_t = beta_1 * m_t + (1 - beta_1) * grad
        v_t = beta_2 * v_t + (1-beta_2) * (grad ** 2)
        m_t_hat = m_t / (1-beta_1**epoch)
        v_t_hat = v_t / (1-beta_2**epoch)
        
        sub_vec_num = (learning_rate * m_t_hat) 
        sub_vec_den = (np.sqrt(v_t_hat) + epsilon)
        sub_vec = sub_vec_num/sub_vec_den
        
        theta_0 = theta_0 - sub_vec
        if epoch % len(X_train) == 0 and epoch > 0:
            '''
            print("m_t: %3f" %np.sum(m_t ** 2))    
            print("v_t: %3f" %np.sum(v_t ** 2))
            print("m_t_hat: %3f" %np.sum(m_t_hat **2))
            print("v_t_hat: %3f" %np.sum(v_t_hat ** 2))
            print("sub_vec_num: %3f" %np.sum(sub_vec_num **2))
            print("sub_vec_den: %3f" %np.sum(sub_vec_den ** 2))
            print("min of den: %3f" %np.min(sub_vec_den))
            print("sub_vec: %3f" %np.sum(sub_vec**2))
            '''
            
            iteration += 1
            call_func.call_back(theta_0)
            if iteration >= n_epoch:
                print("Iteration limit")
                break
        
        if np.sum(abs(theta_0 - theta_0_prev)) <= epsilon:
            print("Small change in theta")
            break
        
    return theta_0


rd.seed(10)
X_train, y_train = gd.read_data("train_sgd.txt")
X_test, y_test = gd.read_data("test_sgd.txt")

cf = cf.callback_function(X_train, y_train, X_test, y_test, "adam_1e-2.txt")
cf.delete_file()
print("computing optimal params")
opt_params = adam_crf(cf, 0.01, 50, learning_rate = 0.0005)
