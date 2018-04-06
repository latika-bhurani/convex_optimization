import numpy as np
import full_gradient as fg
import random as rd
import get_data as gd
import callback_function as cf

def adam_crf(call_func, lmda, n_epoch, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    epoch = 0
    iteration = 0
    
    #this was m_0, so the update was always referencing the 0th m_t and v_t rather than t-1
    m_t = 0
    v_t = 0
    theta_0 = np.zeros(129 * 26 + 26 ** 2)
    grad = np.zeros_like(theta_0)

    while True:
        epoch += 1
        theta_0_prev = theta_0
        grad = fg.grad_func_word(theta_0, call_func.X_train, call_func.y_train, rd.randint(0, len(X_train) - 1), lmda)
        
        m_t = beta_1 * m_t + (1 - beta_1) * grad
        
        #this was grad.transpose()*grad.  Can't gaurentee that this is completely right either but adam said
        #"element wise square" and v_t wouldn't be a vector without somre reference to the gradient since beta_2 is a
        #scalar.  So I'm 80% convinced this is right.
        v_t = beta_2 * v_t + (1-beta_2) * (grad ** 2)
        
        
        m_t_hat = m_t / (1-beta_1**epoch)
        v_t_hat = v_t / (1-beta_2**epoch)
        
        #this was math.sqrt rather than np.sqrt.  math only works on scalars, np works on vectors
        theta_0 = theta_0 - (learning_rate * m_t_hat) / (np.sqrt(v_t_hat) + epsilon)
        if epoch % len(X_train) == 0 and epoch > 0:            
            iteration += 1
            call_func.call_back(theta_0, lmda)
            if iteration >= n_epoch:
                print("Iteration limit")
                break
        
        if np.sum(abs(theta_0 - theta_0_prev)) <= epsilon:
            print("Small change in theta")
            break
        
    return theta_0


X_train, y_train = gd.read_data("train_sgd.txt")
X_test, y_test = gd.read_data("test_sgd.txt")

cf = cf.callback_function(X_train, y_train, X_test, y_test, "adam_1e-2.txt")
cf.delete_file()
print("computing optimal params")
opt_params = adam_crf(cf, 0.01, 50, learning_rate = 0.00005)
fg.print_accuracies(opt_params, X_train, y_train, X_test, y_test)