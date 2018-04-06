
import get_data as gd
import gradient_calculation as gc
import numpy as np
import full_gradient as fg
import random as rd
import callback_function as cf

    
def sgd_crf(call_func, params, l_rate, n_epoch = 100):
    epoch = 0
    iteration = 0
    while(True):
        #calculate gradient with respect to a randomly selected word
        gradient = fg.grad_func_word(params, call_func.X_train, call_func.y_train, 
                                     rd.randint(0, len(X_train) - 1), 
                                     call_func.lmda)
     
        params = params - l_rate * gradient
        
        #stopping criteria.  Here, if the epoch limit is reached or the gradient
        # is less than the gradient tolerance then stop
        iteration += 1
        
        if(iteration % len(X_train) == 0):
            epoch += 1
            
            call_func.call_back(params)
            if(epoch >= n_epoch):
                print("Epoch limit")
                break
                
    return params
 

X_train, y_train = gd.read_data("train_sgd.txt")
X_test, y_test = gd.read_data("test_sgd.txt")
params = np.zeros(129*26 + 26 **2)


cf = cf.callback_function(X_train, y_train,  X_test, y_test, "sgd_1e-2.txt", 0.01)
cf.delete_file()
print("computing optimal params")
#args are callbackfunction,      learning rate
opt_params = sgd_crf(cf, params, 0.0001)
print("Final accuracy:")
fg.print_accuracies(opt_params, X_train, y_train, X_test, y_test)



cf = cf.callback_function(X_train, y_train,  X_test, y_test, "sgd_1e-4.txt", 0.0001)
cf.delete_file()
print("computing optimal params")
#args are callbackfunction,      learning rate
opt_params = sgd_crf(cf, params, 0.001)
print("Final accuracy:")
fg.print_accuracies(opt_params, X_train, y_train, X_test, y_test)



cf = cf.callback_function(X_train, y_train,  X_test, y_test, "sgd_1e-6.txt", 0.000001)
cf.delete_file()
print("computing optimal params")
#args are callbackfunction,      learning rate
opt_params = sgd_crf(cf, params, 0.001)
print("Final accuracy:")
fg.print_accuracies(opt_params, X_train, y_train, X_test, y_test)


