import numpy as np
import random
import callback_function as cf
import get_data as gd
import full_gradient as fg
import random as rd
import gradient_calculation as gc
from numpy import matlib

letter_dict = {1: 'a', 2:'b', 3:'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i',
               10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15:'o',
               16:'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v',
               23: 'w', 24: 'x', 25: 'y', 26: 'z'}

reverse_lookup = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8,
                  'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16,
                  'r':17, 's': 18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23,
                  'y':24, 'z':25}

class GibbsSample:

    def __init__(self):
        self.theta = np.ones(129 * 26 + 26 ** 2)
        self.learning_rate = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.iterations = 3700000
        self.lambda_param = 0.000001

    def generate_samples(self, X_train, w, t, samples_per_word):

        # m - length of the word to be sampled
        index = random.randint(0, len(X_train)-1)
        m = len(X_train[index])

        init_y_sample = random.sample(range(26), m)
        samples = []
        numerator = np.zeros(26)
        sample = np.copy(init_y_sample)
        for k in range(samples_per_word):
            rand_char_index = random.randint(0, m-1)

            # for i in range(0, m):
            #     if i != rand_char_index and (i != rand_char_index - 1):
            #         sum = sum + np.dot(w[sample[i]], self.X_train[index][i]) + t[sample[i]][sample[i+1]]

            #initialize a sum array with this sum value


            for b in range(100):
                for j in range(26):
    
    
                    if rand_char_index == 0:
                        numerator[j] = np.inner(X_train[index][rand_char_index], w[j]) \
                                         + t[j][sample[rand_char_index+1]]
    
                    elif rand_char_index == m - 1:
                        numerator[j] = t[sample[rand_char_index-1]][j]\
                                         + np.inner(X_train[index][rand_char_index], w[j])
    
                    else:
    
                        numerator[j] = t[sample[rand_char_index-1]][j]\
                                         + np.inner(X_train[index][rand_char_index], w[j]) \
                                         + t[j][sample[rand_char_index+1]]
                #
                # numerator[j - 1] = np.dot(w[init_y_sample[rand_char_index - 1]], X_train[index][rand_char_index - 1]) \
                #                    + t[129 * 26 + 26 * (init_y_sample[rand_char_index - 1] - 1) + sample[j]] \
                #                    + np.dot(w[j], X_train[index][rand_char_index]) \
                #                    + t[129 * 26 + 26 * (sample[j] - 1) + sample[rand_char_index]]
            best_index = np.argmax(numerator)
            sample[rand_char_index] = best_index

            samples.append(np.copy(sample))

        return (samples, index)


    def train_with_samples(self, call_func):
        # parameter tuning using ADAM

        epoch = 0
        m_t = 0  # first order moment
        v_t = 0  # second order moment

        # init params
        theta = np.zeros(129 * 26 + 26 ** 2)

        # init gradient
        grad = np.ones_like(theta)
        while True:
            epoch += 1

            theta_prev = theta
            #print("Iterations : " + str(epoch))

            w = gc.w_matrix(theta)
            t = gc.t_matrix(theta)
            samples, sampled_word_index = self.generate_samples(call_func.X_train, w, t, 100)#theta[:3354], theta[3354:], 5)
            # calculate gradient
            x_train_array = []
            x_train_array.append(call_func.X_train[sampled_word_index])

            y_train_array = []
            y_train_array.append(samples[-1])

            grad = fg.grad_func_word(theta, x_train_array, y_train_array, 0, self.lambda_param)
        #   grad = fg.grad_func_word(theta, x_train_array, samples, 0, 0.01)
        #   grad = gc.gradient_avg(theta, (X_train[sampled_word_index], len(samples), 1), samples, len(samples) - 1)

            # biased moments
            m_t = self.beta_1 * m_t + (1 - self.beta_1) * grad
            v_t = self.beta_2 * v_t + (1 - self.beta_2) * np.square(grad)

            # bias corrected first and second order moments
            m_t_hat = m_t / (1 - self.beta_1 ** epoch)
            v_t_hat = v_t / (1 - self.beta_2 ** epoch)

            # update params
            theta = theta_prev - (self.learning_rate * m_t_hat) / (np.sqrt(v_t_hat) + self.epsilon)

            #print(theta)
            # termination condition
            if sum(abs(theta - theta_prev)) <= self.epsilon or epoch == self.iterations:
                break
            
            if(epoch % 100 == 0 and epoch > 0):
                call_func.call_back(theta)

        return theta

gibbs  = GibbsSample()
X_train, y_train = gd.read_data("train_sgd.txt")
X_test, y_test = gd.read_data("test_sgd.txt")
call_func = cf.callback_function(X_train, y_train, X_test, y_test, "eoslan2-test.txt", 0.01)
opt_params = gibbs.train_with_samples(call_func)


print("Final accuracy:")
fg.print_accuracies(opt_params, X_train, y_train, X_test, y_test)
