import numpy as np
import full_gradient as fg
import random as rd

def adam_crf(X_train, y_train, n_epoch, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):

    epoch = 0
    m_0 = 0
    v_0 = 0
    theta_0 = np.ones(129 * 26 + 26 ** 2)
    grad = np.ones_like(theta_0)

    while True:
        epoch += 1
        theta_0_prev = theta_0
        grad = fg.grad_func_word(theta_0, X_train, y_train, rd.randint(0, len(X_train) - 1), learning_rate)
        m_t = beta_1 * m_0 + (1 - beta_1) * grad
        v_t = beta_2 * v_0 + (1-beta_2) * grad.transpose() * grad
        m_t_hat = m_t / (1-beta_1**epoch)
        v_t_hat = v_t / (1-beta_2**epoch)

        theta_0 = theta_0 - (learning_rate * m_t_hat) / (math.sqrt(v_t_hat) + epsilon)

        if abs(theta_0 - theta_0_prev) <= epsilon:
            break

    return theta_0

def print_params(params):
    global iteration
    with open("adam.txt", "a") as text_file:
        text_file.write("%d " % iteration)
        for p in params:
            text_file.write("%f " % p)
        text_file.write("\n")
        text_file.close()
    iteration += 1

def calculate_adam_accuracy(X_train, y_train, X_test, y_test):
    file = open("adam.txt", 'r')
    print("iter: test_word, test_letter, train_word, train_letter, train_word")
    for line in file:
        split = line.split()
        print(split[0] + ": ")
        params = np.array(split[1:]).astype(np.float)
        fg.print_accuracies(params, X_train, y_train, X_test, y_test)
    file.close()

