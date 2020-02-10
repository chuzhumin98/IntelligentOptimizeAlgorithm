from libsvm.svmutil import *
from libsvm.commonutil import *
import numpy as np

from scipy.sparse import vstack
from scipy import hstack as hstack_array

from PSO import PSO
import os
import time
import matplotlib.pyplot as plt

y_0, x_0 = svm_read_problem('data/australian_onehot_normal_libsvm.dat', return_scipy=True)

def shufffle_index(size):
    index_array = np.array(range(size))
    np.random.shuffle(index_array)
    return index_array

def fit_function(x_t):
    # x_t = [C, gamma]
    size = len(y_0)

    cut_piece = 7
    piece_size = size // cut_piece

    accuracys = np.zeros(cut_piece)
    MSEs = np.zeros(cut_piece)

    index_array = shufffle_index(size)
    y, x = y_0[index_array[:]], x_0[index_array[:]]

    for k in range(cut_piece):
        piece_low = piece_size * k
        if k == cut_piece - 1:
            piece_high = size
        else:
            piece_high = piece_size * (k + 1)
        x_train = vstack([x[:piece_low], x[piece_high:]])
        y_train = hstack_array((y[:piece_low], y[piece_high:]))
        x_val = x[piece_low:piece_high]
        y_val = y[piece_low:piece_high]

        model = svm_train(y_train, x_train, '-t 2 -c {} -g {} -h 0'.format(x_t[0], x_t[1]))
        p_label, p_acc, p_val = svm_predict(y_val, x_val, model, log=False)
        # print('accuracy: {}, MSE: {}'.format(p_acc[0] / 100., p_acc[1]))
        accuracys[k] = p_acc[0] / 100.
        MSEs[k] = p_acc[1]

    accuracy_avg = np.mean(accuracys)
    MSE_avg = np.mean(MSEs)
    # print('average accuracy = {}, average MSE = {}'.format(accuracy_avg, MSE_avg))

    return MSE_avg

def generate_initial_particles(m):
    return np.exp((np.random.rand(m, 2) - 0.5) * 10)

def generate_initial_velocities(m):
    return (np.random.rand(m, 2) - 0.5) * 100

def support_set(x_t):
    return x_t[0] > 0 and x_t[1] > 0

def generate_initial_particle():
    return np.exp((np.random.rand(2) - 0.5) * 10)

def plot_hist_svm(size, path=None):
    if path is None:
        values = np.zeros([size])
        for i in range(size):
            values[i] = 1 - fit_function(generate_initial_particle())

        if not os.path.exists('./result'):
            os.makedirs('./result')

        if not os.path.exists('./image'):
            os.makedirs('./image')

        np.save('./result/hist_svm_10k.npy', values)
    else:
        values = np.load('./result/hist_svm_10k.npy')

    plt.switch_backend('agg')
    plt.figure(1)
    plt.hist(values)
    plt.savefig('./image/SVM_hist.png', dpi=150)

def calculate_position_svms(path, value):
    values = np.load(path)
    print('the maximum of the {} result is {}'.format(len(values), values.max()))
    print('the rank of the {} in this array is {}'.format(value, np.sum(values > value) + 1))




if __name__ == '__main__':
    print('well')

    # plot_hist_svm(10000)
    calculate_position_svms('./result/hist_svm_10k.npy', 0.657983193)

    '''

    m = 15

    K = 10
    values = np.zeros([4, K])
    for k in range(K):
        time_start = time.time()
        optimizer = PSO(omega=0.9, C_1=1.48, C_2=1.48, T_max=10, V_max=100)
        values[0, k], G_best_position, _ = optimizer.optimize(fit_function, support_set, generate_initial_particles(m), generate_initial_velocities(m))
        time_end = time.time()
        values[3, k] = time_end - time_start
        values[1, k], values[2, k] = G_best_position[0], G_best_position[1]

    if not os.path.exists('./result'):
        os.makedirs('./result')


    file = open('./result/SVM_pso_1.csv', 'w')
    file.write('No.,Accuracy,MSE,C,gamma,time\n')
    for k in range(K):
        file.write('{},{},{},{},{},{}\n'.format(k+1, 1-values[0,k], values[0,k], values[1,k], values[2,k], values[3,k]))

    file.write('Avg,{},{},{},{},{}\n'.format(1 - np.mean(values[0, :]), np.mean(values[0, :]), np.mean(values[1, :]), np.mean(values[2, :]), np.mean(values[3, :])))
    file.close()


    file = open('./result/SVM_pso_onehot_normal.csv', 'w')
    file.write('K,Avg_Accuracy,Max_Accuracy,Min_Accuracy,Std_Accuracy,Time_Avg\n')

    file.write('{},{},{},{},{},{}'.format(K, 1-np.mean(values[0, :]), 1-np.min(values[0,:]), 1-np.max(values[0,:]),
                                          np.std(values[0,:]), np.mean(values[3,:])))
    file.close()
    '''








