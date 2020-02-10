from method.SA import SA
from method.PSO import PSO
from method.DE import DE

from sample.test_sample_1 import function_1
from sample.test_sample_1 import support_set_1
from sample.test_sample_1 import generate_initial_point_1
from sample.test_sample_1 import generate_initial_points_1
from sample.test_sample_1 import generate_initial_velocity_1

from sample.test_sample_2 import function_2
from sample.test_sample_2 import support_set_2
from sample.test_sample_2 import generate_initial_point_2
from sample.test_sample_2 import generate_initial_points_2
from sample.test_sample_2 import generate_initial_velocity_2

import time

import numpy as np
import os

import matplotlib.pyplot as plt

def plot_curve(ys, name, plot_index=0):
    plt.figure(plot_index)
    plt.plot(np.array(range(len(ys))), ys, 'b')
    plt.title(name)
    plt.savefig('../image/{}.png'.format(name), dpi=150)

if __name__ == '__main__':
    print('main')

    '''

    optimizer = SA()
    best_result, best_x, xs, ys = optimizer.optimize(function_1, support_set_1, generate_initial_point_1())

    # plot_curve(ys, 'y vs iter in f_1 function with SA method')
    # plot_curve(xs, 'x vs iter in f_1 function with SA method', plot_index=1)
    
    

    

    m = 15
    optimizer = PSO()
    optimizer.optimize(function_1, support_set_1, generate_initial_points_1(m), generate_initial_velocity_1(m, optimizer.V_max), plot=True)




    m = 30
    optimizer = DE()
    optimizer.optimize(function_1, support_set_1, generate_initial_points_1(m), plot=True)
    


    

    n = 10
    optimizer = SA()
    best_result, best_x, xs, ys = optimizer.optimize(function_2, support_set_2, generate_initial_point_2(n))

    plot_curve(ys, 'y vs iter in f_2 function with SA method')





    n = 10
    m = 15
    optimizer = PSO()
    optimizer.optimize(function_2, support_set_2, generate_initial_points_2(m, n),
                       generate_initial_velocity_2(m, n, optimizer.V_max), plot=True)


    '''

    n = 10
    m = 30
    optimizer = DE()
    optimizer.optimize(function_2, support_set_2, generate_initial_points_2(m, n), plot=True)

    '''
    
    
    n = 10

    # optimizer = SA()
    #m = 15
    #optimizer = PSO()

    m = 30
    optimizer = DE()
    optimizer.optimize(function_1, support_set_1, generate_initial_points_1(m))

    K = 20
    results = np.zeros([K])
    times = np.zeros([K])
    for k in range(K):
        time_start = time.time()
        # results[k], _, _, _ = optimizer.optimize(function_1, support_set_1, generate_initial_point_1()) # SA 1
        # results[k], _, _, _ = optimizer.optimize(function_2, support_set_2, generate_initial_point_2(n)) # SA 2
        # results[k], _, _ = optimizer.optimize(function_1, support_set_1, generate_initial_points_1(m),
        #                   generate_initial_velocity_1(m, optimizer.V_max)) # PSO 1
        # results[k], _, _ = optimizer.optimize(function_2, support_set_2, generate_initial_points_2(m, n),
        #                   generate_initial_velocity_2(m, n, optimizer.V_max)) # PSO 2
        # results[k], _, _ = optimizer.optimize(function_1, support_set_1, generate_initial_points_1(m)) # DE 1
        results[k], _, _ = optimizer.optimize(function_2, support_set_2, generate_initial_points_2(m, n))
        time_end = time.time()
        times[k] = time_end - time_start
        print('time uses {}'.format(times[k]))
    print(results)
    print(times)

    if not os.path.exists('../result'):
        os.makedirs('../result')

    file_out = open('../result/DE_2.csv', 'w')
    file_out.write('item,average,best,worst,variance\n')
    file_out.write('minimum,{},{},{},{}\n'.format(np.mean(results), np.min(results), np.max(results), np.var(results)))
    file_out.write('time,{},{},{},{}\n'.format(np.mean(times), np.min(times), np.max(times), np.var(times)))
    
    '''