from method.TS import TS
from method.GA import GA

from sample.test_sample_1 import get_points_position
from sample.test_sample_1 import TSP

import numpy as np
import os
import time


if __name__ == '__main__':
    print('main')

    tsp = TSP(get_points_position('./sample/point_position_sample1.txt'))


    optimizer = TS()
    best_value, best_x, best_values, current_values = optimizer.optimize(tsp.function_1, tsp.generate_initial_point(), plot=True)

    optimizer.plot_TSP_curve(best_x, tsp.points, 'best TSP curve in TS method', figure_index=2)

    '''


    m = 100
    optimizer = GA()
    best_value, best_x, best_values, current_best_values = optimizer.optimize(tsp.function_1, tsp.generate_initial_points(m), plot=True)

    optimizer.plot_TSP_curve(best_x, tsp.points, 'best TSP curve in GA method', figure_index=2)
    



    m = 100
    K = 20

    # optimizer = GA()
    optimizer = TS()

    results = np.zeros([K])
    times = np.zeros([K])
    for k in range(K):
        time_start = time.time()
        # results[k], _, _, _ = optimizer.optimize(tsp.function_1, tsp.generate_initial_points(m)) # GA
        results[k], _, _, _ = optimizer.optimize(tsp.function_1, tsp.generate_initial_point())  # TS
        time_end = time.time()
        times[k] = time_end - time_start
        print('time uses {}'.format(times[k]))
    print(results)
    print(times)

    if not os.path.exists('../result'):
        os.makedirs('../result')

    file_out = open('../result/TS_2.csv', 'w')
    file_out.write('item,average,best,worst,variance\n')
    file_out.write('minimum,{},{},{},{}\n'.format(np.mean(results), np.min(results), np.max(results), np.var(results)))
    file_out.write('time,{},{},{},{}\n'.format(np.mean(times), np.min(times), np.max(times), np.var(times)))
    '''
