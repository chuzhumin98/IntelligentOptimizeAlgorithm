import numpy as np
from math import sqrt

import matplotlib.pyplot as plt

import os


def get_points_position(path):
    file = open(path, 'r')
    points = []
    while True:
        line = file.readline()
        if line:
            x, y = line.split(' ', 1)
            x = float(x.strip())
            y = float(y.strip())
            points.append([x, y])
        else:
            break
    file.close()
    return np.array(points)


class TSP:
    def __init__(self, points):
        self.points = points
        self.n = len(self.points)

    def l_2_distance(self, x, y):
        return sqrt(np.sum(np.square(x - y)))

    def function_1(self, x):
        # x is an order of the position, i.e. permutation of range(n)
        n = len(x)
        distances = np.array([self.l_2_distance(self.points[x[i], :], self.points[x[i-1], :]) for i in range(n)])
        return np.sum(distances)

    def generate_initial_point(self):
        state = np.array(range(self.n))
        np.random.shuffle(state)
        return state

    def generate_initial_points(self, m):
        samples = np.zeros([m, self.n], dtype=np.int)
        for i in range(m):
            state = np.array(range(self.n))
            np.random.shuffle(state)
            samples[i, :] = state
        return samples

    def plot_surface(self):
        plt.figure()
        plt.scatter(self.points[:, 0], self.points[:, 1], s=5, c='b')
        for i in range(self.n):
            plt.text(self.points[i, 0]+10, self.points[i, 1]-10, str(i))
        plt.title('TSP points distribution')
        plt.savefig('../../image/TSP_distribution.png', dpi=150)

    def hist_plot(self, size, path=None):
        if path is None:
            values = np.zeros([size])
            for i in range(size):
                values[i] = self.function_1(self.generate_initial_point())

            if not os.path.exists('image'):
                os.makedirs('image')


            if not os.path.exists('result'):
                os.makedirs('result')

            np.save('result/TSP_hist_result.npy', values)
        else:
            values = np.load(path)

        plt.switch_backend('agg')
        plt.figure(1)
        plt.hist(values)
        plt.savefig('image/TSP_hist.png', dpi=150)

    def hist_position(self, path, value):
        values = np.load(path)
        print('the minimum of the {} result is {}'.format(len(values), values.min()))
        print('the rank of the {} in this array is {}'.format(value, np.sum(values < value)+1))



if __name__ == '__main__':
    tsp = TSP(get_points_position('./point_position_sample1.txt'))
    # tsp.plot_surface()
    # tsp.hist_plot(10000000, path='result/TSP_hist_result_10M.npy')
    tsp.hist_position('result/TSP_hist_result_10M.npy', 29314.083)
    tsp.hist_position('result/TSP_hist_result_10M.npy', 30536.371)




