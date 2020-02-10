import numpy as np
from itertools import accumulate
from random import random
from random import randint

import matplotlib.pyplot as plt

class GA:
    def __init__(self, rate_crossover=0.7, rate_mutation=0.02, cycle_max=1000):
        self.rate_crossover = rate_crossover
        self.rate_mutation = rate_mutation
        self.cycle_max = cycle_max

    def crossover(self, x, y):
        n = len(x)
        while True:
            left, right = randint(0, n-1), randint(0, n-1)
            if left != right and abs(right - left) != n-1:
                break
        if (left > right):
            left, right = right, left
        slices = x[left:right+1]
        son = np.zeros([n], dtype=np.int)
        son_cutoff = np.zeros([n-len(slices)], dtype=np.int)
        cnt_cutoff = 0
        for i in range(n):
            if y[i] not in slices:
                son_cutoff[cnt_cutoff] = y[i]
                cnt_cutoff += 1
        son[0:left] = son_cutoff[0:left]
        son[left:right+1] = slices
        son[right+1:] = son_cutoff[left:]
        return son


    def mutation(self, x):
        n = len(x)
        while True:
            one, another = randint(0, n-1), randint(0, n-1)
            if (one != another):
                break
        son = np.copy(x)
        son[one], son[another] = son[another], son[one]
        return son

    def optimize(self, function, xs_0, plot=False):
        m, n = xs_0.shape
        self.life_num = m

        results = np.array([function(xs_0[i, :]) for i in range(m)])
        best_x = np.copy(xs_0[np.argmin(results), :])
        best_value = np.min(results)

        best_values = [best_value]
        current_best_values = [best_value]

        xs_t = np.copy(xs_0)

        cycle = 0

        while True:
            adapt_rate = 1 / results  # each sample's adapt rate
            adapt_rate_accumulate = np.array(list(accumulate(list(adapt_rate))))
            adapt_rate_accumulate /= adapt_rate_accumulate[-1]

            parents = np.zeros([2*m, n], dtype=np.int)
            sons = np.zeros([m, n], dtype=np.int)
            # generate the initial parent with given prob
            for i in range(2*m):
                random_value = random()
                random_index = np.sum(adapt_rate_accumulate < random_value)
                parents[i, :] = np.copy(xs_t[random_index, :])

            # crossover process
            for i in range(m):
                if random() <= self.rate_crossover:
                    sons[i, :] = self.crossover(parents[i*2, :], parents[i*2+1, :])
                else:
                    sons[i, :] = parents[i*2+1]

            # mutation process
            for i in range(m):
                if random() <= self.rate_mutation:
                    sons[i, :] = self.mutation(sons[i, :])

            xs_t = sons
            results = np.array([function(xs_t[i, :]) for i in range(m)])

            current_best_x = np.copy(xs_t[np.argmin(results), :])
            current_best_value = np.min(results)

            if current_best_value < best_value:
                best_value = current_best_value
                best_x = np.copy(current_best_x)

            best_values.append(best_value)
            current_best_values.append(current_best_value)

            cycle += 1
            if cycle >= self.cycle_max:
                break

        print('GA search result, the minimum = {} when the combination order = {}'.format(best_value, best_x))
        print('best_values = {}'.format(best_values))
        print('current_values = {}'.format(current_best_values))

        if plot:
            self.plot_graph(best_values, 'TSP best cost vs iter in GA method')
            self.plot_graph(current_best_values, 'TSP current best cost vs iter in GA method', figure_index=1)


        return best_value, best_x, best_values, current_best_values


    def plot_graph(self, values, name, figure_index=0):
        plt.figure(figure_index)
        plt.plot(np.array(range(len(values))), values, 'b')
        plt.title(name)
        plt.savefig('../image/{}.png'.format(name), dpi=150)

    def plot_TSP_curve(self, x, points, name, figure_index=0):
        curves = np.zeros([2, len(x)+1])
        for i in range(len(x)):
            curves[:, i] = points[x[i], :]
        curves[:, -1] = points[x[0], :]

        plt.figure(figure_index)
        plt.plot(curves[0,:], curves[1,:], 'b', markersize=5, marker='.', lw=1, markerfacecolor='r', markeredgecolor='r')
        plt.title(name)
        plt.savefig('../image/{}.png'.format(name), dpi=150)






