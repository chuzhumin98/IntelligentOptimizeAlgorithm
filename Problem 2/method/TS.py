import numpy as np
from random import randint

import matplotlib.pyplot as plt

# note that this is for combination optimize problem
class TS:
    def __init__(self, tabu_size=20, candidate_number=100, iter_max=1000):
        self.tabu_size = tabu_size
        self.candidate_number = candidate_number
        self.iter_max = iter_max

    def generate_neighbor(self, x, m):
        # generate x's neighbors with size = m
        n = len(x)
        neighbors_list = np.zeros([m, n], dtype=np.int)
        changes_list = np.zeros([m, 2], dtype=np.int)
        for i in range(m):
            while True:
                r_1, r_2 = randint(0, n - 1), randint(0, n - 1)
                if (r_1 != r_2):
                    break
            changes_list[i, 0] = min(r_1, r_2)
            changes_list[i, 1] = max(r_1, r_2)
            neighbors_list[i, :] = np.copy(x)
            neighbors_list[i, r_1] = x[r_2]
            neighbors_list[i, r_2] = x[r_1]

        return neighbors_list, changes_list


    def optimize(self, function, x_0, plot=False):
        n = len(x_0)
        tabu_list = np.zeros([n, n], dtype=np.int) # 0-not in tabu list, 1-in
        history_tabu = []
        tabu_len = 0

        x_t = np.copy(x_0)
        best_x = np.copy(x_0)
        best_value = function(x_0)
        best_values = [best_value]
        current_values = [best_value]

        for iter in range(self.iter_max):
            candidates_set, changes_set = self.generate_neighbor(x_t, self.candidate_number)
            candidates_value = np.array([function(candidates_set[i, :]) for i in range(self.candidate_number)])
            sort_index = np.argsort(candidates_value)

            used_index = 0 # the index used for the next iter
            for k in range(self.candidate_number):
                if (candidates_value[sort_index[k]] < best_value):
                    used_index = sort_index[k]
                    break
                else:
                    if tabu_list[changes_set[sort_index[k], 0], changes_set[sort_index[k], 1]] == 0:
                        used_index = sort_index[k]
                        break

            if tabu_list[changes_set[used_index, 0], changes_set[used_index, 1]] == 0:
                if (tabu_len == self.tabu_size): # if tabu is full, pop the first one
                    pop_1, pop_2 = history_tabu[len(history_tabu) - self.tabu_size]
                    tabu_list[pop_1, pop_2] = 0

                # add this change to tabu
                tabu_list[changes_set[used_index, 0], changes_set[used_index, 1]] == 1
                history_tabu.append([changes_set[used_index, 0], changes_set[used_index, 1]])

            x_t = candidates_set[used_index, :]
            current_values.append(candidates_value[used_index])

            # update the best if necessary
            if candidates_value[used_index] < best_value:
                best_x = np.copy(candidates_set[used_index, :])
                best_value = candidates_value[used_index]

            best_values.append(best_value)

        if plot:
            self.plot_graph(current_values, 'TSP cost vs iter in TS method')

        print('TS search result, the minimum = {} when the combination order = {}'.format(best_value, best_x))
        print('best_values = {}'.format(best_values))
        print('current_values = {}'.format(current_values))
        return best_value, best_x, best_values, current_values

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

if __name__ == '__main__':
    print('TS algorithm')



