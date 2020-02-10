import numpy as np

from random import randint
from random import random

import matplotlib.pyplot as plt

class DE:
    def __init__(self, F=0.5, CR=0.8, cycle_max=1000, beta=0.5):
        # F - the parameter to control for mutation
        # CR - the parameter to control for crossover
        self.F = F
        self.CR = CR
        self.cycle_max = cycle_max
        self.beta = beta # search decay rate

    def optimize(self, function, support_set, x_0s, plot=False):
        m, n = x_0s.shape

        x_ts = np.copy(x_0s) # current feature
        y_ts = np.array([function(x_ts[i, :]) for i in range(m)])

        G_best_values = [y_ts.min()]  # global best values

        if plot:
            P_positions = []
            P_values = []
            for i in range(m):
                P_positions.append([x_0s[i, 0]])
                P_values.append([function(x_0s[i, :])])

        V_ts = np.zeros([m, n]) # current V
        U_ts = np.zeros([m, n]) # current U

        for cycle in range(self.cycle_max):
            # mutation process
            for i in range(m):
                r_1, r_2, r_3 = randint(0, m - 1), randint(0, m - 1), randint(0, m - 1)
                delta = self.F * (x_ts[r_2] - x_ts[r_3])
                rate = 1
                while True:
                    if support_set(x_ts[r_1, :] + rate * delta):
                        break
                    else:
                        rate *= self.beta
                V_ts[i, :] = x_ts[r_1, :] + rate * delta

            # crossover process
            random_values = np.random.rand(m, n)
            U_ts = (random_values <= self.CR) * V_ts + (random_values > self.CR) * x_ts

            # selection
            for i in range(m):
                current_value = function(U_ts[i, :])
                if (current_value < y_ts[i]):
                    x_ts[i, :] = np.copy(U_ts[i, :])
                    y_ts[i] = current_value

                if plot:
                    P_values[i].append(y_ts[i])
                    P_positions[i].append(x_ts[i, 0])

            G_best_values.append(y_ts.min())

        G_best_position = x_ts[y_ts.argmin(), :]  # global best position
        G_best_value = y_ts.min()  # global best value


        if plot:
            self.plot_combination(P_values, 'y vs iter in f_2 function with DE method')
            if n == 1:
                self.plot_combination(P_positions, 'x vs iter in f_1 function with DE method', plot_index=3)

        print('DE search result, the minimum = {} when x = {}'.format(G_best_value, G_best_position))
        return G_best_value, G_best_position, G_best_values

    def plot_combination(self, P_ys, name, plot_index=0):
        plt.figure(plot_index)
        for i in range(len(P_ys)):
            plt.plot(np.array(range(len(P_ys[i]))), P_ys[i])

        plt.title(name)
        plt.savefig('../image/{}.png'.format(name), dpi=150)


    def plot_singular(self, P_y, name, plot_index=0):
        plt.figure(plot_index)
        plt.plot(np.array(range(len(P_y))), P_y, 'b')
        plt.title(name)
        plt.savefig('../image/{}.png'.format(name), dpi=150)


if __name__ == '__main__':
    print('DE algorithm')
