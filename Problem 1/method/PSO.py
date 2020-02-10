import numpy as np
from random import random

import matplotlib.pyplot as plt

class PSO:
    def __init__(self, omega=1, C_1=2, C_2=2, T_max=500, V_max = 0.5, beta = 0.5):
        self.omega = omega
        self.C_1 = C_1
        self.C_2 = C_2
        self.T_max = T_max
        self.V_max = V_max # the velocity upper bound
        self.beta = beta # search decay rate

    def optimize(self, function, support_set, x_0s, v_0s, plot=False):
        # support_set(x): true for x in support set
        # x_0s: the initial particles' position, with format [m, n], m is particles number, n is dim
        # v_0s: the initial particles' velocity, same format with x_0s
        m, n = x_0s.shape
        P_best_position = np.copy(x_0s) # each particle's best position
        P_best_value = np.array([function(x_0s[i, :]) for i in range(m)]) # each particle's best value
        G_best_position = P_best_position[P_best_value.argmin(), :] # global best position
        G_best_value = P_best_value.min() # global best value

        if plot:
            P_positions = []
            P_values = []
            for i in range(m):
                P_positions.append([x_0s[i, 0]])
                P_values.append([function(x_0s[i, :])])

        x_ts = np.copy(x_0s) # current position
        v_ts = np.copy(v_0s) # current velocity

        G_best_values = [G_best_value] # record the change of G_best_value

        for t in range(self.T_max):
            for i in range(m):
                v_ts[i, :] = self.omega * v_ts[i, :] + self.C_1 * random() * (P_best_position[i, :] - x_ts[i, :]) + self.C_2 * random() * (G_best_position - x_ts[i, :])
                v_ts[i, :] = np.maximum(-self.V_max, v_ts[i, :]) # to satisfy the constraints
                v_ts[i, :] = np.minimum(self.V_max, v_ts[i, :])

                rate = 1
                while True: # search for a valid rate
                    if (support_set(x_ts[i, :] + rate * v_ts[i, :])):
                        break
                    else:
                        rate *= self.beta

                x_ts[i, :] += rate * v_ts[i, :]
                current_value = function(x_ts[i, :])

                if plot:
                    P_positions[i].append(x_ts[i, 0])
                    P_values[i].append(current_value)

                if current_value < P_best_value[i]: # update for the best value of this particle
                    P_best_value[i] = current_value
                    P_best_position[i, :] = np.copy(x_ts[i, :])

            current_best_value = P_best_value.min()
            if (current_best_value < G_best_value): # update for the global best value
                G_best_value = current_best_value
                G_best_position = P_best_position[P_best_value.argmin(), :]

            G_best_values.append(G_best_value)

        if plot:
            self.plot_combination(P_values, 'y vs iter in f_2 function with PSO method')
            self.plot_singular(P_values[P_best_value.argmin()], 'y vs iter in f_2 function with PSO method at best particle', plot_index=1)
            self.plot_singular(P_values[P_best_value.argmax()], 'y vs iter in f_2 function with PSO method at worst particle', plot_index=2)
            if n == 1:
                self.plot_combination(P_positions, 'x vs iter in f_1 function with PSO method', plot_index=3)
                self.plot_singular(P_positions[P_best_value.argmin()], 'x vs iter in f_1 function with PSO method at best particle', plot_index=4)
                self.plot_singular(P_positions[P_best_value.argmax()], 'x vs iter in f_1 function with PSO method at worst particle', plot_index=5)


        print('PSO search result, the minimum = {} when x = {}'.format(G_best_value, G_best_position))
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
    print('PSO algorithm')



