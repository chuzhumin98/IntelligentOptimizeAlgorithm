import numpy as np
from random import random


class PSO:
    def __init__(self, omega=1, C_1=2, C_2=2, T_max=500, V_max = 0.5, beta = 0.5):
        self.omega = omega
        self.C_1 = C_1
        self.C_2 = C_2
        self.T_max = T_max
        self.V_max = V_max # the velocity upper bound
        self.beta = beta # search decay rate

    def optimize(self, function, support_set, x_0s, v_0s):
        # support_set(x): true for x in support set
        # x_0s: the initial particles' position, with format [m, n], m is particles number, n is dim
        # v_0s: the initial particles' velocity, same format with x_0s
        m, n = x_0s.shape
        P_best_position = np.copy(x_0s) # each particle's best position
        P_best_value = np.array([function(x_0s[i, :]) for i in range(m)]) # each particle's best value
        G_best_position = P_best_position[P_best_value.argmin(), :] # global best position
        G_best_value = P_best_value.min() # global best value

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
                if current_value < P_best_value[i]: # update for the best value of this particle
                    P_best_value[i] = current_value
                    P_best_position[i, :] = np.copy(x_ts[i, :])

            current_best_value = P_best_value.min()
            if (current_best_value < G_best_value): # update for the global best value
                G_best_value = current_best_value
                G_best_position = P_best_position[P_best_value.argmin(), :]

            G_best_values.append(G_best_value)

        print('PSO search result, the minimum = {} when x = {}'.format(G_best_value, G_best_position))
        return G_best_value, G_best_position, G_best_values


if __name__ == '__main__':
    print('PSO algorithm')



