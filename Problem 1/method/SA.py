import numpy as np
from random import random
import math

class SA:
    def __init__(self, T_0=100, T_min=1e-8, iter_num=100, decay_rate=0.98):
        self.T_0 = T_0
        self.T_min = T_min
        self.iter_num = iter_num
        self.decay_rate = decay_rate

    def optimize(self, function, support_set, x_0): # support_set(x): true for x in support set
        n = len(x_0) # the deminsion of x
        xs = [x_0] # record the process of x's change
        ys = [function(x_0)] # record the process of y's change
        t = self.T_0 # set for the current temperature
        while t > self.T_min:
            for k in range(self.iter_num):
                x_new = xs[-1] + (2 * np.random.rand(n) - 1) * t
                if (support_set(x_new)): # to confirm the new point is in the support set
                    y_new = function(x_new)
                    if y_new < ys[-1]: # compare with the current point
                        xs.append(x_new)
                        ys.append(y_new)
                    else:
                        prob = 1 / (1 + math.exp((y_new - ys[-1]) / self.T_0))
                        if (random() < prob): # accept the new point with P = prob
                            xs.append(x_new)
                            ys.append(y_new)

            t *= self.decay_rate

        xs = np.array(xs)
        ys = np.array(ys)
        best_result = np.min(ys)
        best_index = np.argmin(ys)

        print('SA search result, the minimum = {} when x = {}'.format(best_result, xs[best_index, :]))
        return best_result, xs[best_index, :], xs, ys





if __name__ == '__main__':
    print('SA algorihtm')