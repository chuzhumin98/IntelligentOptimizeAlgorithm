import numpy as np
import matplotlib.pyplot as plt

from math import sin
from math import cos
from math import pi
from random import random

def function_1(x):
    return 2 * x + 3 * sin(4*x) + 1 * cos(5*x)

def support_set_1(x):
    return (x >= 0) and (x <= 2*pi)

def generate_initial_point_1():
    return np.array([random() * 2 * pi])

def generate_initial_points_1(m):
    return np.random.rand(m, 1) * 2 * pi

def generate_initial_velocity_1(m, V_max):
    return (2 * np.random.rand(m, 1) - 1) * 2 * pi

def plot_f1():
    x = np.array(range(10001)) / 10000 * 2 * pi
    y = np.array([function_1(x_i) for x_i in x])

    plt.figure()
    plt.plot(x, y, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('../../image/f1.png', dpi=150)

if __name__ == '__main__':
    plot_f1()
