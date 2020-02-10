import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def function_2(x):
    return np.sum(np.square(x)) / 4000 - np.prod(np.cos(x / np.sqrt(np.array(range(1, len(x)+1))))) + 1


def support_set_2(x):
    return True


def generate_initial_point_2(n):
    # n is the deminsion
    return np.random.randn(n)


def generate_initial_points_2(m, n):
    return np.random.randn(m, n)


def generate_initial_velocity_2(m, n, V_max):
    return (2 * np.random.rand(m, n) - 1) * V_max


def plot_f2():
    size = 201
    X = (np.array(range(size)) - (size - 1) / 2) / ((size - 1) / 20)
    Y = (np.array(range(size)) - (size - 1) / 2) / ((size - 1) / 20)

    X, Y = np.meshgrid(X, Y)

    Z = np.zeros([size, size])

    for i in range(size):
        for j in range(size):
            Z[i, j] = function_2(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    plt.draw()
    plt.savefig('3D.jpg')


if __name__ == '__main__':
    plot_f2()
