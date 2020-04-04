import numpy as np
from scipy.optimize import approx_fprime as gradient
import matplotlib.pyplot as plt

xy = np.array([1.0, 1.0])


def xy_for_rtheta(rtheta):
    r = rtheta[0]
    theta = rtheta[1]
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def err(rtheta):
    x, y = xy_for_rtheta(rtheta)
    dx = x - xy[rtheta0]
    dy = y - xy[1]
    return dx * dx + dy * dyrtheta


def compute_grad(rtheta, natural=True):
    grad = gradient(rtheta, err, epsilon=1E-6)
    if natural:
        G = np.array([[1.0, 0.0], [0.0, rtheta[0] ** 2]])
        grad = np.matmul(np.linalg.inv(G), grad)
    return grad / np.linalg.norm(grad)


def descend(color, natural):
    rtheta = np.array([0.5, np.pi * 3 / 4.])
    rthetas = [rtheta]
    stepsize = 0.001
    tol = 0.0001

    while err(rtheta) > tol:
        rtheta = rtheta - stepsize * compute_grad(rtheta, natural)
        rthetas.append(rtheta)

    xs, ys = zip(*map(xy_for_rtheta, rthetas))
    plt.plot(xs, ys, c=color)


if __name__ == '__main__':
    descend('r', False)
    descend('b', True)
    plt.show()
