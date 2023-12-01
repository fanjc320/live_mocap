# https://zhuanlan.zhihu.com/p/439723957

import numpy as np
import matplotlib.pyplot as plt


def gaussian_w(xi, x0, r=0.1):
    return np.exp(-(xi-x0)**2/r**2)


def generate_data():
    xx = np.linspace(0, 1, 500)
    yy = np.sin(8 * np.pi * xx) + np.random.randn(500) * 0.2
    return xx, yy


class LocalRegress:
    def __init__(self, kernel, basis):
        self.kernel = kernel
        self.basis = basis

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.phi = self.basis(x)

    def predict(self, x):
        W = self.kernel(self.x, x)
        A = self.phi @ (W * self.phi).T
        b = (W * self.phi) @ self.y
        a = np.linalg.solve(A, b)
        return np.sum(self.basis(x).T * a)

basis = lambda x: np.stack([np.ones_like(x), np.array(x).copy()])

xx, yy = generate_data()
# lr = LocalRegress(kernel=lambda xi, x0: w(xi, x0, r=0.05), basis=basis)
lr = LocalRegress(kernel=lambda xi, x0: gaussian_w(xi, x0, r=0.05), basis=basis)
lr.fit(xx, yy)
y = [lr.predict(x) for x in np.linspace(0, 1, 1000)]