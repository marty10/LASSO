import numpy as np
from scipy.spatial.distance import pdist, squareform


class HSIC():
    def __init__(self):
        self.K = 0
        self.L = 0
        self.H = 0
        self.KL_HSIC = 0
        self.KK_HSIC = 0

    def computeKernels(self, X, Y):
        n, p = X.shape
        self.K = np.zeros([p, n, n])
        self.H = np.eye(n) - 1. / n * np.ones(n)  # centering_matrix

        for j in range(0, p):
            Kx = self.gaussian_kernel(X[:, j].transpose(), 1.0)
            self.K[j, :, :] = self.H * Kx * self.H
        self.L = self.H * self.gaussian_kernel(Y, 1.0) * self.H

    def HSIC_Measure_KL(self):
        p = self.K.shape[0]
        KL_HSIC = np.empty([p, 1])
        for j in range(0, p):
            KL_HSIC[j] = np.trace(np.dot(self.K[j, :, :], self.L))
        self.KL_HSIC = KL_HSIC
        return KL_HSIC

    def HSIC_Measure_KK(self):
        p = self.K.shape[0]
        KK_HSIC = np.empty([p, p])
        for j in range(0, p):
            for l in range(0, p):
                KK_HSIC[j, l] = np.trace(np.dot(self.K[j, :, :], self.K[l, :, :]))
        self.KK_HSIC = KK_HSIC
        return KK_HSIC

    def gaussian_kernel(self, x, sigma):
        x_matrix = np.reshape(x, (x.shape[0], 1))
        return np.exp((-squareform(pdist(x_matrix, 'sqeuclidean')))) / (2 * sigma ** 2)

