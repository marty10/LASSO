import timeit
import numpy as np
from scipy.spatial.distance import pdist, squareform


class HSIC():
    def __init__(self):
        self.KL_HSIC = 0
        self.KK_HSIC = 0
        self.H = 0
        self.L = 0

    def computeKernels(self, X, j):
        Kj = self.gaussian_kernel(X[:, j].transpose(), 1.0)
        centered_Kj = self.H * Kj * self.H
        return centered_Kj

    def HSIC_Measures(self,X,Y):
        n,p = X.shape
        self.H = np.eye(n) - 1. / n * np.ones(n)
        self.L = self.H * self.gaussian_kernel(Y, 1.0) * self.H
        KL_HSIC = np.empty([p, 1])
        KK_HSIC = np.empty([p, p])
        for j in range(0, p):
            if j%100==0:
                print("Computing HSIC measure for feature ",j )
            KL_HSIC[j] = np.trace(np.dot(self.computeKernels(X,j), self.L))
            for l in range(0, p):
                t_1 =  timeit.default_timer()
                a = self.computeKernels(X,j)
                b = self.computeKernels(X,l)
                print("computeKernels", timeit.default_timer()-t_1)
                t_2 = timeit.default_timer()
                d = np.dot(a, self.computeKernels(X,l))
                print("dotProduct", timeit.default_timer()-t_2)
                t_3 = timeit.default_timer()
                KK_HSIC[j, l] = np.trace(d)
                print("trace time ", timeit.default_timer-t_3)
                print("trace total", timeit.default_timer()-t_1)
                print ()
        self.KL_HSIC = KL_HSIC
        self.KK_HSIC = KK_HSIC
        return KK_HSIC, KL_HSIC


    def gaussian_kernel(self, x, sigma):
        x_matrix = np.reshape(x, (x.shape[0], 1))
        return np.exp((-squareform(pdist(x_matrix, 'sqeuclidean')))) / (2 * sigma ** 2)

