from abc import ABCMeta
import abc
import timeit
import numpy as np
from scipy.spatial.distance import pdist, squareform


class FeatureMeasures:
    __metaclass__ = ABCMeta


    @abc.abstractmethod
    def compute_measures(self, x, y, **params):
        """fit"""

    def apply_wrapper_rule(self, corr, beta, active_set):
        beta = np.reshape(beta, (beta.shape[0], 1))
        corr_beta = corr*beta
        j_max = np.argmax(np.max(corr_beta,axis=1))
        if len(active_set)!=0:
            i=1
            indexes = np.argsort(corr_beta, axis = 0)
            #print(indexes)
            while active_set.__contains__(j_max) and i<=len(indexes):
                j_max = indexes[len(indexes)-i, 0]
                i+=1

        return j_max


class HSIC(FeatureMeasures):
    def __init__(self):
        self.KL_HSIC = 0
        self.KK_HSIC = 0
        self.H = 0
        self.L = 0

    def computeKernels(self, X, j):
        Kj = self.gaussian_kernel(X[:, j].transpose(), 1.0)
        centered_Kj = self.H * Kj * self.H
        return centered_Kj

    def compute_measures(self, X, Y):
        n, p = X.shape
        self.H = np.eye(n) - 1. / n * np.ones(n)
        self.L = self.H * self.gaussian_kernel(Y, 1.0) * self.H
        KL_HSIC = np.empty([p, 1])
        KK_HSIC = np.empty([p, p])
        for j in range(0, p):
            if j % 100 == 0:
                print("Computing HSIC measure for feature ", j)
            KL_HSIC[j] = np.trace(np.dot(self.computeKernels(X, j), self.L))
            # for l in range(0, p):
            #     t_1 =  timeit.default_timer()
            #     a = self.computeKernels(X,j)
            #     b = self.computeKernels(X,l)
            #     print("computeKernels", timeit.default_timer()-t_1)
            #     t_2 = timeit.default_timer()
            #     d = np.dot(a, self.computeKernels(X,l))
            #     print("dotProduct", timeit.default_timer()-t_2)
            #     t_3 = timeit.default_timer()
            #     KK_HSIC[j, l] = np.trace(d)
            #     print("trace time ", timeit.default_timer-t_3)
            #     print("trace total", timeit.default_timer()-t_1)
            #     print ()
        self.KL_HSIC = KL_HSIC
        # self.KK_HSIC = KK_HSIC
        return KK_HSIC  # , KL_HSIC

    def gaussian_kernel(self, x, sigma):
        x_matrix = np.reshape(x, (x.shape[0], 1))
        return np.exp((-squareform(pdist(x_matrix, 'sqeuclidean')))) / (2 * sigma ** 2)

class DistanceCorrelation(FeatureMeasures):
    def __init__(self):
        self.dcor = 0

    def compute_dist(self, x_j):
        x_j = np.reshape(x_j.transpose(),(x_j.transpose().shape[0], 1))
        dist = squareform(pdist(x_j))
        mean_dist = dist - dist.mean(axis=0)[None, :] - dist.mean(axis=1)[:, None] + dist.mean()
        return mean_dist

    def compute_dcov(self,mean_dist1,mean_dist2):
        n = mean_dist1.shape[0]
        dcov2 = (mean_dist1 * mean_dist2).sum() / float(n * n)
        return dcov2

    def compute_measures(self, X, y):
        n,p = X.shape
        dcor = np.empty([p,1])

        mean_dist_y = self.compute_dist(y)
        dcov2_yy = self.compute_dcov(mean_dist_y,mean_dist_y)
        sqrt_dcov2_yy = np.sqrt(dcov2_yy)

        for j in range(0, p):
            #t_2 = timeit.default_timer()

            x_j = X[:,j]
            mean_dist_x_j = self.compute_dist(x_j)

            #print("compute mean_dist", timeit.default_timer()-t_3)

            #t_4 = timeit.default_timer()
            dcov2_xy = self.compute_dcov(mean_dist_x_j, mean_dist_y)
            #print("compute d_cov", timeit.default_timer()-t_4)
            dcov2_xx = self.compute_dcov(mean_dist_x_j, mean_dist_x_j)
            dcor[j] = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * sqrt_dcov2_yy)
        self.dcor = dcor
        return dcor