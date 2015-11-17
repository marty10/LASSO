import timeit
from scipy.spatial.distance import squareform, pdist
import numpy as np


class DistanceCorrelation():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.dcor = 0

    def compute_dist(self, x_j):
        x_j = np.reshape(x_j.transpose(),(x_j.transpose().shape[0], 1))
        dist = squareform(pdist(x_j))
        mean_dist = dist - dist.mean(axis=0)[None, :] - dist.mean(axis=1)[:, None] + dist.mean()
        return mean_dist

    def compute_dcov(self,mean_dist1,mean_dist2):
        n = self.y.shape[0]
        dcov2 = (mean_dist1 * mean_dist2).sum() / float(n * n)
        return dcov2

    def compute_dist_corr(self):
        n,p = self.X.shape
        dcor = np.empty([p,1])
        t_1 = timeit.default_timer()
        mean_dist_y = self.compute_dist(self.y)
        dcov2_yy = self.compute_dcov(mean_dist_y,mean_dist_y)
        sqrt_dcov2_yy = np.sqrt(dcov2_yy)
        print("time yy", timeit.default_timer()-t_1)
        for j in range(0, p):
            #t_2 = timeit.default_timer()
            #t_3 = timeit.default_timer()
            x_j = self.X[:,j]
            mean_dist_x_j = self.compute_dist(x_j)
            #print("compute mean_dist", timeit.default_timer()-t_3)
            t_4 = timeit.default_timer()
            dcov2_xy = self.compute_dcov(mean_dist_x_j, mean_dist_y)
            #print("compute d_cov", timeit.default_timer()-t_4)
            dcov2_xx = self.compute_dcov(mean_dist_x_j, mean_dist_x_j)
            dcor[j] = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * sqrt_dcov2_yy)
            # a = 0
            # for l in range(0,p):
            #     x_l = np.reshape(self.X[:, l].transpose(),(self.X[:, l].transpose().shape[0], 1))
            #     mean_dist_x_l = self.compute_dist(x_l)
            #     dcov2_xjxl = self.compute_dcov(mean_dist_x_j, mean_dist_x_l)
            #     dcov2_xl_xl = self.compute_dcov(mean_dist_x_l, mean_dist_x_l)
            #     a = a+np.sqrt( dcov2_xjxl) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_xl_xl))
            # dcor[j] = dcor[j]-a
        self.dcor = dcor
        return dcor
