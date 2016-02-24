import abc
from abc import ABCMeta
import numpy as np
import numpy.linalg as li

class Kernel:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def transform(self, X):
        """kernel transformation"""


class Gaussian_kernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma

    def transform(self,x):
        n,m = x.shape
        x_transf = np.zeros([m,m])

        for l in range(0,m):
            x_l = x[:,l]
            for k in range(l,m):
                x_k = x[:,k]
                x_transf[l,k] = np.exp(-self.sigma*li.norm(x_l-x_k)**2/ 2)
                if k!=l:
                    x_transf[k,l] = x_transf[l,k]
        return x_transf

    def transform_y(self,x,y):
        n,m = x.shape
        y_transf = np.zeros([m,1])
        for l in range(0,m):
            x_l = x[:,l]
            y_transf[l] = np.exp(-self.sigma*li.norm(x_l-y)**2/ 2)
        return y_transf


class Scalar_kernel(Kernel):
    def __init__(self):
        pass

    def transform(self,x):
        n,m = x.shape
        x_transf = np.zeros([m,m])
        dict_ = dict.fromkeys(np.arange(0,49),np.array([]))

        for l in range(0,m):
            x_l = x[:,l]
            for k in range(l,m):
                x_k = x[:,k]
                x_transf[l,k] = np.dot(x_l,x_k)
                if k!=l:
                    x_transf[k,l] = x_transf[l,k]
        for key in (list)(dict_.keys()):
            dict_[key] = np.append(dict_[key],np.arange(key*12,key*12+12))
        return x_transf, dict_

    def transform_y(self,x,y):
        n,m = x.shape
        y_transf = np.array([])
        for l in range(0,m):
            x_l = x[:,l]
            y_transf = np.append(y_transf,np.dot(x_l,y))
        return y_transf



