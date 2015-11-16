from abc import ABCMeta
import abc
import numpy as np
__author__ = 'Martina'


class Transformation:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def transform(self, X):
        """fit poly"""


class PolinomialTransformation(Transformation):
    def __init__(self, degree):
        self.degree = degree

    def transform(self,x):
        n,m = x.shape
        x_transf = np.empty([n, m*(self.degree)])
        for d in range(0, self.degree):
            x_transf[:,m*(d):m*(d)+m] = np.power(x,d+1)
        return x_transf


class NullTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self,X):
        #onesVec = np.ones([len(X), 1])
        #X = np.append(onesVec, X, axis=1)
        return X
