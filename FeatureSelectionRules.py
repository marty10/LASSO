from abc import ABCMeta
import abc
import numpy as np
import numpy.linalg as li
import scipy
from scipy.spatial.distance import squareform, pdist


class FeatureSelectionRule:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def apply_rule(self, x, y):
        """apply rule"""

    @abc.abstractmethod
    def apply_wrapper_rule(self, x, y, beta):
        """apply wrapper rule"""


class DPP_rule(FeatureSelectionRule):
    def __init__(self, degree):
        self.degree = degree

    def apply_rule(self, x, y):
        p = x.shape[1]
        lambda_max = np.max(abs(np.dot(x.transpose(), y)))
        print("lambda_max", lambda_max)

        lambda_opt = 411000
        print("lambda_opt", lambda_opt)
        index = []

        for j in range(p):
            corr = abs(np.dot(self.x[:, j].transpose(), y))
            # print("corr",a)
            # print("valu",lambda_max - li.norm(XTrain[:,j]) * li.norm(YTrain) * (lambda_max - lambda_opt) / lambda_opt)
            if corr < lambda_max - li.norm(self.X[:, j]) * li.norm(y) * (lambda_max - lambda_opt) / lambda_opt:
                index.append(j)

        print(len(index))
        x = scipy.delete(x, index, 1)
        return x

    def apply_wrapper_rule(self, x, y, beta):
        """apply wrapper rule"""


class null_rule(FeatureSelectionRule):
    def __init__(self):
        pass

    def apply_rule(self, x, y):
        # onesVec = np.ones([len(X), 1])
        # X = np.append(onesVec, X, axis=1)
        return x

    def apply_wrapper_rule(self, x, y, beta):
        """apply wrapper rule"""


class dist_corr_Criterion(FeatureSelectionRule):
    def __init__(self,active_set):
        self.active_set = active_set

    def apply_rule(self, corr):
        j_max = np.argmax(np.max(corr,axis=1))
        if len(self.active_set)!=0:
            i=1
            indexes = np.argsort(corr, axis = 0)
            #print(indexes)
            while self.active_set.__contains__(j_max) and i<=len(indexes):
                j_max = indexes[len(indexes)-i, 0]
                i+=1

        return j_max



    def apply_wrapper_rule(self, corr, beta):
        corr_beta = corr*beta
        j_max = np.argmax(np.max(corr_beta,axis=1))
        if len(self.active_set)!=0:
            i=1
            indexes = np.argsort(corr_beta, axis = 0)
            #print(indexes)
            while self.active_set.__contains__(j_max) and i<=len(indexes):
                j_max = indexes[len(indexes)-i, 0]
                i+=1

        return j_max


class HSIC_Criterion(FeatureSelectionRule):
    def __init__(self, active_set):
        self.active_set = active_set

    def apply_rule(self, x, y):
        """apply rule"""

    def apply_wrapper_rule(self, KK_HSIC, KL_HSIC, beta):
        beta = np.reshape(beta, (beta.shape[0], 1))
        corr = KL_HSIC - np.dot(KK_HSIC, beta)
        j_max = np.argmax(np.max(corr,axis=1))
        if len(self.active_set)!=0:
            i=1
            indexes = np.argsort(corr, axis = 0)
            while self.active_set.__contains__(j_max) and i<=len(indexes):
                j_max = indexes[len(indexes)-i, 0]
                i+=1
        return j_max


