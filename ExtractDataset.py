from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.linear_model.base import center_data
from FeatureSelectionRules import null_rule
from Transformation import NullTransformation
import scipy
from scipy.spatial.distance import pdist, squareform

__author__ = 'Martina'


class Dataset:
    def __init__(self, n_samples, n_features, n_informative, transformation=NullTransformation(), feat_select_rule = null_rule(), fit_intercept = True):
        self.n_samples = n_samples
        self.n_features = n_features
        X, Y = datasets.make_regression(n_samples=self.n_samples, n_features=self.n_features,
                                                  n_informative=n_informative, shuffle=False, random_state=11)
        self.X, self.Y, _, _, _ = center_data(X, Y, fit_intercept=fit_intercept, normalize = True)

        self.Y = self.Y.transpose()
        self.transformation = transformation
        self.feat_select_rule = feat_select_rule
        self.X = self.feat_select_rule.apply_rule(self.X, self.Y)
        self.XTransf = self.transformation.transform(self.X)
        self.XTrain, self.XTest, self.YTrain, self.YTest = train_test_split(self.X, self.Y, test_size=0.33,
                                                                            random_state=0)
        self.XTrainTransf = self.transformation.transform(self.XTrain)
        self.XTestTransf = self.transformation.transform(self.XTest)
        # ## onesVec = np.ones([len(self.X), 1]) self.X = np.append(onesVec, self.X, axis=1)