from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.base import center_data
from FeatureSelectionRules import null_rule
from Transformation import NullTransformation


__author__ = 'Martina'


class Dataset:
    def __init__(self, n_samples, n_features, n_informative, feat_select_rule = null_rule(), transformation=NullTransformation(), fit_intercept = True):
        self.n_samples = n_samples
        self.n_features = n_features
        X, Y, beta = datasets.make_regression(n_samples=self.n_samples, n_features=self.n_features,
                                                  n_informative=n_informative, shuffle=False, random_state=11)
        Y = Y.transpose()
        #self.XTrain, self.YTrain,_,_,_ = center_data(X, Y, fit_intercept=fit_intercept, normalize = True)
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.33,random_state=0)
        self.XTrain, self.YTrain, X_mean, y_mean, X_std = center_data(XTrain, YTrain, fit_intercept=fit_intercept, normalize = True)

        self.XTest, self.YTest = self.center_test(XTest,YTest,X_mean,y_mean,X_std)

        self.transformation = transformation
        self.feat_select_rule = feat_select_rule
        self.XTrain = self.feat_select_rule.apply_rule(self.XTrain, self.YTrain)
        #self.XTransf = self.transformation.transform(self.X)
        self.XTrainTransf = self.transformation.transform(self.XTrain)
        self.XTestTransf = self.transformation.transform(self.XTest)
        # ## onesVec = np.ones([len(self.X), 1]) self.X = np.append(onesVec, self.X, axis=1)
        self.beta = beta

    def center_test(self, X, y, X_mean, y_mean, X_std, normalize = True):
        X -= X_mean
        if normalize:
            X /= X_std
        y = y - y_mean
        return X,y