from sklearn import cross_validation
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
#import  numpy as np
from ExtractDataset import Dataset
from ExtractResult import Result

# ## LASSO CV                                              VVCVBNN CC
### dataset
from sklearn.metrics import mean_squared_error

# lasso_cv.fit(X[train], y[train])
# alphas = np.logspace(-4, -.5, 30)
# k_fold = cross_validation.KFold(len(X), n_folds=3)

###########model selection
# lasso_cv = linear_model.LassoCV(eps=0.001, n_alphas=100, alphas=None, precompute='auto', max_iter=100000000,
#                                 tol=0.0001, cv=None, verbose=False, n_jobs=1, positive=False)
# lasso_cv.fit(XTrain, YTrain)
# print lasso_cv.alpha_

# lassoCoordDescent = linear_model.Lasso(lasso_cv.alpha_)



# linear = linear_model.LinearRegression(normalize=True)
# linear.fit(XTrain, YTrain)
# ypred = linear.predict(XTest)
# print(mean_squared_error(YTest, ypred)
# ## CROSS VALIDATION BUONA!!
#alphas = np.linspace(0, 20, 10)
#parameters = {"alpha": alphas}
#clf = GridSearchCV(lasso, parameters, fit_params = {"verbose" : False}, cv=3, scoring="mean_squared_error")
#clf.fit(XTrainTransf, YTrain)
#lambda_opt = clf.best_params_




# ## CROSS VALIDATION A MANO
from sklearn import cross_validation
from sklearn import linear_model, datasets
#import numpy as np
from FeatureSelectionRules import FeatureSelectionRule, dist_corr
from LARS_CGDrun import ShootingAlgorithm
from numpy.linalg import inv

# ## dataset
# diabetes = datasets.load_diabetes()
# X = diabetes.data
# Y = diabetes.target
#
# # ##add column for beta_0
# Y = Y.transpose()  # nx1
# onesVec = np.ones([len(X), 1]);
# X = np.append(onesVec, X, axis=1)
#
# # ##
# tol = 0.0001
#
# # ## split train and test
# k_fold = cross_validation.KFold(len(X), 3)
# for k, (train, test) in enumerate(k_fold):
#     XTrain = X[train]
#     YTrain = Y[train]
#     XTest = X[test]
#     YTest = Y[test]

#alphas = np.logspace(-4, -.5, 30)
# lambdas = np.array([5,6])
# k_foldCross = cross_validation.KFold(len(XTrain), 3)
# err_vector = np.empty([3,3])
# shoot = ShootingAlgorithm()
#
# for k, (train, val) in enumerate(k_foldCross):
#     XTrainVal = XTrain[train]
#     YTrainVal = YTrain[train]
#     XTestVal = XTrain[val]
#     YTestVal = YTrain[val]
#     XTransp = XTrainVal.transpose()
#     XInv = inv(np.dot(XTransp, XTrainVal))
#     beta = np.dot(np.dot(XInv, XTransp), YTrainVal)
#     count=0
#     mean_std = np.empty([len(lambdas),2])
#     for alpha in lambdas:
#         count += 1
#         num_iter, timeit, beta_est = shoot.shooting_algorithm(XTrain,YTrain,beta,alpha,tol)
#         Ypred = np.dot(XTestVal, beta_est)
#         err_vector[:count] = np.dot(YTest-Ypred, YTest-Ypred)/len(YTest)
#         mean_std[count,0] = np.mean(err_vector[:,count])
#         mean_std[count,1] = np.std(err_vector[:,count])
#     max_mean = max(mean_std[:,0])
#     index_max = mean_std[:,0].argmax()
#     err_vector[k,0] =  lambdas(index_max)
#     err_vector[k,1] =  max_mean
#     err_vector[k,2] =  mean_std[index_max,1]
#
# print (err_vector)

from ExtractDataset import Dataset

# n_samples = 2000
# n_features = 6000
# n_informative = 1000
#
# #transformation = PolinomialTransformation(degree = 2)
# dataset = Dataset(n_samples, n_features, n_informative = n_informative)#, transformation = transformation)
#
# XTrain = dataset.XTrain
# YTrain = dataset.YTrain
# XTest = dataset.XTest
# YTest = dataset.YTest
import numpy
dist_cor = numpy.load("dist_cor.pkl")
print(dist_cor)