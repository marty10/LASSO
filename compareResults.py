from sklearn.externals import joblib
from ExtractDataset import Dataset
from LASSOModel import LASSOEstimator, ISTA, FISTA, ADMM, Shooting, modifiedShooting, modifiedShooting2
from Transformation import PolinomialTransformation
from sklearn.metrics import mean_squared_error
import numpy as np


n_samples = 2000
n_features = 6000
n_informative = 1000

#transformation = PolinomialTransformation(degree = 2)
dataset = Dataset(n_samples,n_features,n_informative = n_informative)#transformation = transformation)

XTrain = dataset.XTrain
YTrain = dataset.YTrain
XTest = dataset.XTest
YTest = dataset.YTest


XTrainTransf = dataset.XTrainTransf
XTestTransf = dataset.XTestTransf

lambda_opt = {"alpha": 67.1590893061}

#model_list = {ISTA(), FISTA(), Shooting(), ADMM()}
model_list = {modifiedShooting2()}
ext_data = ".npz"
ext_model = ".pkl"
folder = "AlgorithmResults/"

for model in model_list:
    lasso = LASSOEstimator(model)
    lasso.set_params(**lambda_opt)
    lasso.fit(XTrainTransf,YTrain)

    y_pred_test = lasso.predict(XTestTransf)
    mse_test = mean_squared_error(YTest, y_pred_test)
    print ("mse_test "+model.__class__.__name__,mse_test)

    y_pred_train = lasso.predict(XTrainTransf)
    mse_train = mean_squared_error(YTrain, y_pred_train)

    print("mse_train "+model.__class__.__name__,mse_train)

    np.savez(folder+model.__class__.__name__+ext_data, XTrain=XTrain, YTrain = YTrain, mse_test=mse_test, XTest=dataset.XTest, YTest = YTest, y_pred_test=y_pred_test,
         XTrainTransf=XTrainTransf, XTestTransf=XTestTransf, mse_train = mse_train)

    joblib.dump(lasso, folder+model.__class__.__name__+'_model'+ext_model, compress=9)