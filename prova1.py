import scipy
from sklearn.externals import joblib
from ExtractDataset import Dataset
from LASSOModel import LASSOEstimator, ISTA, FISTA, ADMM, Shooting
from Transformation import PolinomialTransformation
from sklearn.metrics import mean_squared_error
import numpy as np

dist_cor = joblib.load("distance_cor_poly.pkl")
print(dist_cor)

first_1000 = dist_cor[:1000]
b = [x for x in first_1000 if x<=0.02]
print(len(b))
second_1000 = dist_cor[2000:3000]
c = [x for x in second_1000 if x<=0.02]
print(len(c))
a = [x for x in dist_cor if x<=0.02]
indices = np.where(dist_cor<=0.02)
print("max",max(dist_cor))
print("min", min(dist_cor))
print(indices)
print(len(a))



n_samples = 2000
n_features = 2000
n_informative = 1000

transformation = PolinomialTransformation(degree = 2)
dataset = Dataset(n_samples,n_features,n_informative = n_informative, transformation = transformation)

XTrain = dataset.XTrain
YTrain = dataset.YTrain
XTest = dataset.XTest
YTest = dataset.YTest

XTrainTransf = dataset.XTrainTransf
XTrainTransf = scipy.delete(XTrainTransf, indices, 1)

XTestTransf = dataset.XTestTransf
XTestTransf = scipy.delete(XTestTransf, indices, 1)
lambda_opt = {"alpha": 67.1590893061}
folder = "AlgorithmResults/FeatureSelection/Polynomial2/"

model_list = {ISTA(), FISTA(), ADMM(), Shooting()}
ext_data = ".npz"
ext_model = ".pkl"

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