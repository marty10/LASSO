from sklearn import linear_model
from sklearn.externals import joblib
from ExtractDataset import Dataset
from sklearn.metrics import mean_squared_error
import numpy as np

# ## Dataset Extraction
from Transformation import PolinomialTransformation

n_samples = 2000
n_features = 3000
n_informative = 1000

transformation = PolinomialTransformation(degree = 2)
dataset = Dataset(n_samples,n_features,n_informative = n_informative, transformation = transformation)

X = dataset.X
XTrain = dataset.XTrain
YTrain = dataset.YTrain
XTest = dataset.XTest
YTest = dataset.YTest

XTrainTransf = dataset.XTrainTransf
XTestTransf = dataset.XTestTransf

lambda_opt = 67.1590893061/XTrainTransf.shape[0]
folder = "AlgorithmResults/Polynomial2/"
model_list = {linear_model.Lasso(alpha=lambda_opt, fit_intercept=False, tol=1e-4, max_iter=200000), linear_model.LassoLars(alpha=lambda_opt, fit_intercept=False, max_iter = 10000)}
ext_data = ".npz"
ext_model = ".pkl"


for model in model_list:
    lasso = model
    lasso.fit(XTrainTransf,YTrain)

    y_pred_test = lasso.predict(XTestTransf)
    mse_test = mean_squared_error(YTest, y_pred_test)
    print ("mse_test "+model.__class__.__name__,mse_test)

    print("beta"+model.__class__.__name__, sum(abs(lasso.coef_)))
    y_pred_train = lasso.predict(XTrainTransf)
    mse_train = mean_squared_error(YTrain, y_pred_train)
    print("mse_train "+model.__class__.__name__,mse_train)

    print("loss", 0.5 *np.sum((np.dot(XTrainTransf, lasso.coef_) - YTrain) ** 2.0)  + lambda_opt*XTrainTransf.shape[0]*sum(abs(lasso.coef_)))

    beta_informative = lasso.coef_[:1000]
    n_informative_zero1 = [x for x in beta_informative if x==0]

    print("beta = 0", len(n_informative_zero1))
    np.savez(folder+model.__class__.__name__+ext_data, XTrain=dataset.XTrain, YTrain = YTrain, mse_test=mse_test, XTest=dataset.XTest, YTest = YTest, y_pred_test=y_pred_test, best_lambda=lambda_opt,
         XTrainTransf=XTrainTransf, XTestTransf=XTestTransf, beta = lasso.coef_)

    joblib.dump(lasso,folder+model.__class__.__name__+'_model'+ext_model, compress=9)
