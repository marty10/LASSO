# coding=utf-8
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def compute_weightedLASSO(lasso, XTrain_current, YTrain, XTest_current, YTest, scoring, score_f, verbose, values_TM):
    # values_TM è una matrice contenente i valori di t e m per il train e il test
    alphas = _alpha_grid(XTrain_current, YTrain, fit_intercept=False)
    parameters = {"alpha": alphas}

    clf = GridSearchCV(lasso, parameters, fit_params = {"verbose" : False}, cv=3, scoring=scoring)
    clf.fit(XTrain_current, YTrain)
    lambda_opt = clf.best_params_

    print("best lambda", lambda_opt)

    lasso.set_params(**lambda_opt)
    lasso.fit(XTrain_current,YTrain)

    y_pred_train = lasso.predict(XTrain_current)
    mse_train = score_f(YTrain, y_pred_train)

    y_pred_test = lasso.predict(XTest_current)
    mse_test = score_f(YTest, y_pred_test)

    if verbose:
        print("mse_train "+lasso.__class__.__name__,mse_train)
        print ("mse_test weights "+lasso.__class__.__name__,mse_test)
        print("mae train",100*mean_absolute_error(YTrain,y_pred_train)/89.7)
        print("mae test",100*mean_absolute_error(YTest,y_pred_test)/89.7)

        print("mse train",100*np.sqrt(mean_squared_error(YTrain,y_pred_train))/89.7)
        print("mse test",100*np.sqrt(mean_squared_error(YTrain,y_pred_train))/89.7)

    ##values[0] = 24, values[1] = 281, values[2] = 214
    if len(values_TM)!=0:
        abs_error_train = 100*mean_absolute_error(YTrain,y_pred_train)*len(YTrain)/(89.7 * values_TM[0, 0] * values_TM[0, 1])

        abs_error_test = 100*mean_absolute_error(YTest,y_pred_test)*len(YTest)/(89.7 * values_TM[1, 0] * values_TM[1,1])

        mse_error_train = 100.*np.sqrt(mean_squared_error(YTrain,y_pred_train)*len(YTrain)/(values_TM[0, 0] * values_TM[0, 1]))/(89.7)
        print("mean squared error train", mse_error_train )

        mse_error_test = 100.*np.sqrt(mean_squared_error(YTest,y_pred_test)*len(YTest)/(values_TM[1, 0] * values_TM[1, 1]))/(89.7)
        print("mean squared error test", mse_error_test )

        if verbose:
            print("abs test", abs_error_test)
            print("abs train", abs_error_train)

    return mse_test, lasso.beta


def compute_lasso(XTrain, YTrain, XTest, YTest, score,values_TM):
    lasso_cv = linear_model.LassoCV(fit_intercept=False,  max_iter=100000, n_jobs = -1)
    lasso_cv.fit(XTrain,YTrain)
    best_alpha = lasso_cv.alpha_

    model = linear_model.Lasso(fit_intercept=False,alpha=best_alpha)
    new_loss,beta = compute_mse(model, XTrain, YTrain,XTest, YTest, score,values_TM)

    return new_loss,beta

def compute_mse(model,x_train_current_tmp,YTrain,x_test_current_tmp,YTest, score ,values_TM = []):
    model.fit(x_train_current_tmp, YTrain)
    y_pred_train = model.predict(x_train_current_tmp)
    y_pred_test = model.predict(x_test_current_tmp)

    if len(values_TM)!=0:
        abs_error_train = 100.*mean_absolute_error(YTrain,y_pred_train)*len(YTrain)/(89.7* values_TM[0, 0] * values_TM[0,1])
        print("abs train", abs_error_train)

        abs_error_test = 100.*mean_absolute_error(YTest,y_pred_test)*len(YTest)/(89.7* values_TM[1, 0] * values_TM[1,1])
        print("abs test", abs_error_test)

        mse_error_train = 100.*np.sqrt(mean_squared_error(YTrain,y_pred_train)*len(YTrain)/(values_TM[0, 0] * values_TM[0, 1]))/(89.7)
        print("mean squared error train", mse_error_train )

        mse_error_test = 100.*np.sqrt(mean_squared_error(YTest,y_pred_test)*len(YTest)/(values_TM[1, 0] * values_TM[1, 1]))/(89.7)
        print("mean squared error test", mse_error_test )

    if score=="mean_squared_error":
        new_loss = mean_squared_error(YTest,y_pred_test)
    elif score== "mean_absolute_error":
        new_loss = mean_absolute_error(YTest,y_pred_test)
    else:
        new_loss = r2_score(YTest,y_pred_test)
    beta = model.coef_

    if x_train_current_tmp.shape[1]==1:
        beta = np.array([beta])
    beta = beta.reshape([len(beta),1])

    return new_loss, beta