from sklearn.linear_model.base import center_data
from sklearn.metrics import mean_squared_error, r2_score
from ExtractResult import Result
from LASSOModel import Shooting, LASSOEstimator
import numpy as np
import sys
from Lasso_utils import compute_lasso, compute_weightedLASSO
from utility import center_test, assign_weights, get_current_data, get_beta_div_zeros, \
    print_features_active,extract_level

sys.argv[1:] = [str(x) for x in sys.argv[1:]]
file_name = sys.argv[1]
weights_all = 1

score = "mean_squared_error"
if score=="r2_score":
    score_f = r2_score
    scoring = "r2"
else:
    score_f = mean_squared_error
    scoring = "mean_squared_error"

ext = ".npz"
file_cross_val = file_name+ext

fine_name_weights = file_name+"_ranking_not_levels"+ext
results_weighted_lasso = Result(fine_name_weights, "lasso")

mses = results_weighted_lasso.extract_mses()
results_cross_val = Result(file_cross_val, "lasso")

##get transformed data
XTrain, YTrain, XTest, YTest = results_cross_val.extract_train_test()

### centratura dei dati
XTrain, YTrain, X_mean, y_mean, X_std = center_data(XTrain, YTrain, fit_intercept=True, normalize = True)
XTest, YTest = center_test(XTest,YTest,X_mean,y_mean,X_std)

##ranking
verbose = True

weights_data = results_cross_val.extract_weights()

index_mse = len(weights_data)-1
weights_data = weights_data[index_mse]

ordered_final_weights = np.argsort(weights_data)[::-1]

if verbose:
    print("-------------")
    print("ranking of the featues:", ordered_final_weights)
    print("-------------")

###compute LASSO
#resultsData = Result(file_data,"lasso")
n_informative = 100
values_TM = []
new_loss, beta = compute_lasso(XTrain, YTrain, XTest, YTest, score = score,values_TM = values_TM)
beta = np.abs(beta[:, 0])
beta_indexes,beta_ordered = get_beta_div_zeros(beta)
index_beta = np.where(np.abs(beta)>=0.1)[0]
index_inf = index_beta[index_beta<n_informative]
index_not_inf = index_beta[index_beta>n_informative]
print("loss", new_loss, "beta inf", len(index_inf), ", beta non inf",len(index_beta))

real_indexes = []
iter = np.argmin(mses)
print("iter", iter)
indexes = ordered_final_weights[:iter+1].astype("int64")

if weights_all:
    weights = assign_weights(weights_data.copy())
    weights = weights[indexes]
else:
    weights = assign_weights(weights_data.copy()[indexes])

###compute weighted LASSO on val
XTrainVal, YTrainVal, XVal, YVal = results_cross_val.extract_train_val()
XTrain_Valcurrent, XVal_current = get_current_data(XTrainVal, XVal, indexes)

print("----------------------------")

model = Shooting(weights)
lasso = LASSOEstimator(model)

loss, beta = compute_weightedLASSO(lasso,XTrain_Valcurrent,YTrainVal, XVal_current, YVal,scoring, score_f, verbose, values_TM)

beta = np.abs(beta)
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

##new indexes
final_indexes = indexes[beta_indexes]
print("final indexes", final_indexes)
XTrain_current, XTest_current = get_current_data(XTrain, XTest,final_indexes)

weights = assign_weights(weights_data.copy())
weights = weights[final_indexes]
###compute LASSO

new_loss, beta = compute_lasso(XTrain_current, YTrain, XTest_current, YTest, score=score,values_TM = values_TM)
beta = np.abs(beta[:, 0])
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

print("loss insieme ridotto", new_loss)
print(final_indexes[beta_indexes])


model = Shooting(weights)
lasso = LASSOEstimator(model)
loss, beta = compute_weightedLASSO(lasso,XTrain_current,YTrain, XTest_current, YTest,scoring, score_f, verbose,values_TM = values_TM)

beta = np.abs(beta)
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

print(final_indexes[beta_indexes])