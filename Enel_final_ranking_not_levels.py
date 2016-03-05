from sklearn.linear_model.base import center_data
from sklearn.metrics import mean_squared_error, r2_score
from ExtractResult import Result
from LASSOModel import Shooting, LASSOEstimator
import numpy as np
from Lasso_utils import compute_lasso, compute_weightedLASSO
from utility import center_test, assign_weights, get_current_data, get_beta_div_zeros, \
    print_features_active,extract_level
import sys

sys.argv[1:2] = [int(x) for x in sys.argv[1:2]]
weights_all = sys.argv[1]
file_name = str(sys.argv[2])


score = "mean_squared_error"
if score=="r2_score":
    score_f = r2_score
    scoring = "r2"
else:
    score_f = mean_squared_error
    scoring = "mean_squared_error"

folder = "ENEL_2014/"
ext = ".npz"
file_cross_val = folder+file_name+ext
fine_name_weights = file_name+"ranking_not_levels"+ext

results_cross_val = Result(file_cross_val, "lasso")
results_weighted_lasso = Result(fine_name_weights, "lasso")

iter = np.argmin(results_weighted_lasso.extract_mses())
indexes = results_weighted_lasso.extract_indexes_tot()[iter]


XTrain_val, YTrain_val, XVal, YVal = results_cross_val.extract_train_val()
XTrain, XTest = results_cross_val.extract_data_transf()
_,YTrain,_, YTest = results_cross_val.extract_train_test()

### centratura dei dati
XTrain, YTrain, X_mean, y_mean, X_std = center_data(XTrain, YTrain, fit_intercept=True, normalize = True)
XTest, YTest = center_test(XTest,YTest,X_mean,y_mean,X_std)


##ranking
verbose = True
dict_ = results_cross_val.extract_dict()

weights_data = results_cross_val.extract_weights()

index_mse = len(weights_data)-1
weights_data = weights_data[index_mse]
values = list(dict_.values())

keys_ = np.array((list)(dict_.keys())).astype("int64")
ordered_final_weights = np.argsort(weights_data)[::-1]

weights_level = extract_level(ordered_final_weights, values)

if verbose:
    print("-------------")
    print("ranking of the featues:", ordered_final_weights)
    print("-------------")

###compute LASSO
#resultsData = Result(file_data,"lasso")
dict_ = results_cross_val.extract_dict()

values_TM = np.array([[24,281], [24,214]])
new_loss, beta = compute_lasso(XTrain, YTrain, XTest, YTest, score = score,values_TM = values_TM)
beta = np.abs(beta[:, 0])
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

real_indexes = []

if verbose:
    print("loss LASSO test", new_loss)
    print("------------------")


if weights_all:
    weights = assign_weights(weights_data.copy())
    weights = weights[indexes]
else:
    weights = assign_weights(weights_data.copy()[indexes])


##ricompute weighted lasso on val
XTrain_current_val, XVal_current = get_current_data(XTrain_val, XVal, indexes)

model = Shooting(weights)
lasso = LASSOEstimator(model)

loss, beta = compute_weightedLASSO(lasso,XTrain_current_val,YTrain_val, XVal_current, YVal,scoring, score_f, verbose, values_TM = [])

beta = np.abs(beta)
beta_indexes_,beta_ordered = get_beta_div_zeros(beta)

XTrain_current, XTest_current = get_current_data(XTrain, XTest,indexes[beta_indexes_])

###recompute weights
if weights_all:
    weights = assign_weights(weights_data.copy())
    weights = weights[indexes]
else:
    weights = assign_weights(weights_data.copy()[indexes[beta_indexes_]])

###compute LASSO
new_loss, beta = compute_lasso(XTrain_current, YTrain, XTest_current, YTest, score=score,values_TM = values_TM)
beta = np.abs(beta[:, 0])
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

print("loss insieme ridotto", new_loss)
print(indexes[beta_indexes])

print(weights_level[beta_indexes])


model = Shooting(weights)
lasso = LASSOEstimator(model)
print(XTrain_current.shape[1])
print(XTest_current.shape[1])
print(len(weights))
loss, beta = compute_weightedLASSO(lasso,XTrain_current,YTrain, XTest_current, YTest,scoring, score_f, verbose,values_TM = values_TM)

beta = np.abs(beta)
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

print(indexes[beta_indexes])
print(weights_level[beta_indexes])