from sklearn.linear_model.base import center_data
from sklearn.metrics import mean_squared_error, r2_score
from ExtractResult import Result
from LASSOModel import Shooting, LASSOEstimator
import numpy as np
from utility import compute_mse, center_test, assign_weights, get_current_data, compute_lasso, get_beta_div_zeros, \
    print_features_active, compute_weightedLASSO, extract_level

iter = 170
score = "mean_squared_error"
if score=="r2_score":
    score_f = r2_score
    scoring = "r2"
else:
    score_f = mean_squared_error
    scoring = "mean_squared_error"

folder = "ENEL_2014/"

file_cross_val =  folder+"Enel_cross_val_blocks.npz"

results_cross_val = Result(file_cross_val, "lasso")

##get transformed data
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

keys_ = np.array(dict_.keys()).astype("int64")
ordered_final_weights = np.argsort(weights_data)[::-1]

weights_level = extract_level(ordered_final_weights, values)

if verbose:
    print("-------------")
    print("ranking of the featues:", ordered_final_weights)
    print("-------------")

###compute LASSO
#resultsData = Result(file_data,"lasso")
dict_ = results_cross_val.extract_dict()

new_loss, beta = compute_lasso(XTrain, YTrain, XTest, YTest, score = score)
beta = np.abs(beta[:, 0])
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

real_indexes = []

ordered_indexes = np.argsort(weights_data)[::-1]

print_features_active(keys_, beta_indexes, dict_)

if verbose:

    print("loss LASSO test", new_loss)
    print("------------------")

indexes = ordered_final_weights[:iter].astype("int64")

weights = assign_weights(weights_data.copy())
weights = weights[indexes]
XTrain_current, XTest_current = get_current_data(XTrain, XTest,indexes)


###compute LASSO
new_loss, beta = compute_lasso(XTrain_current, YTrain, XTest_current, YTest, score=score)
beta = np.abs(beta[:, 0])
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

print("loss insieme ridotto", new_loss)
print(indexes[beta_indexes])

print(weights_level[beta_indexes])

model = Shooting(weights)
lasso = LASSOEstimator(model)
loss, beta = compute_weightedLASSO(lasso,XTrain_current,YTrain, XTest_current, YTest,scoring, score_f, verbose)

beta = np.abs(beta)
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

print(indexes[beta_indexes])
print(weights_level[beta_indexes])