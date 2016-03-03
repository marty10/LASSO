from sklearn.linear_model.base import center_data
from sklearn.metrics import mean_squared_error, r2_score
from ExtractResult import Result
from LASSOModel import Shooting, LASSOEstimator
import numpy as np
from utility import center_test, assign_weights, get_current_data, compute_lasso, get_beta_div_zeros, \
    print_features_active, compute_weightedLASSO
import sys

sys.argv[1:2] = [int(x) for x in sys.argv[1:2]]
iter = sys.argv[1]
file_name = (str)(sys.argv[2])


compute_lasso_current = False
score = "mean_squared_error"
if score=="r2_score":
    score_f = r2_score
    scoring = "r2"
else:
    score_f = mean_squared_error
    scoring = "mean_squared_error"

folder = "ENEL_2014/"
ext = ".npz"

file_cross_val =  folder+file_name+ext

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

ordered_final_weights = np.argsort(weights_data)[::-1]
if verbose:
    print("-------------")
    print("ranking of the featues:", ordered_final_weights)
    print("-------------")

###compute LASSO
#resultsData = Result(file_data,"lasso")
ordered_indexes = ordered_final_weights

indexes = ordered_final_weights[:iter].astype("int64")

weights = assign_weights(weights_data.copy())
weights = weights[indexes]
XTrain_current, XTest_current = get_current_data(XTrain, XTest,indexes)


###compute LASSO
if compute_lasso_current:
    new_loss, beta = compute_lasso(XTrain_current, YTrain, XTest_current, YTest, score=score)
    beta = np.abs(beta[:, 0])
    beta_indexes,beta_ordered = get_beta_div_zeros(beta)

    print("loss insieme ridotto", new_loss)
    print(indexes[beta_indexes])


model = Shooting(weights)
lasso = LASSOEstimator(model)
loss, beta = compute_weightedLASSO(lasso,XTrain_current,YTrain, XTest_current, YTest,scoring, score_f, verbose)

beta = np.abs(beta)
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

print(indexes[beta_indexes])
