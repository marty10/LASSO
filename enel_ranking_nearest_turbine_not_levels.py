from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from ExtractResult import Result
from LASSOModel import Shooting, LASSOEstimator
from Lasso_utils import compute_lasso, compute_weightedLASSO
from utility import assign_weights, get_current_data, get_beta_div_zeros, print_features_active, extract_level
import sys
from pprint import pprint

sys.argv[1:2] = [str(x) for x in sys.argv[1:2]]
file_name = sys.argv[1]
weights_all = (int)(sys.argv[2])

ext = ".npz"
file = "ENEL_2014/"+file_name+ext

results = Result(file, "lasso")

dict_ = results.extract_dict()

XTrain, YTrain, XVal, YVal = results.extract_train_val()

new_loss, beta = compute_lasso(XTrain, YTrain, XVal, YVal,score = "mean_squared_error",values_TM = [])
print("loss lineare", new_loss)

score = "mean_squared_error"
if score=="r2_score":
    score_f = r2_score
    scoring = "r2"
else:
    score_f = mean_squared_error
    scoring = "mean_squared_error"

verbose = True

###compute ranking

weights_data = results.extract_weights()

index_mse = len(weights_data) - 1
weights_data = weights_data[index_mse]
ordered_final_weights = np.argsort(weights_data)[::-1]

values = dict_.values()
weights_level = extract_level(ordered_final_weights, values)


if verbose:
    print("-------------")
    print("ranking of the features:")
    pprint(weights_level)
    print("-------------")

losses = []
ordered_indexes = ordered_final_weights


losses = []
indexes_tot = []
n_features = len(ordered_final_weights)
values_TM = np.array([])

if weights_all:
    weights_ = assign_weights(weights_data.copy())

for i in range(n_features):

        indexes = ordered_final_weights[:i+1].astype("int64")
        indexes_tot.append(indexes)

        XTrain_current, XTest_current = get_current_data(XTrain, XVal, indexes)

        print("----------------------------")
        print("iteration ", i)

        if not weights_all:
            weights_ = assign_weights(weights_data.copy()[indexes])

        model = Shooting(weights_)
        lasso = LASSOEstimator(model)

        loss, beta = compute_weightedLASSO(lasso,XTrain_current,YTrain, XTest_current, YVal,scoring, score_f, verbose, values_TM)
        losses.append(loss)

        beta = np.abs(beta)
        beta_indexes,beta_ordered = get_beta_div_zeros(beta)

        print("livelli selezionati:")
        pprint(weights_level[beta_indexes])

        np.savez(file_name+"ranking_not_levels_weights_all"+str(weights_all)+ext, mses = losses, indexes = indexes_tot)

print("min mse", np.min(losses), "with:", indexes_tot(np.argmin(losses)))