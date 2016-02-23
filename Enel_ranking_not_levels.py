from sklearn.metrics import r2_score, mean_squared_error
from ExtractResult import Result
import numpy as np
from LASSOModel import Shooting, LASSOEstimator
from utility import get_current_data, assign_weights, compute_mse, assign_weights_ordered, compute_lasso, \
    get_beta_div_zeros, print_features_active, compute_weightedLASSO

file_name = "Enel_cross_val_blocks"
ext = ".npz"
file = "ENEL_2014/"+file_name+ext
results = Result(file, "lasso")

dict_ = results.extract_dict()

XTrain, YTrain, XVal, YVal = results.extract_train_val()

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
weights = assign_weights(weights_data.copy())

keys_ = np.array(list(dict_.keys())).astype("int64")


ordered_final_weights = np.argsort(weights_data)[::-1]
values = list(dict_.values())

weights_livel = []
for w in ordered_final_weights:
    key = np.where(values==w)[0][0]
    a = values[key]
    level = np.where(values[key]==w)[0][0]
    weights_livel.append([key, level])

if verbose:
    print("-------------")
    print("ranking of the featues:", weights_livel)
    print("-------------")
ordered_indexes = np.argsort(weights_data)[::-1]
losses = []

new_loss, _ = compute_lasso(XTrain, YTrain, XVal, YVal, score)
print("new_loss", new_loss)

losses = []
indexes_tot = []
n_features = XTrain.shape[1]

weights_livel = np.array(weights_livel)
for i in range(n_features):

        ###compute LASSO
        indexes = ordered_final_weights[:i+1].astype("int64")
        indexes_tot.append(indexes)

        XTrain_current, XTest_current = get_current_data(XTrain, XVal, indexes)

        print("----------------------------")
        print("iteration ", i)


        keys_sel = ordered_final_weights[:i+1]

        weights_ = weights[indexes]

        model = Shooting(weights_)
        lasso = LASSOEstimator(model)

        loss, beta = compute_weightedLASSO(lasso,XTrain_current,YTrain, XTest_current, YVal,scoring, score_f, verbose)
        losses.append(loss)

        beta = np.abs(beta)
        beta_indexes,beta_ordered = get_beta_div_zeros(beta)

        print(indexes[beta_indexes])
        print(weights_livel[beta_indexes])

        np.savez(file_name+"_ranking_not_levels"+ext, mses = losses, indexes = indexes_tot)

print("min mse", np.min(losses), "with:", indexes_tot[np.argmin(losses)])

