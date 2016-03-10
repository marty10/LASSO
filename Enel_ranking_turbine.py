from Lasso_utils import compute_lasso, compute_weightedLASSO
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from ExtractResult import Result
from LASSOModel import Shooting, LASSOEstimator
from utility import assign_weights, get_current_data, get_beta_div_zeros, print_features_active
import sys
from pprint import pprint


sys.argv[1:] = [str(x) for x in sys.argv[1:]]
file_name = sys.argv[1]

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
original_features = len(keys_)
final_weights = np.zeros(original_features)

for key in keys_:
    final_weights[key] += np.sum(weights_data[dict_.get(key)[:,0].astype("int64")])

ordered_final_weights = np.argsort(final_weights)[::-1]
if verbose:
    print("-------------")
    print("ranking of the features:")
    pprint(ordered_final_weights)
    print("-------------")
ordered_indexes = np.argsort(weights_data)[::-1]
losses = []


new_loss, _ = compute_lasso(XTrain, YTrain, XVal, YVal, score,values_TM = [])
print("new_loss", new_loss)

losses = []
indexes_tot = []
beta_div_zeros = []
n_features = len(ordered_final_weights)
print(n_features)

for i in range(n_features):

        ###compute LASSO
        indexes = []
        print(ordered_final_weights[:i+1])
        for k in ordered_final_weights[:i+1]:
            current_value = dict_.get(k)[:,0]
            indexes = np.union1d(indexes,current_value)

        del_ = np.array([], dtype = "int64")
        for key in ordered_final_weights[:i+1]:
            value_key = dict_.get(key)[:,0]
            for key_1 in ordered_final_weights[i+1:]:
                value_key1 = dict_.get(key_1)[:,0]
                del_ = np.append(del_,np.intersect1d(value_key,value_key1))

        a = np.in1d(indexes,del_)

        indexes = np.delete(indexes, np.where(a==True)[0])

        indexes = indexes.astype("int64")
        XTrain_current, XTest_current = get_current_data(XTrain, XVal, indexes)

        print("----------------------------")
        print("iteration ", i)

        keys_sel = ordered_final_weights[:i+1]

        weights_ = weights[indexes]

        model = Shooting(weights_)
        lasso = LASSOEstimator(model)

        loss, beta = compute_weightedLASSO(lasso,XTrain_current,YTrain, XTest_current, YVal,scoring, score_f, verbose,values_TM = [])
        losses.append(loss)

        beta = np.abs(beta)
        beta_indexes,beta_ordered = get_beta_div_zeros(beta)
        beta_div_zeros.append(indexes[beta_indexes])

        print("livelli selezionati")
        pprint(indexes[beta_indexes])
        print_features_active(keys_sel, indexes[beta_indexes], dict_)

        np.savez(file_name+"ranking"+ext, mses = losses, indexes = indexes_tot, beta_div_zeros = beta_div_zeros)

print("min mse", np.min(losses), "with:", indexes_tot(np.argmin(losses)))

