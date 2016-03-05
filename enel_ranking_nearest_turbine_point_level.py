from Enel_utils import extract_new_dict
from Lasso_utils import compute_lasso, compute_weightedLASSO
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from ExtractResult import Result
from LASSOModel import Shooting, LASSOEstimator
from utility import assign_weights, get_current_data, get_beta_div_zeros, extract_point_level
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
dict_point_level = extract_new_dict(dict_)
weights_data = results.extract_weights()

index_mse = len(weights_data) - 1
weights_data = weights_data[index_mse]
weights = assign_weights(weights_data.copy())

#weights_data = np.arange(0,XTrain.shape[1])

keys_ = np.array(list(dict_point_level.keys())).astype("int64")

## si prende la prima key perchè una vale l altra
keys_level = np.array(list(dict_point_level[keys_[0]].keys())).astype("int64")
original_features = len(keys_)*len(keys_level)
final_weights = np.zeros(original_features)

count = 0
matrix_point_level = np.zeros([original_features,2])
for key in keys_:
    current_values = dict_point_level.get(key)
    for level in keys_level:
        current_values_level = current_values[level]
        final_weights[count] += np.sum(weights_data[current_values_level])
        matrix_point_level[count,:] = [key,level]
        count+=1

ordered_final_weights = np.argsort(final_weights)[::-1]

if verbose:
    print("-------------")
    print("ranking of the features:")
    pprint(matrix_point_level[ordered_final_weights,:])
    print("-------------")

losses = []


new_loss, _ = compute_lasso(XTrain, YTrain, XVal, YVal, score,values_TM = [])
print("new_loss", new_loss)

losses = []
indexes_tot = []
n_features = len(ordered_final_weights)


for i in range(n_features):

        ###compute LASSO
        indexes = []
        for k in ordered_final_weights[:i+1]:
            current_key = matrix_point_level[k,0]
            current_level = matrix_point_level[k,1]
            current_value = dict_point_level[current_key][current_level]
            indexes = np.union1d(indexes,current_value)

        del_ = np.array([], dtype = "int64")
        for key in ordered_final_weights[:i+1]:
            current_key = matrix_point_level[key,0]
            current_level = matrix_point_level[key,1]
            value_key = dict_point_level[current_key][current_level]
            for key_1 in ordered_final_weights[i+1:]:
                current_key_1 = matrix_point_level[key_1,0]
                current_level_1 = matrix_point_level[key_1,1]
                value_key1 = dict_point_level[current_key_1][current_level_1]
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

        print("livelli selezionati")
        pprint (extract_point_level(indexes[beta_indexes], dict_))


        np.savez(file_name+"ranking_point_level"+ext, mses = losses, indexes = indexes_tot)

print("min mse", np.min(losses), "with:", indexes_tot(np.argmin(losses)))

