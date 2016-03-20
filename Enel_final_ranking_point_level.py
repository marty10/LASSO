from pprint import pprint
from sklearn.linear_model.base import center_data
from sklearn.metrics import mean_squared_error, r2_score

from Enel_utils import extract_new_dict
from ExtractResult import Result
from LASSOModel import Shooting, LASSOEstimator
import numpy as np
from Lasso_utils import compute_lasso, compute_weightedLASSO
from utility import center_test, assign_weights, get_current_data, get_beta_div_zeros, \
    print_features_active,extract_level, extract_point_level
import sys

sys.argv[1:] = [str(x) for x in sys.argv[1:]]
file_name = sys.argv[1]


score = "mean_squared_error"
if score=="r2_score":
    score_f = r2_score
    scoring = "r2"
else:
    score_f = mean_squared_error
    scoring = "mean_squared_error"

ext = ".npz"
file_cross_val = file_name+ext
fine_name_weights = file_name+"ranking_point_level"+ext

results_cross_val = Result(file_cross_val, "lasso")
results_weighted_lasso = Result(fine_name_weights, "lasso")

mses = results_weighted_lasso.extract_mses()
mses_int = list(map(int, mses))
min_mse = np.min(mses_int)
iter_min = np.where(mses_int==min_mse)[0]

indexes_beta_min = results_weighted_lasso.extract_beta_div_zeros()[iter_min]
print(indexes_beta_min)
print("-----------")
print(indexes_beta_min.shape)

print("--------------")
len_mylist = (list)(map(len, indexes_beta_min))

iter = iter_min[len_mylist.index(min(len_mylist))]

print ("iter chosen:",iter, "with mse:",mses_int[iter])
print("--------------")


indexes_beta = indexes_beta_min[iter]


XTrain, XTest = results_cross_val.extract_data_transf()
_,YTrain,_, YTest = results_cross_val.extract_train_test()

### centratura dei dati
XTrain, YTrain, X_mean, y_mean, X_std = center_data(XTrain, YTrain, fit_intercept=True, normalize = True)
XTest, YTest = center_test(XTest,YTest,X_mean,y_mean,X_std)


##ranking
verbose = True
###compute ranking
dict_ = results_cross_val.extract_dict()
dict_point_level = extract_new_dict(dict_)

weights_data = results_cross_val.extract_weights()

index_mse = len(weights_data) - 1
weights_data = weights_data[index_mse]
weights = assign_weights(weights_data.copy())

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
        current_values_level = current_values[level].astype("int64")
        final_weights[count] += np.sum(weights_data[current_values_level])
        matrix_point_level[count,:] = [key,level]
        count+=1

ordered_final_weights = np.argsort(final_weights)[::-1]

if verbose:
    print("-------------")
    print("ranking of the features:")
    pprint(matrix_point_level[ordered_final_weights,:])
    print("-------------")


values_TM = np.array([[24,281], [24,214]])
#new_loss, beta = compute_lasso(XTrain, YTrain, XTest, YTest, score = score,values_TM = values_TM)
#beta = np.abs(beta[:, 0])
#beta_indexes,beta_ordered = get_beta_div_zeros(beta)

#real_indexes = []

#if verbose:
 #   print("loss LASSO test", new_loss)
  #  print("------------------")


###recompute weights
weights = weights[indexes_beta]

XTrain_current, XTest_current = get_current_data(XTrain, XTest,indexes_beta)
###compute LASSO
print("-------------")
new_loss, beta = compute_lasso(XTrain_current, YTrain, XTest_current, YTest, score=score,values_TM = values_TM)
beta = np.abs(beta[:, 0])
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

print("loss insieme ridotto", new_loss)
pprint(extract_point_level(indexes_beta[beta_indexes], dict_))
print("------------")


print("----------------")
model = Shooting(weights)
lasso = LASSOEstimator(model)

loss, beta = compute_weightedLASSO(lasso,XTrain_current,YTrain, XTest_current, YTest,scoring, score_f, verbose,values_TM = values_TM)

beta = np.abs(beta)
beta_indexes,beta_ordered = get_beta_div_zeros(beta)

print("weighted lasso losss", loss)
pprint(extract_point_level(indexes_beta[beta_indexes], dict_))

print("----------------")