from sklearn import linear_model
from ExtractResult import Result
from Transformation import F1,  F2, F3, F4, F5, F1_transf, F2_transf
from sklearn.grid_search import GridSearchCV
from LASSOModel import LASSOEstimator,Shooting
from sklearn.metrics import mean_squared_error
import numpy as np
from utility import get_current_data, assign_weights, compute_mse
import sys


folder = "AlgorithmResults/"


print(sys.argv[1])
sys.argv[1:] = [int(x) for x in sys.argv[1:]]
n_samples = sys.argv[1]
print("n_samples", n_samples)
original_features = sys.argv[2]
print("original_features", original_features)
transformation = F2()
print("function", transformation)
end = sys.argv[3]
print("lambda_max", end)

verbose = True
file = "nonLinearDataset/"+transformation.__class__.__name__+"/test"+transformation.__class__.__name__+"num_blocks_modified1000num_samples"+str(n_samples)+"n_features"+str(original_features)+"dynamic_set.npz"
results = Result(file, "lasso")
XTrain, YTrain, XTest, YTest,mses = results.extract_data()

#dict_ = results.extract_dict()

weights_data = results.extract_weights()
informative_indexes = results.extract_informative()
print (informative_indexes)

n_features = XTrain.shape[1]
index_mse = len(weights_data)-1
weights_data = weights_data[index_mse]

final_weights = np.zeros(original_features)

#keys_ = np.array(dict_.keys()).astype("int64")
#for key in keys_:
    #final_weights[key] += np.sum(weights_data[dict_.get(key).astype("int64")])

ordered_final_weights = np.argsort(final_weights)[::-1]
if verbose:
    print("-------------")
    print("ranking of the featues:", ordered_final_weights)
    print("-------------")
ordered_indexes = np.argsort(weights_data)[::-1]
if verbose:
    print("position informarives", np.where(np.in1d(ordered_indexes, informative_indexes)==True)[0])


lasso_cv = linear_model.LassoCV(fit_intercept=False,  max_iter=10000, n_jobs = -1)
lasso_cv.fit(XTrain,YTrain)
best_alpha = lasso_cv.alpha_

model = linear_model.Lasso(fit_intercept=False,alpha=best_alpha)
new_loss,beta,_ = compute_mse(model, XTrain, YTrain,XTest, YTest)
beta = np.abs(beta[:,0])
beta_ord = np.sort(beta)[::-1]
beta_ordered = beta_ord[beta_ord>=1.0]
len_div_zero = len(beta_ordered)
beta_indexes = np.argsort(np.abs(beta))[::-1][:len_div_zero]

real_indexes = []

# for b in beta_indexes:
#     values_ = dict_.values()
#     count=0
#     for dict_value in values_:
#         if b in dict_value:
#             real_indexes.append(count)
#         count+=1

if verbose:
    print("real features:",real_indexes )
_, idx = np.unique(real_indexes, return_index=True)

sorted_idx = np.sort(idx)
if verbose:
    print("unique real features", np.array(real_indexes)[sorted_idx])
    print("valori dei beta ordinati", beta_ordered)

    print("loss LASSO", new_loss)
    print("------------------")


active_sets = np.arange(1,n_features+1)
#file1 = np.load("AlgorithmResults/Shooting_model_vary_active_set.npz")
#mses = file1["mses"]
#min_mse = np.argmin(mses)
#print(mses[min_mse])
#active_set = active_sets[min_mse]
#active_sets= []
#active_sets.append(active_set)
mses = []
beta_div_zeros = []
indexes_to_extract = []
ordered_indexes_div_zeros = []
current_informatives = []

alphas = np.linspace(0.001, end, 100)
parameters = {"alpha": alphas}

for active_set in active_sets:

    ordered_values = np.sort(weights_data)[::-1][:active_set]
    ordered_indexes = np.argsort(weights_data)[::-1][:active_set]

    #active_indexes = extract_results(file)

    weights = assign_weights(ordered_values.copy())
    #weights[:70]= 0.0
    #ordered_values = ordered_values[:len(weights)]
    #ordered_indexes = ordered_indexes[:len(weights)]
    #weights = assign_weights_ordered(ordered_values.copy())

    #min_ = np.min(ordered_values)
    #max_ = np.max(ordered_values)

    #weights_std = (ordered_values -  min_)/ (max_ - min_)
    #weights = weights_std

    #norm = Normalizer(norm = "max")
    #weights_std = norm.fit_transform(ordered_values)[0]
    #weights = 1-weights_std

    #weights = weights[:index_mse*3]
    current_informative = np.intersect1d(ordered_indexes, informative_indexes)
    current_not_informative = np.array(list(set(ordered_indexes)-set(current_informative)))
    if verbose:
        print("informative", len(current_informative), "su", len(ordered_indexes))
        print("non informative",len(current_not_informative),"su", len(ordered_indexes))

    current_train, current_test = get_current_data(XTrain, XTest, ordered_indexes)
    indexes_to_extract.append(ordered_indexes)

    model_list = {Shooting(weights)}
    ext_data = ".npz"
    ext_model = ".pkl"


    for model in model_list:
        lasso = LASSOEstimator(model)

        clf = GridSearchCV(lasso, parameters, fit_params = {"verbose" : False}, cv=3, scoring="mean_squared_error")
        clf.fit(current_train, YTrain)
        lambda_opt = clf.best_params_
        if verbose:
            print("best lambda", lambda_opt)

        lasso.set_params(**lambda_opt)
        lasso.fit(current_train,YTrain)

        y_pred_train = lasso.predict(current_train)
        mse_train = mean_squared_error(YTrain, y_pred_train)
        if verbose:
            print("mse_train "+model.__class__.__name__,mse_train)

        y_pred_test = lasso.predict(current_test)
        mse_test = mean_squared_error(YTest, y_pred_test)
        if verbose:
            print ("active_set", active_set, "mse_test "+model.__class__.__name__,mse_test)
        mses.append(mse_test)

        beta = lasso.beta

        beta_zero = np.where(np.abs(beta)<0.1)[0]
        ordered_indexes_zeros = ordered_indexes[beta_zero]

        beta_div_zero = np.where(np.abs(beta)>=0.1)[0]
        ordered_indexes_div_zero = ordered_indexes[beta_div_zero]
        if verbose:
            print(ordered_indexes_div_zero)
        indexing = np.argsort(ordered_indexes_div_zero)
        #print beta
        beta_ordered_index = np.argsort(np.abs(beta))
        #print(ordered_indexes[beta_ordered_index])
        ordered_indexes_div_zeros.append(ordered_indexes_div_zero)

        assert(len(beta_div_zero)+len(beta_zero)==len(beta))

        inf_zero = np.intersect1d(ordered_indexes_zeros,current_informative)
        non_inf_zero = np.intersect1d(ordered_indexes_zeros,current_not_informative)
        if verbose:
            print("beta inf a zero",len(inf_zero) , "su", len(current_informative))
            print("beta non inf a zero", len(non_inf_zero), "su",len(current_not_informative))
            print("----------------------------------------")
        np.savez(folder+model.__class__.__name__+transformation.__class__.__name__+"model_features"+str(n_features)+"samples"+str(n_samples)+ext_data, informative_indexes = informative_indexes, ordered_indexes_div_zeros = ordered_indexes_div_zeros,XTrain = XTrain, XTest = XTest, YTrain = YTrain, YTest = YTest, indexes_to_extract = indexes_to_extract, beta_div_zeros = beta_div_zeros, active_sets = active_sets, mses = mses)

        #joblib.dump(lasso, folder+model.__class__.__name__+'_model_2'+ext_model, compress=9)

min_mse = np.argmin(mses)
min_active_set = active_sets[min_mse]
print("loss minima", mses[min_mse], "con active_set", min_active_set)