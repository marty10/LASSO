from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.metrics import r2_score, mean_squared_error
from ExtractResult import Result
import numpy as np
from LASSOModel import Shooting, LASSOEstimator
from utility import get_current_data, assign_weights, compute_mse, assign_weights_ordered, compute_lasso
import matplotlib.pyplot as plt

file = "ENEL_2014/Enel_cross_val_blocks.npz"
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
    final_weights[key] += np.sum(weights_data[dict_.get(key).astype("int64")])

ordered_final_weights = np.argsort(final_weights)[::-1]
if verbose:
    print("-------------")
    print("ranking of the featues:", ordered_final_weights)
    print("-------------")
ordered_indexes = np.argsort(weights_data)[::-1]
losses = []


new_loss, _ = compute_lasso(XTrain, YTrain, XVal, YVal, score)

print("new_loss", new_loss)

losses = []

n_features = XTrain.shape[1]
lasso_cv = linear_model.LassoCV(fit_intercept=False, max_iter=100000, n_jobs=-1)

for i in range(n_features):

        ###compute LASSO
        indexes = np.array([], dtype="int64")
        for k in ordered_final_weights[:i+1]:
            indexes = np.union1d(indexes,dict_.get(k))

        del_ = np.array([], dtype = "int64")
        for key in ordered_final_weights[:i+1]:
            for key_1 in ordered_final_weights[i+1:]:
                    del_ = np.append(del_,np.intersect1d(list(dict_.values())[key], list(dict_.values())[key_1]))

        a = np.in1d(indexes,del_)
        indexes = np.delete(indexes, np.where(a==True)[0])

        indexes = indexes.astype("int64")
        XTrain_current, XTest_current = get_current_data(XTrain, XVal, indexes)

        print("----------------------------")
        print("iteration ", i)

        new_loss, _ = compute_lasso(XTrain_current, YTrain, XTest_current, YVal, score)

        losses.append(new_loss)
        beta = np.abs(beta[:, 0])
        beta_ord = np.sort(beta)[::-1]
        beta_ordered = beta_ord[beta_ord >= 0.1]
        len_div_zero = len(beta_ordered)
        beta_indexes = np.argsort(np.abs(beta))[::-1][:len_div_zero]

        keys_sel = ordered_final_weights[:i+1]
        print("lasso",new_loss)
        for key in keys_sel:
            for j in indexes[beta_indexes]:
                if j in dict_.get(key):
                    print (key)
                    break


        weights_ = weights[indexes]

        model = Shooting(weights_)
        lasso = LASSOEstimator(model)

        alphas = _alpha_grid(
                   XTrain_current, YTrain, fit_intercept=False)
        #print ("alpha",alphas[0])
        parameters = {"alpha": alphas}

        clf = GridSearchCV(lasso, parameters, fit_params={"verbose": False}, cv=3, scoring="r2")
        clf.fit(XTrain_current, YTrain)
        lambda_opt = clf.best_params_
        print (lambda_opt)

        lasso.set_params(**lambda_opt)
        lasso.fit(XTrain_current, YTrain)


        y_pred_test = lasso.predict(XTest_current)
        mse_test = score_f(YVal, y_pred_test)
        if verbose:
            print ("mse_test " + model.__class__.__name__, mse_test)

        beta = np.abs(lasso.beta)

        beta_ord = np.sort(beta)[::-1]
        beta_ordered = beta_ord[beta_ord >= 0.1]
        len_div_zero = len(beta_ordered)
        beta_indexes = np.argsort(np.abs(beta))[::-1][:len_div_zero]


        for key in keys_sel:
            for j in indexes[beta_indexes]:
                if j in dict_.get(key):
                    print (key)
                    break




x_vect = np.arange(1,original_features+1)
n_plots = 1
fig, axarr = plt.subplots(n_plots, sharex=True)

axarr.plot(x_vect, losses, 'b-+')



plt.xlabel('features')
plt.ylabel('mse')
plt.grid(True)
plt.show()