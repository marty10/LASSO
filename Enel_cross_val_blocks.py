from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.base import center_data
import numpy as np
from ExtractResult import Result
from Transformation import EnelTransformation
from utility import generate_samples_dynamic_set, get_current_data, compute_mse, get_common_indexes, \
    extract_chosen_indexes_from_start, center_test
import sys
from Transformation import EnelWindSpeedTransformation

####load data
file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")

XTrain, YTrain, XTest, YTest = results.extract_train_test()

sys.argv[1:] = [str(x) for x in sys.argv[1:]]
kind_transformation = sys.argv[1]

##transformation of data

if kind_transformation=="windSpeed":
    transf = EnelWindSpeedTransformation()
    YTrain = YTrain**(1./3)
    YTest = YTest**(1./3)
    output_file_name = "Enel_cross_val_blocks_wind_speed.npz"
elif kind_transformation=="windSpeedPower3":
    transf = EnelTransformation()
    output_file_name = "Enel_cross_val_blocks.npz"

XTrain_transf, dict_ = transf.transform(XTrain)
XTest_transf, dict_ = transf.transform(XTest)

##center data
##center data
XTrain_noCenter, XVal_noCenter, YTrain_noCenter, YVal_noCenter = train_test_split(XTrain_transf, YTrain, test_size=0.33,random_state=0)
XTrain_, YTrain_, X_mean, y_mean, X_std = center_data(XTrain_noCenter, YTrain_noCenter, fit_intercept=True, normalize = True)
XVal_, YVal_ = center_test(XVal_noCenter,YVal_noCenter,X_mean,y_mean,X_std)

#new_loss, _ = compute_lasso(XTrain_, YTrain_, XVal_, YVal_,score = "mean_squared_error")
#print("loss", new_loss)

n_features_transf = XTrain_.shape[1]

####generation blocks
num_blocks = 1000

r = np.random.RandomState(11)
r1 = np.random.RandomState(12)
r2 = np.random.RandomState(13)
r4 = np.random.RandomState(15)

n_samples_val = XVal_.shape[0]

active_set_samples = (int)(8./9.*n_samples_val)

saved_indexes = []

saved_indexes_list = []
mses = []
weights_list = []


weights_indexes = np.zeros(n_features_transf)
r3 = np.random.RandomState(14)
countIter = 4

num_cycle = 0
cycles = 10
min_set = 5
max_set = min_set+5
max_active_set = int(n_features_transf/10)
active_set = 0

compute_mse_current = "False"
lasso_cv = linear_model.LassoCV(fit_intercept=False, n_jobs = -1)
flag_linear = 0
score = "mean_squared_error"

while num_cycle<cycles:

    losses = []
    betas = []

    if len(saved_indexes)>=max_active_set:
        num_cycle +=1
        print ("ciclo", num_cycle)
        saved_indexes = []
        active_set = 0
        countIter=4
    if len(saved_indexes)>0:
        if countIter==5:
            lasso_cv.fit(x_train_saved,YTrain_)
            best_alpha = lasso_cv.alpha_
            print(best_alpha)
            countIter = 0
        model = linear_model.Lasso(fit_intercept=False,alpha=best_alpha)
        flag_linear = 0
    else:
        model = linear_model.LinearRegression(fit_intercept=False)
        flag_linear = 1
    blocks_generated,active_set,num_blocks = generate_samples_dynamic_set(num_blocks, n_features_transf, r,saved_indexes,r1, min_set, max_set,active_set,max_active_set)

    for i in range(0, num_blocks):
        x_train_i, x_val_i = get_current_data(XTrain_, XVal_, blocks_generated[i,:])
        rand_vect = r1.choice(n_samples_val,active_set_samples, replace = False)
        x_val_i = x_val_i[rand_vect,:]
        YVal_i = YVal_[rand_vect]
        new_loss, beta= compute_mse(model, x_train_i, YTrain_,x_val_i, YVal_i, score)
        losses.append(new_loss)
        betas.append(beta)

    ordered_losses = np.argsort(losses)
    orderd_losses_ = np.array(losses)[ordered_losses]

    standard_deviation = np.std(orderd_losses_)
    mean_weights = np.mean(orderd_losses_)

    chosen_losses = len(orderd_losses_[orderd_losses_+standard_deviation<=mean_weights])
    if chosen_losses>num_blocks/3:
        chosen_losses = num_blocks/3

    print("losses scelte", chosen_losses)
    index_chosen_losses = ordered_losses[:chosen_losses]
    betas = np.hstack(betas)
    if flag_linear == 0:
        weights_indexes = get_common_indexes(weights_indexes, index_chosen_losses,blocks_generated,betas,n_features_transf)
        weights_abs = np.abs(weights_indexes)
    else:
        weights_indexes_copy = get_common_indexes(weights_indexes.copy(), index_chosen_losses,blocks_generated,betas,n_features_transf)
        weights_abs = np.abs(weights_indexes_copy)

    ordered_weights_indexes = np.argsort(weights_abs)[::-1]
    ordered_weights_indexes_values = weights_abs[ordered_weights_indexes]

    chosen_indexes = r3.choice(np.arange(2,4), 1, replace=False)[0]
    if chosen_indexes+len(saved_indexes)>max_active_set:
        chosen_indexes = max_active_set-len(saved_indexes)
    print("chosen indexes", chosen_indexes)
    saved_indexes = extract_chosen_indexes_from_start(saved_indexes, ordered_weights_indexes, chosen_indexes)

    assert(len(saved_indexes)<=max_active_set)
    x_train_saved, x_val_saved = get_current_data(XTrain_, XVal_, saved_indexes)
    end_time = timeit.default_timer()

    if compute_mse_current:
        x_val_saved = x_val_saved[rand_vect,:]
        YVal_saved = YVal_[rand_vect]
        mse_saved,_= compute_mse(model, x_train_saved, YTrain_,x_val_saved, YVal_saved, "mean_squared_error")
        mses.append(mse_saved)
        print("mse", mse_saved)
    weights_list.append(weights_abs)

    print("saved_indexes", saved_indexes)

    print("---------------")
    saved_indexes_list.append(saved_indexes)

    countIter+=1

    np.savez(output_file_name, dict_ = dict_,saved_indexes_list = saved_indexes_list, mses = mses, weights_list = weights_list, XTrain = XTrain, XTest = XTest, YTest = YTest, YTrain = YTrain, XTrainTransf_ = XTrain_transf, XTestTransf_ = XTest_transf, XTrain_ValNoCenter = XTrain_noCenter,
           XValTransf_noCenter = XVal_noCenter, YTrainVal_noCenter = YTrain_noCenter, YVal_noCenter = YVal_noCenter, XTrain_Val = XTrain_, XVal = XVal_ , YVal_ = YVal_, YTrain_Val = YTrain_ )