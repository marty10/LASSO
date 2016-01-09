import sys
from sklearn import linear_model
from ExtractDataset import Dataset
import numpy as np
from utility import get_current_data, compute_mse, get_common_indexes,extract_chosen_indexes_from_start, \
    generate_samples_dynamic_set

####input data
n_samples = 200
n_features = 500
n_informative = 100

#####dataset
dataset = Dataset(n_samples,n_features, n_informative = n_informative)
XTrain = dataset.XTrain
YTrain = dataset.YTrain
XTest = dataset.XTest
YTest = dataset.YTest
n_samples_test = XTest.shape[0]

####generation blocks
print(sys.argv[1])
sys.argv[1:] = [int(x) for x in sys.argv[1:]]
num_blocks = sys.argv[1]
print("num_blockd", num_blocks)
active_set = sys.argv[2]
print("active_set", active_set)
#active_set = sys.argv[1]
r = np.random.RandomState(11)
active_set_samples = (int)(8./9.*n_samples_test)

final_active_set = 500
saved_indexes = np.array([],dtype = "int64")
num_informative = np.array([])
coeffs = np.array([])
saved_indexes_list = []
mses = []
num_informative_list = []
weights_list = []
del_indexes = np.zeros(n_features)
lasso_cv = linear_model.LassoCV(fit_intercept=False,  max_iter=10000, n_jobs = -1, n_alphas=10)
lasso_cv.fit(XTrain,YTrain)
best_alpha = lasso_cv.alpha_
print(best_alpha)
max_alpha = int(best_alpha)

weights_indexes = np.zeros(n_features)
index = 1000
deleted_indexes = np.where(del_indexes>index)[0]
iter = 0
r3 = np.random.RandomState(14)
cv_flag = True
countIter = 0
while len(saved_indexes)<=final_active_set-len(deleted_indexes):
    losses = np.array([])
    betas = np.array([])
    coeffs = np.array([])
    corrs = np.array([])

    if len(saved_indexes)+40>=150:
            saved_indexes = np.array([],dtype = "int64")
    if len(saved_indexes)>0:
        if iter==1:
            weights_indexes=np.zeros(n_features)
        if countIter==3:
            cv_flag=True
            countIter=0
        if cv_flag:
            lasso_cv.fit(x_train_saved,YTrain)
            best_alpha = lasso_cv.alpha_
            print(best_alpha)
        else:
            countIter+=1
        cv_flag=False
        model = linear_model.Lasso(fit_intercept=False,alpha=best_alpha)
    else:
        model = linear_model.LinearRegression(fit_intercept=False)
    blocks_generated = generate_samples_dynamic_set(num_blocks, n_features, r,saved_indexes,r1, deleted_indexes)

    for i in range(0, num_blocks):
        x_train_i, x_test_i = get_current_data(XTrain, XTest, blocks_generated[i,:])
        rand_vect = r1.choice(n_samples_test,active_set_samples, replace = False)
        x_test_i = x_test_i[rand_vect,:]
        YTest_i = YTest[rand_vect]
        new_loss,beta,_ = compute_mse(model, x_train_i, YTrain,x_test_i, YTest_i)
        losses = np.append(losses, new_loss)

        if len(betas)==0:
            betas = beta
        else:
            betas = np.append(betas, beta, axis =1)
    ordered_losses = np.argsort(losses)
    orderd_losses_ = np.sort(losses)
    #losses_to_select = r3.choice(np.arange(100,200), 1, replace=False)[0]
    #print("loss scelte", losses_to_select)
    first_loss = orderd_losses_[0]
    chosen_losses = len(orderd_losses_[orderd_losses_<=first_loss*7./6])
    if chosen_losses<10:
        chosen_losses=10
    print("losses scelte", chosen_losses)
    index_chosen_losses = ordered_losses[:chosen_losses]

    weights_indexes,beta_sign = get_common_indexes(weights_indexes, index_chosen_losses,blocks_generated,betas,n_features,deleted_indexes,saved_indexes)
    ordered_weights_indexes = np.argsort(weights_indexes)[::-1]
    ordered_weights_indexes_values = np.sort(weights_indexes)[::-1]

    chosen_indexes = r3.choice(np.arange(2,5), 1, replace=False)[0]
    if chosen_indexes+len(saved_indexes)>final_active_set:
        chosen_indexes = final_active_set-len(saved_indexes)
    print("chosen indexes", chosen_indexes)
    saved_indexes,del_indexes = extract_chosen_indexes_from_start(saved_indexes, ordered_weights_indexes, chosen_indexes,del_indexes)
    deleted_indexes = np.where(del_indexes>index)[0]
    assert(len(deleted_indexes)+len(saved_indexes)<=n_features)
    x_train_saved, x_test_saved = get_current_data(XTrain, XTest, saved_indexes)
    x_test_saved = x_test_saved[rand_vect,:]
    YTest_saved = YTest[rand_vect]
    mse_saved,_,_ = compute_mse(model, x_train_saved, YTrain,x_test_saved, YTest_saved)
    mses.append(mse_saved)
    weights_list.append(weights_indexes)

    num_informative = saved_indexes[saved_indexes<=99]
    print("num_inf", len(num_informative), "su", len(saved_indexes), "mse", mse_saved)

    saved_indexes_list.append(saved_indexes)
    num_informative_list.append(num_informative)

    iter+=1
    np.savez("cross_val_blocks_"+str(num_blocks)+"active_set"+ str(final_active_set)+"dynamic_set_lasso_reduced.npz", saved_indexes_list = saved_indexes_list, mses = mses, num_informative_list = num_informative_list, weights_list = weights_list)