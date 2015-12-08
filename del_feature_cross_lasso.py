from sklearn import linear_model
from utility import generate_samples, get_current_data, extract_losses, get_common_indexes, \
    extracte_chosen_indexes, compute_mse, get_low_common_indexes, generate_samples_del
from ExtractDataset import Dataset
import numpy as np

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


####generation blocks
num_blocks = 10000
r = np.random.RandomState(11)
active_set = 100

saved_indexes = np.array([])
chosen_indexes = 3
num_informative = np.array([])

r1 = np.random.RandomState(0)
rand_vect = generate_samples(1, n_features, active_set, r1, saved_indexes)[0]
# print(rand_vect)
x_train_i, x_test_i = get_current_data(XTrain, XTest, rand_vect)
lasso_cv = linear_model.LassoCV(fit_intercept=False,  max_iter=10000, n_jobs = -1)
lasso_cv.fit(x_train_i,YTrain)
best_alpha = lasso_cv.alpha_
# print("best_alpha", best_alpha)

lasso = linear_model.Lasso(alpha = best_alpha, fit_intercept=False)

while len(num_informative)<n_informative and len(saved_indexes)<active_set:
    losses = np.array([])
    betas = np.array([])
    corrs = np.array([])
    blocks_generated = generate_samples_del(num_blocks, XTrain.shape[1], active_set, r, saved_indexes)
    for i in range(0, num_blocks):
        x_train_i, x_test_i = get_current_data(XTrain, XTest, blocks_generated[i,:])
        new_loss,beta,corr= compute_mse(lasso, x_train_i, YTrain,x_test_i, YTest)
        losses = np.append(losses, new_loss)
        beta = beta.reshape([len(beta),1])

        if len(betas)==0:
            betas = beta
        else:
            betas = np.append(betas, beta, axis =1)
    ordered_losses = np.argsort(losses)

    orderd_losses_ = np.sort(losses)
    ordered_loss_ten = extract_losses(ordered_losses)

    weights_indexes = get_low_common_indexes(ordered_loss_ten,blocks_generated,n_features,betas)
    ordered_weights_indexes = np.argsort(weights_indexes)[::-1]
    ordered_weights_indexes_values = np.sort(weights_indexes)[::-1]

    saved_indexes = extracte_chosen_indexes(saved_indexes, ordered_weights_indexes,ordered_weights_indexes_values, chosen_indexes)
    if len(saved_indexes)>=0:#active_set/2:
        rand_vect = generate_samples_del(1, n_features, active_set, r1, saved_indexes)[0]
        x_train_i, x_test_i = get_current_data(XTrain, XTest, rand_vect)
        lasso_cv.fit(x_train_i,YTrain)
        best_alpha = lasso_cv.alpha_
        lasso = linear_model.Lasso(alpha = best_alpha, fit_intercept=False)
        #model = lasso

    num_informative = [f for f in saved_indexes if f<=99]
    print("num_inf", len(num_informative), "su", len(saved_indexes))

np.savez("cross_val_blocks_"+str(num_blocks)+"active_set"+str(active_set)+"lasso.npz", saved_indexes = saved_indexes)