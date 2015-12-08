from sklearn import linear_model
from ExtractDataset import Dataset
import numpy as np
from utility import get_current_data, compute_mse, extract_losses, get_low_common_indexes, extracte_chosen_indexes, generate_samples_del

__author__ = 'Martina'

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

saved_indexes = np.array([],dtype = "int64")
chosen_indexes = 3
num_informative = np.array([])
coeffs = np.array([])

beta_sign = np.empty(active_set)

linearRegression = linear_model.LinearRegression(fit_intercept=False, n_jobs = -1)

while len(num_informative)<n_informative and len(saved_indexes)<active_set:
    losses = np.array([])
    betas = np.array([])
    coeffs = np.array([])
    corrs = np.array([])
    blocks_generated = generate_samples_del(num_blocks, XTrain.shape[1], active_set, r, saved_indexes)
    for i in range(0, num_blocks):
        x_train_i, x_test_i = get_current_data(XTrain, XTest, blocks_generated[i,:])
        new_loss,beta,corr = compute_mse(linearRegression, x_train_i, YTrain,x_test_i, YTest)
        losses = np.append(losses, new_loss)

        if len(betas)==0:
            betas = beta
            corrs = corr
        else:
            betas = np.append(betas, beta, axis =1)
            corrs = np.append(corrs, corr,axis = 1)
    ordered_losses = np.argsort(losses)
    orderd_losses_ = np.sort(losses)
    ordered_loss_ten = extract_losses(ordered_losses)

    weights_indexes = get_low_common_indexes(ordered_loss_ten,blocks_generated,n_features,betas)
    ordered_weights_indexes = np.argsort(weights_indexes)[::-1]
    ordered_weights_indexes_values = np.sort(weights_indexes)[::-1]

    saved_indexes = extracte_chosen_indexes(saved_indexes, ordered_weights_indexes, ordered_weights_indexes_values, chosen_indexes)
    num_informative = [f for f in saved_indexes if f<=99]
    print("num_inf", len(num_informative), "su", len(saved_indexes))


np.savez("del_val_blocks_"+str(num_blocks)+"active_set",str(active_set),".npz", saved_indexes = saved_indexes)