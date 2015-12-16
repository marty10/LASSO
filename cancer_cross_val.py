from sklearn import linear_model
from sklearn.linear_model.base import center_data
import sys
from ExtractDataset import Dataset
import numpy as np
from utility import generate_samples, get_current_data, compute_mse, extract_losses, get_common_indexes, \
    extract_chosen_indexes, extract_chosen_indexes_from_start



#####dataset

def center_test(X, y, X_mean, y_mean, X_std, normalize = True):
    X -= X_mean
    if normalize:
        X /= X_std
        y = y - y_mean
    return X,y

print(sys.argv[1])
sys.argv[1:] = [int(x) for x in sys.argv[1:]]
num_blocks = sys.argv[1]
print("num_blockd", num_blocks)
active_set = sys.argv[2]
print("active_set", active_set)

file = np.load("cancer_dataset.npz")
x = file["x"]
y = file["y"]
n_samples = len(x)
n_features = len([val for (key,val) in x[0].iteritems()])
X = np.zeros([n_samples,n_features])


for i in range(0,n_samples):
    X[i,:] = [val for (key,val) in x[i].iteritems()]

XTrain = X[:33,:]
XTest =  X[33:,:]

YTrain = y[:33]
YTest = y[33:]

XTrain, YTrain, X_mean, y_mean, X_std = center_data(XTrain, YTrain, fit_intercept=True, normalize = True)
XTest, YTest = center_test(XTest,YTest,X_mean,y_mean,X_std)

####generation blocks

r = np.random.RandomState(11)


saved_indexes = np.array([],dtype = "int64")
chosen_indexes = 3
num_informative = np.array([])
coeffs = np.array([])
saved_indexes_list = []
mses = []
num_informative_list = []
###filter
# file = np.load("DistanceCorrelation500.npz")
# d_cor_xy = file["d_cor_xy"]
# d_cor_xx = file["d_cor_xx"]
#
# d_cor_xy_ordered = np.argsort(d_cor_xy)
# XTrain = np.delete(XTrain, d_cor_xy_ordered[:24],axis = 1)
# XTest = np.delete(XTest, d_cor_xy_ordered[:24],axis = 1)
linearRegression = linear_model.LinearRegression(fit_intercept=False, n_jobs = -1)
mse,_,_ = compute_mse(linearRegression, XTrain, YTrain,XTest,YTest)
print("start_mse", mse)
weights_indexes = np.zeros(n_features)
while len(saved_indexes)<active_set:
    losses = np.array([])
    betas = np.array([])
    coeffs = np.array([])
    corrs = np.array([])
    blocks_generated = generate_samples(num_blocks, XTrain.shape[1], active_set, r, saved_indexes, np.array([]))
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
    ordered_loss_ten = ordered_losses[:10]

    weights_indexes,beta_sign = get_common_indexes(weights_indexes, ordered_loss_ten,blocks_generated,betas,n_features )
    ordered_weights_indexes = np.argsort(weights_indexes)[::-1]
    ordered_weights_indexes_values = np.sort(weights_indexes)[::-1]

    saved_indexes = extract_chosen_indexes_from_start(saved_indexes, ordered_weights_indexes, ordered_weights_indexes_values, chosen_indexes)
    x_train_saved, x_test_saved = get_current_data(XTrain, XTest, saved_indexes)
    mse_saved,_,_ = compute_mse(linearRegression, x_train_saved, YTrain,x_test_saved, YTest)
    mses.append(mse_saved)

    num_informative = [f for f in saved_indexes if f<=99]
    print("num_inf", len(num_informative), "su", len(saved_indexes), "mse", mse_saved)

    saved_indexes_list.append(saved_indexes)
    num_informative_list.append(num_informative)

np.savez("cross_val_blocks_"+str(num_blocks)+"active_set"+ str(active_set)+"cancer.npz", saved_indexes_list = saved_indexes_list, mses = mses, num_informative_list = num_informative_list)