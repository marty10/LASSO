from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error
from ExtractDataset import Dataset
import numpy as np
import matplotlib.pyplot as plt

def update_removed_indexes(indexes, updated_indexes):
    tmp = np.zeros(len(indexes))
    indexes = np.sort(indexes)
    count = 0
    for i in indexes:
        less_i = len(updated_indexes)
        if less_i==0:
            tmp[count] = i
        else:
            tmp[count] = i+less_i
        count+=1
    updated_indexes = np.append(updated_indexes,tmp)
    return updated_indexes

def generate_samples(num_blocks, n_features, n_informative,r,saved_indexes):
    blocks_generated = np.empty([num_blocks, n_informative], dtype = 'int64')
    for i in range (0,num_blocks):
        rand_vect = r.choice(n_features,n_informative-len(saved_indexes), replace = False)
        rand_vect = np.append(rand_vect,saved_indexes)
        blocks_generated[i,:] = rand_vect
    return blocks_generated

def get_current_data(XTrain, XTest, blocks_generated_i):
    x_train_i =  XTrain[:,blocks_generated_i]
    x_test_i = XTest[:,blocks_generated_i]
    return x_train_i, x_test_i

def compute_mse(x_train_current_tmp, YTrain,x_test_current_tmp, YTest):
    mse = linear_model.LinearRegression(fit_intercept=False)
    mse.fit(x_train_current_tmp, YTrain)
    y_pred_test = mse.predict(x_test_current_tmp)
    new_loss = mean_squared_error(YTest,y_pred_test)
    beta = mse.coef_
    return new_loss, beta

def get_common_indexes(ordered_loss_ten,blocks_generated, n_features, ordered_beta):
    weights_indexes = np.zeros(n_features)
    count=0
    for i in ordered_loss_ten:
        weights_indexes[blocks_generated[i]] +=np.abs(ordered_beta[:,i])
        count+=1
    return weights_indexes

def extract_losses(indexes_losses):
    i=0
    ordered_losses_extract = np.array([],dtype = 'int64')
    while i<(10):
        ordered_losses_extract = np.append(ordered_losses_extract, indexes_losses[i])
        i+=1
    return ordered_losses_extract

def extracte_chosen_indexes(saved_indexes, ordered_weights_indexes, chosen_indexes):
    i=0
    inserted_indexes = 0
    while inserted_indexes<chosen_indexes:
        current_value = ordered_weights_indexes[i]
        if current_value not in saved_indexes:
            saved_indexes = np.append(saved_indexes, current_value)
            inserted_indexes+=1
        i+=1
    return saved_indexes

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
active_set = 250

saved_indexes = np.array([])
chosen_indexes = 3
num_informative = np.array([])

while len(num_informative)<n_informative and len(saved_indexes)<active_set:
    losses = np.array([])
    betas = np.array([])
    blocks_generated = generate_samples(num_blocks, n_features, active_set, r, saved_indexes)
    for i in range(0, num_blocks):
        x_train_i, x_test_i = get_current_data(XTrain, XTest, blocks_generated[i,:])
        new_loss,beta = compute_mse(x_train_i, YTrain,x_test_i, YTest)
        losses = np.append(losses, new_loss)
        beta = beta.reshape([len(beta),1])
        if len(betas)==0:
            betas = beta
        else:
            betas = np.append(betas, beta, axis =1)
    ordered_losses = np.argsort(losses)

    orderd_losses_ = np.sort(losses)
    ordered_loss_ten = extract_losses(ordered_losses)

    weights_indexes = get_common_indexes(ordered_loss_ten,blocks_generated,n_features,betas)
    ordered_weights_indexes = np.argsort(weights_indexes)[::-1]

    saved_indexes = extracte_chosen_indexes(saved_indexes, ordered_weights_indexes, chosen_indexes)
    num_informative = [f for f in saved_indexes if f<=99]
    print("num_inf", len(num_informative), "su", len(saved_indexes))
    print("indici salvati", saved_indexes)

np.savez("cross_val_blocks"+active_set+".npz", saved_indexes = saved_indexes, num_informative = num_informative)