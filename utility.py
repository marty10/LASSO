from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import numpy as np

def generate_samples(num_blocks, n_features, active_set,r,saved_indexes):
    blocks_generated = np.empty([num_blocks, active_set], dtype = 'int64')
    for i in range (0,num_blocks):
        gen_vect = np.arange(0,n_features)
        gen_vect = np.delete(gen_vect, saved_indexes)
        rand_vect = r.choice(gen_vect,active_set-len(saved_indexes), replace = False)
        rand_vect = np.append(rand_vect,saved_indexes)
        blocks_generated[i,:] = rand_vect
    return blocks_generated

def generate_samples_del(num_blocks, n_features, active_set,r,saved_indexes):
    blocks_generated = np.empty([num_blocks, active_set], dtype = 'int64')
    for i in range (0,num_blocks):
        gen_vect = np.arange(0,n_features)
        gen_vect = np.delete(gen_vect, saved_indexes)
        rand_vect = r.choice(gen_vect,active_set, replace = False)
        blocks_generated[i,:] = rand_vect
    return blocks_generated

def generate_train(num_blocks, n_samples, active_set,r):
    blocks_generated = np.empty([num_blocks, active_set], dtype = 'int64')
    for i in range (0,num_blocks):
        rand_vect = r.choice(n_samples,active_set, replace = False)
        blocks_generated[i,:] = rand_vect
    return blocks_generated

def get_current_data(XTrain, XTest, blocks_generated_i):
    x_train_i =  XTrain[:,blocks_generated_i]
    x_test_i = XTest[:,blocks_generated_i]
    return x_train_i, x_test_i

def get_current_train(XTrain, blocks_generated_i):
    x_train_i =  XTrain[blocks_generated_i,:]
    return x_train_i

def compute_mse(model,x_train_current_tmp,YTrain,x_test_current_tmp,YTest):
    model.fit(x_train_current_tmp, YTrain)
    y_pred_test = model.predict(x_test_current_tmp)
    new_loss = mean_squared_error(YTest,y_pred_test)
    beta = model.coef_
    beta = beta.reshape([len(beta),1])
    #coeff = compute_mse_coefficient(y_pred_test, YTest)
    corr = compute_corr(x_test_current_tmp, YTest, y_pred_test)
    corr = corr.reshape([len(corr),1])
    return new_loss, beta,np.abs(corr)

def compute_corr(x,y, y_pred):
    n,p = x.shape
    cor_xy = np.empty(p)
    for i in range(0,p):
        x_i = x[:,i]
        cor_xy[i],_= pearsonr(x_i, (y-y_pred))

    return cor_xy

def get_common_indexes( ordered_loss_ten,blocks_generated, n_features, betas):
    weights_indexes = np.zeros(n_features)
    count=0
    for i in ordered_loss_ten:
        weights_indexes[blocks_generated[i]] += np.abs(betas[:,i])#1./orderd_losses_[count]*np.abs(betas[:,i])
        count+=1
    return weights_indexes


def get_low_common_indexes(ordered_loss_ten,blocks_generated, n_features, betas):
    weights_indexes = np.zeros(n_features)
    count=0
    for i in ordered_loss_ten:
        weights_indexes[blocks_generated[i]] += -1-np.abs(betas[:,i])
        count+=1
    return weights_indexes


def get_low_common_indexes_corr(orderd_losses_, ordered_loss_ten,blocks_generated, n_features, betas):
    weights_indexes = np.zeros(n_features)
    count=0

    for i in ordered_loss_ten:
        current_corr = orderd_losses_[:,i]
        index_not_to_add = np.where(current_corr>0.2)[0]
        weights_indexes[blocks_generated[i]] += -np.abs(betas[:,i])
        a1 = weights_indexes[blocks_generated[i]]
        a1[index_not_to_add]-=np.abs(betas[:,i][index_not_to_add])
        weights_indexes[blocks_generated[i]] = a1
        count+=1
    return weights_indexes


def get_common_indexes1(orderd_losses_, ordered_loss_ten,blocks_generated, n_features, betas):
    weights_indexes = np.zeros(n_features)
    count=0

    for i in ordered_loss_ten:
        current_corr = orderd_losses_[:,i]
        index_not_to_add = np.where(current_corr<0.01)[0]
        weights_indexes[blocks_generated[i]] += np.abs(betas[:,i])
        a1 = weights_indexes[blocks_generated[i]]
        a1[index_not_to_add]-=np.abs(betas[:,i][index_not_to_add])
        weights_indexes[blocks_generated[i]] = a1
        count+=1
    return weights_indexes

def extract_losses(indexes_losses):
    i=0
    ordered_losses_extract = np.array([],dtype = 'int64')
    while i<(10):
        ordered_losses_extract = np.append(ordered_losses_extract, indexes_losses[i])
        i+=1
    return ordered_losses_extract

def extracte_chosen_indexes(saved_indexes, ordered_weights_indexes, values, chosen_indexes):
    i=0
    inserted_indexes = 0
    while inserted_indexes<chosen_indexes:
        current_value = ordered_weights_indexes[i]
        current = values[i]
        if current_value not in saved_indexes and current!=0:
            saved_indexes = np.append(saved_indexes, current_value)
            inserted_indexes+=1
        i+=1
    return saved_indexes

def compute_mean(XTrain):
    n,p = XTrain.shape
    mean_vector = np.zeros([n,p])
    for i in range(n):
        mean_vector[i] = np.mean(XTrain[i,:])
    return mean_vector


def compute_scatter_matrix(XTrain, mean_vector):
    n,p = XTrain.shape
    scatter_matrix = np.zeros([n,n])
    for i in range(p):
        scatter_matrix += (XTrain[:,i].reshape(n,1) - mean_vector).dot(
        (XTrain[:,i].reshape(n,1) - mean_vector).T)
    return scatter_matrix

def compute_mse_coefficient(y_pred, y_true):

    y_pred_m = y_pred-np.mean(y_pred)
    y_pred_m_2 = y_pred_m**2

    y_m = y_true-np.mean(y_true)
    y_m_2 = y_m**2
    y_sum = np.dot(y_pred_m,y_m)
    den = np.dot(y_pred_m_2, y_m_2)
    coeff = y_sum/np.sqrt(den)
    return coeff



