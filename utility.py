from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import numpy as np

def generate_samples(num_blocks, n_features, active_set,r,saved_indexes,deleted_indexes):
    blocks_generated = np.empty([num_blocks, active_set], dtype = 'int64')
    for i in range (0,num_blocks):
        gen_vect = np.arange(0,n_features)
        gen_vect = np.delete(gen_vect, saved_indexes)
        if len(deleted_indexes)!=0:
            gen_vect = np.intersect1d(gen_vect,deleted_indexes)
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

def get_common_indexes(ordered_loss_ten,blocks_generated, n_features, betas):
    weights_indexes = np.zeros(n_features)
    count=0
    for i in ordered_loss_ten:
        weights_indexes[blocks_generated[i]] += betas[:,i]#1./orderd_losses_[count]*np.abs(betas[:,i])
        count+=1
    return np.abs(weights_indexes), np.sign(weights_indexes)


def get_low_common_indexes(ordered_loss_ten,blocks_generated, n_features, betas):
    weights_indexes = np.zeros(n_features)
    for i in ordered_loss_ten:
        index_first_inserted = np.where(weights_indexes[blocks_generated[i]]==0)[0]
        a1 = weights_indexes[blocks_generated[i]]
        a1[index_first_inserted] = 1
        weights_indexes[blocks_generated[i]] = a1
        weights_indexes[blocks_generated[i]] += np.abs(betas[:,i])
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
    while i<100:
        ordered_losses_extract = np.append(ordered_losses_extract, indexes_losses[i])
        i+=1
    return ordered_losses_extract

def extracte_chosen_indexes(saved_indexes, ordered_weights_indexes, values, chosen_indexes):
    i = 0
    inserted_indexes = 0
    length = len(saved_indexes)
    saved_indexes = np.array([],dtype = "int64")
    while inserted_indexes < chosen_indexes+length:
        current_value = ordered_weights_indexes[i]
        current = values[i]
        if current_value not in saved_indexes and current != 0:
            saved_indexes = np.append(saved_indexes, current_value)
            inserted_indexes += 1
        i += 1
    return saved_indexes

def extracte_chosen_indexes_beta_check(old_values, saved_indexes, ordered_weights_indexes, values, chosen_indexes):
    i = 0
    inserted_indexes = 0
    saved_indexes, old_values = value_beta(old_values, saved_indexes,ordered_weights_indexes,values)

    while inserted_indexes < chosen_indexes:
        current_value = ordered_weights_indexes[i]
        current = values[i]
        if current_value not in saved_indexes and current != 0:
            saved_indexes = np.append(saved_indexes, current_value)
            old_values = np.append(old_values, current)
            inserted_indexes += 1
        i += 1
        assert(len(old_values)==len(saved_indexes))
    return saved_indexes, old_values

def extract_indexes_beta_sign_check(old_values, saved_indexes, ordered_weights_indexes, values, chosen_indexes,removed_indexes):
    i = 0
    inserted_indexes = 0

    while inserted_indexes < chosen_indexes:
        current_value = ordered_weights_indexes[i]
        current = values[i]
        if current_value not in saved_indexes and current != 0 and current_value not in removed_indexes:
            saved_indexes = np.append(saved_indexes, current_value)
            old_values = np.append(old_values, current)
            inserted_indexes += 1
        i += 1
        assert(len(old_values)==len(saved_indexes))
    return saved_indexes, old_values

def value_beta_sign(beta_sign_old, beta_sign, old_values, saved_indexes,ordered_weights_indexes):
    removed_indexes = np.array([], dtype = "int64")
    index_to_delete = np.array([], dtype = "int64")
    for i,current_value in enumerate(saved_indexes):
        if beta_sign_old[current_value]!=beta_sign[current_value]:
            index_to_delete = np.append(index_to_delete, i)
            removed_indexes = np.append(removed_indexes, current_value)
    saved_indexes = np.delete(saved_indexes, index_to_delete)
    old_values = np.delete(old_values, index_to_delete)
    return saved_indexes, old_values, removed_indexes

def value_beta(old_values, saved_indexes,ordered_weights_indexes,values):
    removed_values = np.array([], dtype= "int64")
    for current_value in saved_indexes:
        i = np.where(ordered_weights_indexes == current_value)[0]
        old_v = np.where(saved_indexes == current_value)[0]
        current = values[i]
        if old_values[old_v]/current>1:
            removed_values = np.append(removed_values, current_value)
            saved_indexes = np.delete(saved_indexes, old_v)
            old_values = np.delete(old_values, old_v)
        else:
            old_values[old_v] = current
    return saved_indexes, old_values, removed_values

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



def compute_active_set(active_sets,p, intersections):
    active_set = intersections.copy()
    count = 0
    num_inserted = len(active_set)
    counter = 0
    len_set = len(active_sets[0])
    while num_inserted<p and count<len_set:
        for set in active_sets:
            current_el = set[count]
            if current_el not in active_set:
                active_set = np.append(active_set,current_el)
                num_inserted+=1
            if num_inserted==p:
                break
            if counter==2:
                count+=1
                counter=0
            else:
                counter+=1
    return active_set

def update_removed_indexes(indexes, updated_indexes):
    tmp = np.zeros(len(indexes))
    indexes = np.sort(indexes)
    count = 0
    for i in indexes:
        less_i = len(np.where(updated_indexes<=i)[0])
        if less_i==0:
            tmp[count] = i
        else:
            tmp[count] = i+less_i
        count+=1
    updated_indexes = np.append(updated_indexes,tmp)
    return updated_indexes


def get_common_indexes_binning(ordered_loss_ten,blocks_generated, betas,dictlist):
    step = 100
    for i in ordered_loss_ten:
        beta_indexes = blocks_generated[i]
        current_beta = betas[:,i]
        assert(len(current_beta)==len(beta_indexes))
        for k,j in enumerate(beta_indexes):
            beta = current_beta[k]
            current_dict = dictlist[j]
            if np.abs(beta)>=1:
                key = int(beta/step)
                if key in current_dict:
                    current_dict[key] += 1
                else:
                    current_dict[key] = 1
    return dictlist

def extract_max_from_beta(dictlist):
    weights_indexes = np.zeros(len(dictlist))
    for j,dict in enumerate(dictlist):
        if len(dict)!=0:
            dict_values = np.array(dict.values())
            dict_keys = np.array(dict.keys())
            key_values = dict_values*(dict_keys+1)
            max_value = np.max(np.abs(key_values))
            weights_indexes[j] = max_value
    return weights_indexes


def extract_results(filename):
    active_set = np.load(filename)
    active_indexes = active_set["saved_indexes_list"]
    last_active_indexes = active_indexes[len(active_indexes)-1][:100]
    return last_active_indexes


def extract_informative(active_set,p):
    beta_index = np.where(active_set<=p)[0]
    beta_index_chosen = active_set[beta_index]
    return beta_index, beta_index_chosen