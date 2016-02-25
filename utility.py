import math
from scipy import spatial
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def extract_level(ordered_final_weights, values):
    weights_level = []
    for w in ordered_final_weights:
        key = np.where(values==w)[0][0]
        a = values[key]
        level = np.where(values[key]==w)[0][0]
        weights_level.append([key, level])
    return np.array(weights_level, dtype="int64")

def compute_weightedLASSO(lasso,XTrain_current,YTrain, XTest_current, YTest,scoring, score_f, verbose ):

    alphas = _alpha_grid(XTrain_current, YTrain, fit_intercept=False)
    parameters = {"alpha": alphas}

    clf = GridSearchCV(lasso, parameters, fit_params = {"verbose" : False}, cv=3, scoring=scoring)
    clf.fit(XTrain_current, YTrain)
    lambda_opt = clf.best_params_

    print("best lambda", lambda_opt)

    lasso.set_params(**lambda_opt)
    lasso.fit(XTrain_current,YTrain)

    y_pred_train = lasso.predict(XTrain_current)
    mse_train = score_f(YTrain, y_pred_train)
    abs_error_train = 100*mean_absolute_error(YTrain,y_pred_train)#*len(YTrain)#/(89.7*9*331)
    if verbose:
        print("mse_train "+lasso.__class__.__name__,mse_train)
        print("abs train", abs_error_train)
    y_pred_test = lasso.predict(XTest_current)
    mse_test = score_f(YTest, y_pred_test)
    abs_error_test = 100*mean_absolute_error(YTest,y_pred_test)#*len(YTest)/(89.7*16*165)
    if verbose:
        print ("mse_test weights "+lasso.__class__.__name__,mse_test)
        print("abs test", abs_error_test)

    return mse_test, lasso.beta

def get_beta_div_zeros(beta):
    beta_ord = np.sort(beta)[::-1]
    beta_ordered = beta_ord[beta_ord >= 0.1]
    len_div_zero = len(beta_ordered)
    beta_indexes = np.argsort(np.abs(beta))[::-1][:len_div_zero]
    return beta_indexes, beta_ordered

def compute_lasso(XTrain, YTrain, XTest, YTest, score):
    lasso_cv = linear_model.LassoCV(fit_intercept=False,  max_iter=10000, n_jobs = -1)
    lasso_cv.fit(XTrain,YTrain)
    best_alpha = lasso_cv.alpha_

    model = linear_model.Lasso(fit_intercept=False,alpha=best_alpha)
    new_loss,beta = compute_mse(model, XTrain, YTrain,XTest, YTest, score)
    return new_loss,beta

def generate_samples_dynamic_set(num_blocks, n_features, r,saved_indexes,r1, min, max,active_set,max_active_set):
    current_lenght = len(saved_indexes)
    diff = max-min
    if active_set!=0:
        min= np.abs(active_set-current_lenght)+1
        max = min+diff
    if current_lenght+max<=n_features:
        start = current_lenght+min
        end = current_lenght+max
    else:
        end = n_features+1
        start = end-1
    if current_lenght>=active_set:
        active_set = r1.choice(np.arange(start,end), 1, replace=False)[0]
    #active_set = 100
    if active_set>max_active_set:
        active_set = max_active_set
    binom_coeff = binomialCoefficient(n_features,active_set)
    if binom_coeff<num_blocks:
        print("###########################################coef binomiale", binom_coeff)
        num_blocks=binom_coeff
    else:
        num_blocks=1000
    print("active_set", active_set)
    blocks_generated = np.empty([num_blocks, active_set], dtype = 'int64')
    gen_vect = np.arange(0,n_features)
    gen_vect = np.delete(gen_vect, saved_indexes)

    for i in range(0,num_blocks):
        rand_vect = r.choice(gen_vect,active_set-len(saved_indexes), replace = False)
        rand_vect = np.append(rand_vect,saved_indexes)
        blocks_generated[i,:] = rand_vect
    return blocks_generated,active_set,num_blocks


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

def get_current_samples(XTrain, blocks_generated_i):
    x_train_i =  XTrain[blocks_generated_i,:]
    return x_train_i

def compute_mse(model,x_train_current_tmp,YTrain,x_test_current_tmp,YTest, score):
    model.fit(x_train_current_tmp, YTrain)
    y_pred_test = model.predict(x_test_current_tmp)

    if score=="mean_squared_error":
        new_loss = mean_squared_error(YTest,y_pred_test)
    elif score== "mean_absolute_error":
        new_loss = mean_absolute_error(YTest,y_pred_test)
    else:
        new_loss = r2_score(YTest,y_pred_test)
    beta = model.coef_
    if x_train_current_tmp.shape[1]==1:
        beta = np.array([beta])
    beta = beta.reshape([len(beta),1])

    return new_loss, beta

def print_features_active(keys_sel, indexes, dict_):
    for key in keys_sel:
        for i in indexes:
            if i in dict_.get(key):
                print (key)
                break

def compute_mse_binary(model,x_train_current_tmp,YTrain,x_test_current_tmp,YTest):
    model.fit(x_train_current_tmp, YTrain)
    y_pred_test = model.predict(x_test_current_tmp)
    n = len(YTest)
    signs = np.sign(YTest) == np.sign(y_pred_test)
    new_loss = 1./n*sum(signs==True)
    beta = model.coef_
    beta = beta.reshape([len(beta),1])
    return new_loss, beta

def compute_ber_binary(model,x_train_current_tmp,YTrain,x_test_current_tmp,YTest):
    model.fit(x_train_current_tmp, YTrain)
    y_pred_test = model.predict(x_test_current_tmp)

    positive_ones = np.where(np.sign(y_pred_test)==1)[0]
    negative_ones = np.where(np.sign(y_pred_test)==-1)[0]

    positive_ones_real = np.where(np.sign(YTest)==1)[0]
    negative_ones_real = np.where(np.sign(YTest)==-1)[0]

    fp = float(len(np.intersect1d(positive_ones,negative_ones_real)))
    tn = (float)(len(np.intersect1d(negative_ones,negative_ones_real)))
    tp = (float)(len(np.intersect1d(positive_ones,positive_ones_real)))
    fn = (float)(len(np.intersect1d(negative_ones,positive_ones_real)))

    if fp!=0:
        first_term = fp/(tn+fp)
    else:
        first_term = 0
    if fn!=0:
        second_term = fn/(fn+tp)
    else:
        second_term = 0
    new_loss = 0.5*(first_term+second_term)
    beta = model.coef_
    beta = beta.reshape([len(beta),1])
    return new_loss, beta


def compute_corr(x,y, y_pred):
    n,p = x.shape
    cor_xy = np.empty(p)
    for i in range(0,p):
        x_i = x[:,i]
        cor_xy[i],_= pearsonr(x_i, (y-y_pred))

    return cor_xy

def get_common_indexes(weights_indexes,ordered_loss_ten,blocks_generated, betas):
    count=0
    for i in ordered_loss_ten:
        weights_indexes[blocks_generated[i]] += betas[:,i]
        count+=1
    return np.abs(weights_indexes), np.sign(weights_indexes)

def get_common_indexes(weights_indexes,ordered_loss_ten,blocks_generated, betas, n_features):
    count=0
    current_weight = np.zeros(n_features)
    for i in ordered_loss_ten:
        weights_indexes[blocks_generated[i]] += betas[:,i]
        current_weight[blocks_generated[i]]+= betas[:,i]
        count+=1
    weights_indexes = np.abs(weights_indexes)
    return weights_indexes, np.sign(current_weight)

def get_common_indexes_threshold(weights_indexes,ordered_loss_ten,blocks_generated, betas,threshold):
    count=0
    for i in ordered_loss_ten:
        current_beta = betas[:,i]
        weights_indexes[blocks_generated[i]] += compute_threshold(current_beta,threshold)
        count+=1
    return np.abs(weights_indexes), np.sign(weights_indexes)

def compute_threshold(current_beta,threshold):
    new_beta = np.zeros(len(current_beta))
    for i,beta_j in enumerate(current_beta):
        if beta_j>threshold and beta_j>0:
            new_beta[i] = 1
        elif beta_j<-threshold and beta_j<0:
            new_beta[i] = -1
        else:
            new_beta[i] = 0
    return new_beta


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

def extract_losses(indexes_losses, number_loss):
    i=0
    ordered_losses_extract = np.array([],dtype = 'int64')
    while i<number_loss:
        ordered_losses_extract = np.append(ordered_losses_extract, indexes_losses[i])
        i+=1
    return ordered_losses_extract

def extract_chosen_indexes(saved_indexes, ordered_weights_indexes, values, chosen_indexes):
    i = 0
    inserted_indexes = 0
    while inserted_indexes < chosen_indexes:
        current_value = ordered_weights_indexes[i]
        current = values[i]
        if current_value not in saved_indexes and np.abs(current) > 0.1:
            saved_indexes = np.append(saved_indexes, current_value)
            inserted_indexes += 1
        i += 1
    return saved_indexes

def extract_chosen_indexes_from_start(saved_indexes, ordered_weights_indexes,chosen_indexes):
    i = 0
    inserted_indexes = 0
    length = len(saved_indexes)
    old_saved_indexes = saved_indexes.copy()
    saved_indexes = np.array([],dtype = "int64")
    while inserted_indexes < chosen_indexes+length:
        current_value = ordered_weights_indexes[i]
        if current_value not in saved_indexes:
            saved_indexes = np.append(saved_indexes, current_value)
            inserted_indexes += 1
        i += 1
    inters = np.intersect1d(old_saved_indexes, saved_indexes)

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
    lenght = len(saved_indexes)
    saved_indexes = np.array([], dtype = "int64")
    old_values = np.array([], dtype = "int64")
    while inserted_indexes < chosen_indexes+lenght:
        current_value = ordered_weights_indexes[i]
        current = values[i]
        if current_value not in saved_indexes and current != 0 and current_value not in removed_indexes:
            saved_indexes = np.append(saved_indexes, current_value)
            old_values = np.append(old_values, current)
            inserted_indexes += 1
        i += 1
    assert(len(old_values)==len(saved_indexes))
    return saved_indexes, old_values

def value_beta_sign(beta_sign_old, beta_sign, old_values, saved_indexes, removed_indexes):
    index_to_delete = np.array([], dtype = "int64")
    for i,current_value in enumerate(saved_indexes):
        if beta_sign_old[current_value]!=beta_sign[current_value]:
            index_to_delete = np.append(index_to_delete, i)
            removed_indexes = np.append(removed_indexes, current_value)
    saved_indexes = np.delete(saved_indexes, index_to_delete)
    old_values = np.delete(old_values, index_to_delete)
    return saved_indexes, old_values, np.unique(removed_indexes)

def value_beta(old_values, saved_indexes,ordered_weights_indexes,values,removed_values):

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
                if key>=0:
                    key+=1
                if key<0:
                    key-=1
                if key in current_dict:
                    current_dict[key] += 1
                else:
                    current_dict[key] = 1
    return dictlist

def get_common_indexes_binning_threshold(ordered_loss_ten,blocks_generated, betas,dictlist):
    step = 500
    for i in ordered_loss_ten:
        beta_indexes = blocks_generated[i]
        current_beta = betas[:,i]
        assert(len(current_beta)==len(beta_indexes))
        for k,j in enumerate(beta_indexes):
            beta = current_beta[k]
            current_dict = dictlist[j]
            key = int(beta/step)
            if key==0:
                key=0.5
            if np.abs(beta)<=1:
                if key in current_dict:
                    current_dict[key] -= 1
                else:
                    current_dict[key] = -1
            elif key in current_dict:
                current_dict[key] += 1
            else:
                current_dict[key] = 1

    return dictlist

def extract_max_from_beta(dictlist):
    weights_indexes = np.zeros(len(dictlist))
    for j,dict in enumerate(dictlist):
        if len(dict)!=0:
            dict_values = np.array(list(dict.values()))
            dict_values[dict_values<0] = 0
            dict_keys = np.array(list(dict.keys()))
            key_values = dict_values*dict_keys
            max_value = np.max(np.abs(key_values))
            weights_indexes[j] = max_value
    return weights_indexes


def extract_results(filename):
    active_set = np.load(filename)
    active_indexes = active_set["saved_indexes_list"]
    last_active_indexes = active_indexes[len(active_indexes)-1]
    return last_active_indexes


def extract_informative(active_set,p):
    beta_index = np.where(active_set<=p)[0]
    beta_index_chosen = active_set[beta_index]
    return beta_index, beta_index_chosen


def read_libsvm_dataset(filename):
    y, x = svm_read_problem(filename)
    #np.savez("cancer_dataset.npz", y = y, x = x)
    keys = get_common_key(x)
    return x,y,keys

def get_common_key(x):
    i=0
    keys = np.array([], "int64")
    for dict in x:
        keys_dict = np.array(dict.keys())
        if i==0:
            keys = keys_dict
        else:
            keys = np.intersect1d(keys_dict, keys)
        i+=1
    return keys


def compute_data(y,x,keys):
    n_samples = len(x)
    X = np.zeros([n_samples,len(keys)])
    for i in range(0,n_samples):
        current_feat = [val for (key,val) in x[i].iteritems() if key in keys]
        X[i,:] = current_feat
    return X,y


def assign_weights(weights_ordered_indexes):
    mean_weigths = np.mean(weights_ordered_indexes)
    weights_ordered_indexes[weights_ordered_indexes<mean_weigths]=1
    weights_ordered_indexes[weights_ordered_indexes>=mean_weigths] = float(mean_weigths)/(weights_ordered_indexes[weights_ordered_indexes>=mean_weigths])
    return weights_ordered_indexes

def assign_weights_ordered(weights_ordered_indexes,active_set):
     weights_ordered_indexes[:active_set-1] = 0
     weights_ordered_indexes[active_set-1:] = 1
     return weights_ordered_indexes

def center_test(X, y, X_mean, y_mean, X_std, normalize = True):
    X_copy = X.copy()
    X_copy -= X_mean
    y_copy = y.copy()
    if normalize:
        X_copy /= X_std
        y_copy = y_copy - y_mean
    return X_copy,y_copy


def binomialCoefficient(p,s):
    a = math.factorial(p)
    b = math.factorial(s)
    c = math.factorial(p-s)  # that appears to be useful to get the correct result
    div = a // (b * c)

    return div


def find_nearest(Coord,k):
    dim = Coord.shape[0]
    dict_ = dict.fromkeys(np.arange(0,dim),np.array([]))
    x = Coord[:,0]
    y = Coord[:,1]
    tree = spatial.KDTree(zip(x.ravel(), y.ravel()))
    for i in range(0,dim):
        d,j = tree.query(Coord[i,:],k = k)
        j = j[j>=i]
        dict_[i] = j
    return dict_