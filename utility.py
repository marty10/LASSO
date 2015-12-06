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

def get_current_data(XTrain, XTest, blocks_generated_i):
    x_train_i =  XTrain[:,blocks_generated_i]
    x_test_i = XTest[:,blocks_generated_i]
    return x_train_i, x_test_i

def compute_mse(model,x_train_current_tmp,YTrain,x_test_current_tmp,YTest):
    model.fit(x_train_current_tmp, YTrain)
    y_pred_test = model.predict(x_test_current_tmp)
    new_loss = mean_squared_error(YTest,y_pred_test)
    beta = model.coef_
    return new_loss, beta

def get_common_indexes(orderd_losses_, ordered_loss_ten,blocks_generated, n_features, betas):
    weights_indexes = np.zeros(n_features)
    count=0
    for i in ordered_loss_ten:
        weights_indexes[blocks_generated[i]] += np.abs(betas[:,i])#1./orderd_losses_[count]*np.abs(betas[:,i])
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
