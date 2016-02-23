import pickle
from sklearn.externals import joblib

__author__ = 'Martina'
import numpy as np


class Result:
    def __init__(self, data_file_name, model_file_name):
        self.data_file_name = data_file_name
        self.model_file_name = model_file_name

    def extract_train_val(self):
        file = np.load(self.data_file_name)
        XTrain = file["XTrain_Val"]
        YTrain = file["YTrain_Val"]
        XVal = file["XVal"]
        YVal = file["YVal_"]

        return XTrain, YTrain, XVal, YVal

    def extract_data_transf(self):
        file = np.load(self.data_file_name)
        XTrainTransf = file["XTrainTransf_"]
        XTestTransf = file["XTestTransf_"]
        return XTrainTransf, XTestTransf

    def extract_train_test(self):
        file = np.load(self.data_file_name)
        XTrain = file["XTrain"]
        YTrain = file["YTrain"]
        XTest = file["XTest"]
        YTest = file["YTest"]
        return XTrain, YTrain, XTest, YTest

    def extract_data(self):
        file = np.load(self.data_file_name)
        XTrain = file["XTrain"]
        YTrain = file["YTrain"]
        XTest = file["XTest"]
        YTest = file["YTest"]
        mses = file["mses"]
        return XTrain, YTrain, XTest, YTest,mses

    def extract_mse(self):
        file = np.load(self.data_file_name)
        mses = file["mses"]
        return mses

    def extract_active_sets_indexes_beta(self):
        file = np.load(self.data_file_name)
        active_sets = file["active_sets"]
        ord_beta_div_zeros = file["ordered_indexes_div_zeros"]
        beta_div_zeros = file["beta_div_zeros"]
        indexes_to_extract = file["indexes_to_extract"]
        return active_sets, beta_div_zeros, ord_beta_div_zeros, indexes_to_extract

    def extract_count_div_beta_zeros(self):
        file = np.load(self.data_file_name)
        count_div_beta_zeros = file["count_div_beta_zeros"]
        return count_div_beta_zeros

    def extract_informative(self):
        file = np.load(self.data_file_name)
        informative_indexes = file["informative_indexes"]
        return informative_indexes

    def extract_weights(self):
        file = np.load(self.data_file_name)
        weights = file["weights_list"]
        return weights

    def extract_dict(self):
        file = np.load(self.data_file_name)
        dict_ = file["dict_"][()]
        return dict_



    def extract_mse_test(self):
        file = np.load(self.data_file_name)
        mse_test = file["mse_test"]
        return mse_test

    def extract_model(self):
        model = joblib.load(self.model_file_name)
        return model

    def extract_mse_train(self):
        file = np.load(self.data_file_name)
        mse_test = file["mse_train"]
        return mse_test

    def extract_lambda(self):
        file = np.load(self.data_file_name)
        lambda_lasso=file["best_lambda"]
        return lambda_lasso