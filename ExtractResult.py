import pickle
from sklearn.externals import joblib

__author__ = 'Martina'
import numpy as np


class Result:
    def __init__(self, data_file_name, model_file_name):
        self.data_file_name = data_file_name
        self.model_file_name = model_file_name

    def extract_data(self):
        file = np.load(self.data_file_name)
        XTrain = file["XTrain"]
        YTrain = file["YTrain"]
        XTest = file["XTest"]
        YTest = file["YTest"]
        return XTrain, YTrain, XTest, YTest

    def extract_data_transf(self):
        file = np.load(self.data_file_name)
        XTrainTransf = file["XTrainTransf"]
        XTestTransf = file["XTestTransf"]
        return XTrainTransf, XTestTransf

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