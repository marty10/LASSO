from sklearn.externals import joblib
import numpy as np


class Result:
    def __init__(self, data_file_name, model_file_name):
        self.data_file_name = data_file_name
        self.model_file_name = model_file_name
        self.open_file = np.load(self.data_file_name)

    def extract_coords(self):
        Coord = self.open_file["Coord"]
        Coord_turb = self.open_file["Coord_turb"]
        power_curve = self.open_file["power_curve"]
        return Coord,Coord_turb, power_curve

    def extract_train_val(self):
        XTrain = self.open_file["XTrain_Val"]
        YTrain = self.open_file["YTrain_Val"]
        XVal = self.open_file["XVal"]
        YVal = self.open_file["YVal_"]
        return XTrain, YTrain, XVal, YVal

    def extract_data_transf(self):
        XTrainTransf = self.open_file["XTrainTransf_"]
        XTestTransf = self.open_file["XTestTransf_"]
        return XTrainTransf, XTestTransf

    def extract_train_test(self):
        XTrain = self.open_file["XTrain"]
        YTrain = self.open_file["YTrain"]
        XTest = self.open_file["XTest"]
        YTest = self.open_file["YTest"]
        return XTrain, YTrain, XTest, YTest

    def extract_mses(self):
        mses = self.open_file["mses"]
        return mses


    def extract_train_val_no_centered(self):
        YTrainVal_noCenter = self.open_file["YTrainVal_noCenter"]
        XTrain_ValNoCenter = self.open_file["XTrain_ValNoCenter"]
        XVal_noCenter = self.open_file["XValTransf_noCenter"]
        YVal_noCenter = self.open_file["YVal_noCenter"]
        return XTrain_ValNoCenter, YTrainVal_noCenter, XVal_noCenter,YVal_noCenter

    def get_saved_indexes(self):
        saved_indexes_list = self.open_file["saved_indexes_list"]
        return saved_indexes_list

    def extract_active_sets_indexes_beta(self):
        active_sets = self.open_file["active_sets"]
        ord_beta_div_zeros = self.open_file["ordered_indexes_div_zeros"]
        beta_div_zeros = self.open_file["beta_div_zeros"]
        indexes_to_extract = self.open_file["indexes_to_extract"]
        return active_sets, beta_div_zeros, ord_beta_div_zeros, indexes_to_extract

    def extract_indexes_tot(self):
        indexes_tot = self.open_file["indexes"]
        return indexes_tot

    def extract_count_div_beta_zeros(self):
        count_div_beta_zeros = self.open_file["count_div_beta_zeros"]
        return count_div_beta_zeros

    def extract_informative(self):
        informative_indexes = self.open_file["informative_indexes"]
        return informative_indexes

    def extract_weights(self):
        weights = self.open_file["weights_list"]
        return weights

    def extract_dict(self):
        dict_ = self.open_file["dict_"].item()
        return dict_

    def extract_mse_test(self):
        mse_test = self.open_file["mse_test"]
        return mse_test

    def extract_mse_train(self):
        mse_test = self.open_file["mse_train"]
        return mse_test

    def extract_model(self):
        model = joblib.load(self.model_file_name)
        return model

    def extract_lambda(self):
        file = np.load(self.data_file_name)
        lambda_lasso=file["best_lambda"]
        return lambda_lasso