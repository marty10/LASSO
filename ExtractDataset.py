import os
import openpyxl
import random
from operator import itemgetter
import numpy as np
import sklearn.feature_extraction
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.base import center_data
from Transformation import NullTransformation


class Dataset:

    def center_test(self, X, y, X_mean, y_mean, X_std, normalize = True):
        X_ = X.copy()
        X_ -= X_mean
        if normalize:
            X_ /= X_std
        y = y - y_mean
        return X_,y


class ArtificialDataset(Dataset):
    def __init__(self, n_samples, n_features, n_informative, normalize_y = False, normalize = True, centerdata = True,
                 transformation=NullTransformation(), fit_intercept = True):
        self.n_samples = n_samples
        self.n_features = n_features
        X, Y = datasets.make_regression(n_samples=self.n_samples, n_features=self.n_features,
                                                  n_informative=n_informative, shuffle=False, random_state=11)
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.33,random_state=0)
        self.XTrain_orig = XTrain
        self.XTest_orig = XTest
        self.YTrain_orig = YTrain
        self.YTest_orig = YTest
        if centerdata==True:
            self.XTrain, YTrain, X_mean, y_mean, X_std = center_data(XTrain, YTrain, fit_intercept=fit_intercept, normalize = normalize)
            self.XTest, YTest = self.center_test(XTest,YTest,X_mean,y_mean,X_std)
            if normalize_y:
                self.YTrain, self.YTest = self.normalize_labels(YTrain, YTest)
            else:
                self.YTrain = YTrain
                self.YTest = YTest
        else:
            self.XTrain = XTrain
            self.YTrain = YTrain
            self.XTest = XTest
            self.YTest = YTest
        self.transformation = transformation
        self.beta = beta

class ArtificialNonLinearDataset(Dataset):
     def __init__(self, n_samples, n_features, interval, test_size = 0.33, normalize = True, centerdata = True, transformation=NullTransformation(), fit_intercept = True):
        self.n_samples = n_samples
        self.n_features = n_features
        self.transformation = transformation
        lower = interval[0]
        upper = interval[1]
        random.seed(1)
        data = [np.array([random.uniform(lower, upper) for j in range(n_features)]) for i in range(n_samples)]
        Y = map(lambda x : self.transformation.transform(x), data)

        self.X = np.row_stack(data)
        self.informative = Y[0][1]
        self.Y = map(itemgetter(0), Y)
        XTrain, XTest, YTrain, YTest = train_test_split(self.X, self.Y, test_size=test_size,random_state=0)
        self.XTrain_orig = XTrain
        self.XTest_orig = XTest
        self.YTrain_orig = YTrain
        self.YTest_orig = YTest
        if centerdata==True:
            self.XTrain, self.YTrain, X_mean, y_mean, X_std = center_data(XTrain, YTrain, fit_intercept=fit_intercept, normalize = normalize)
            self.XTest, self.YTest = self.center_test(XTest,YTest,X_mean,y_mean,X_std)
        else:
            self.XTrain = XTrain
            self.YTrain = YTrain
            self.XTest = XTest
            self.YTest = YTest


class Libsvm_Dataset(Dataset):
    def __init__(self,dir,filename, centerdata = True, split = 0):
        if split:
            Y, X = svm_read_problem(dir+"/"+filename)
            X = self. compute_data(X)
            self.XTrain, self.XTest, self.YTrain, self.YTest = train_test_split(X, Y, test_size=0.33,random_state=0)
        else:
            self.YTrain, XTrain_dict = svm_read_problem(dir+"/"+filename+"/"+filename+"_train")
            self.YTest,XTest_dict = svm_read_problem(dir+"/"+filename+"/"+filename+"_test")
            self.XTrain = self.convert_to_matrix(XTrain_dict)
            self.XTest = self.convert_to_matrix(XTest_dict)
        if centerdata:
            self.XTrain, self.YTrain, X_mean, y_mean, X_std = center_data(self.XTrain, self.YTrain, fit_intercept=True, normalize = True)
            self.XTest, self.YTest = self.center_test(self.XTest,self.YTest,X_mean,y_mean,X_std)
            self.YTest = self.YTest-y_mean


    def convert_to_matrix(self,list_dict):
        v = sklearn.feature_extraction.DictVectorizer(sparse=True, dtype=float)
        v.vocabulary_ = {}
        v.feature_names_ = []
        union_dict = []
        for dict in list_dict:
            keys_dict = np.array(dict.keys())
            union_dict = np.union1d(union_dict,keys_dict).astype("int64")
        for e in union_dict:
            v.vocabulary_[e] = e-1
            v.feature_names_.append(e)
        print('v.vocabulary_ : {0} ; v.feature_names_: {1}'.format(v.vocabulary_, v.feature_names_))

        X = v.transform(list_dict)
        return X


    def compute_data(self,x):
        n_samples = len(x)
        max_key,tot_keys,tot_values = self.get_max_key(x)
        X = np.zeros([n_samples,max_key+1])
        for i in range(0,n_samples):
            X[i,tot_keys[i]] = tot_values[i]
        return X

    def get_max_key(self,x):
        max_key=0
        tot_keys = []
        tot_values = []
        for dict in x:
            keys_dict = np.array(dict.keys())
            keys_values = np.array(dict.values())
            max_tmp = max(keys_dict)
            if max_tmp>max_key:
                max_key=max_tmp
            tot_keys.append(keys_dict-1)
            tot_values.append(keys_values-1)
        return max_key-1,tot_keys,tot_values


class Enel_dataset(Dataset):
    def __init__(self, folder_train, folder_test, label_file, start_data_train = ["24/08/2012", "23"], end_data_train = ["31/05/2013", "23"],
                 start_data_test = ["01/06/2013", "0"], end_data_test = ["31/12/2013", "23"], centerdata = True):
        files = os.listdir(folder_train)

        X_indexes = [0,2]
        self.XTrain, self.YTrain = self.extract_data(folder_train, files, label_file, start_data_train, end_data_train, X_indexes, test_flag = 0)
        self.XTest, self.YTest = self.extract_data(folder_test, files, label_file, start_data_test, end_data_test, X_indexes, test_flag=1)

        if centerdata:
            self.XTrain, self.YTrain, X_mean, y_mean, X_std = center_data(self.XTrain, self.YTrain, fit_intercept=True, normalize = True)
            self.XTest, self.YTest = self.center_test(self.XTest,self.YTest,X_mean,y_mean,X_std)
            self.YTest = self.YTest-y_mean
        dict_ = dict.fromkeys(np.arange(0,49),np.array([]))
        for key in (list)(dict_.keys()):
            dict_[key] = np.arange(key*24,key*24+24)
        self.dict_ = dict_

    def extract_coordinates(self, folder):
        file_name = "Coordinate.xlsx"
        current_coord_file = folder + file_name
        wb = openpyxl.load_workbook(current_coord_file)
        sheet = wb.get_sheet_by_name('PSC')

        ###extract point coord
        X = np.zeros([49,2])
        j=0
        for col in range(1,3):
            X = self.extract_info_columns(sheet, col, j, X)
            j+=1

        ##extract turbine coord
        X_turb = np.zeros([39,2])
        j=0
        for col in range(4,6):
            X_turb = self.extract_info_columns(sheet, col, j,X_turb)
            j+=1
        return X, X_turb

    def extract_info_columns(self, sheet, col, j, X):
        i=0
        for cellObj in sheet.columns[col][1:]:
            X[i,j] = cellObj.value
            i+=1
            if i==X.shape[0]:
                break
        return X

    def extract_power_curve(self,file_name):
        X = np.genfromtxt(file_name, delimiter='\t')
        return X



    def extract_data(self, dir, files, label_file, start_data, end_data, X_indexes, test_flag ):
        X_ = np.array([])
        start_data_ = start_data[:]
        end_data_ = end_data[:]
        if test_flag==1:
            start_data_[1] = str(int(start_data[1])+24)
            end_data_[1] = str(int(end_data[1])+24)

        for e,enel_file in enumerate(files):
            current_enel_file = dir + enel_file
            if e==0:
                index_start,index_end, number_of_days = self.getRowsOfData(current_enel_file, start_data_, end_data_,X_indexes)
            X = np.genfromtxt(current_enel_file, delimiter='\t', usecols=(np.arange(27,51)))

            current_X = X[index_start:index_end+1,:]
            if X_.shape[0]==0:
                X_ = current_X.copy()
            else:
                X_ = np.concatenate((X_,current_X),axis = 1)

        # non serve, gli indici per la y sono gli stessi
        if test_flag==1:
            index_start = index_start+24
            index_end = index_end+24
            number_of_days_test = number_of_days
        else:
            number_of_days_train = number_of_days
        #start_data_y = [" ".join(start_data)+".00"]
        #end_data_y = [" ".join(end_data)+".00"]
        #index_start,index_end = self.getRowsOfData(label_file, start_data_y, end_data_y, Y_indexes)
        Y = np.genfromtxt(label_file, delimiter='\t', usecols=3)
        Y_ = Y[index_start:index_end+1]
        return X_, Y_,number_of_days_train,number_of_days_test

    def getRowsOfData(self, file_name, start_data, end_data, indexes):
        columns = []
        for line in open(file_name):
            columns.append((np.array(line.split('\t'))[indexes]).tolist())
        index_start = columns.index(start_data)
        index_end = columns.index(end_data)
        number_days = self.extract_number_of_days(columns, index_start, index_end)
        return index_start, index_end,number_days

    def extract_number_of_days(self,columns, index_start, index_end):
        columns_of_interest = columns[index_start:index_end+1]
        data_columns = zip(*columns_of_interest)
        data_column = list(data_columns[0])
        unique_data = np.unique(data_column)
        unique_data_len = len(unique_data)
        return unique_data_len

    def get_data(self):
        return self.XTrain, self.YTrain, self.XTest, self.YTest