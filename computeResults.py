from sklearn.metrics import mean_squared_error
from ExtractResult import Result
import numpy as np

model_list = {"ISTA", "ADMM", "FISTA", "Shooting"}
#model_list = {"LassoLars"}
ext_data = ".npz"
ext_model = ".pkl"
folder = "AlgorithmResults/Polynomial2/"

for model in model_list:
    result = Result(folder + "/" + model + ext_data, folder + "/"+model + "_model" + ext_model)
    XTrain, YTrain, XTest, YTest = result.extract_data()
    XTrainTransf, XTestTransf = result.extract_data_transf()

    lasso = result.extract_model()
    m = result.extract_model()
    beta = m.beta
    y_pred_train = lasso.predict(XTrainTransf)
    mse_train = mean_squared_error(YTrain, y_pred_train)

    loss = 0.5 * np.sum((np.dot(XTrainTransf, beta) - YTrain) ** 2.0) + lasso.lambda_lasso * np.sum(abs(beta))

    beta_informative = beta[:1000]
    #eta_informative2 = beta[2000:3000]
    #beta_n_informative = beta[1000:2000]
    #beta_n_informative = beta[3000:4000]
    #n_n_informative_zero1 = [x for x in beta_n_informative if x==0]
    #n_n_informative_zero2 = [x for x in beta_n_informative if x==0]

    #print(len(n_n_informative_zero1)+len(n_n_informative_zero2))

    n_informative_zero1 = [x for x in beta_informative if x==0]
    #n_informative_zero2 = [x for x in beta_informative2 if x==0]
    print(model+' | beta_norm | loss | mse_train | mse_test | n_informative_zero')
    print(sum(abs(beta)),'|',loss,'|',mse_train,'|',result.extract_mse_test(), '|',len(n_informative_zero1))#+ len(n_informative_zero2))
