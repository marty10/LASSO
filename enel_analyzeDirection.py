import numpy as np
from sklearn.cross_validation import train_test_split
from ExtractResult import Result

##load the dataset
from Fit import Linear_fit

file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")

values_TM = np.array([[24,281], [24,214]])

_, YTrain, _, YTest = results.extract_train_test()

file_turb = np.load("ENEL_2014/turbine_180.npz")

XTrain_transf = file_turb["XTrain_transf"]
XTest_transf = file_turb["XTest_transf"]

##analyze turbine alone
print("loss 1 turibina solo sul validation")
XTrain_, XVal_, YTrain_, YVal_ = train_test_split(XTrain_transf, YTrain, test_size=0.33,random_state=0)
Linear_fit().fitting(XTrain_, YTrain_, XVal_,YVal_)

print("loss 1 turibina solo sul test")
Linear_fit().fitting(XTrain_transf, YTrain, XTest_transf,YTest,values_TM)

