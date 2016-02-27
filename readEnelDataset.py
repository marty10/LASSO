from ExtractDataset import Enel_dataset
from ExtractResult import Result
from Fit import  Linear_fit
from Transformation import EnelWindSpeedTransformation
import sys
from utility import find_nearest
import numpy as np

folder_train = "ENEL_2014/PSC/0-23_0001-0049/"
folder_test = "ENEL_2014/PSC/24-47_0001-0049/"
label_file = "ENEL_2014/PSC/Metering_2011-2014_UTC.txt"

sys.argv[1:] = [int(x) for x in sys.argv[1:]]
k = sys.argv[1]

Coord = np.load("ENEL_2014/Coord.npz")["Coord"]

neight_= find_nearest(Coord,k)
print(neight_)
#XTrain, YTrain, XTest, YTest = enel_dataset.get_data()
#np.savez("Enel_dataset.npz", XTrain = XTrain, XTest = XTest, YTrain = YTrain, YTest = YTest)

file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")

XTrain, YTrain, XTest, YTest = results.extract_train_test()

YTrain = YTrain**(1. / 3)
YTest = YTest**(1. / 3)

enel_transf = EnelWindSpeedTransformation()
XTrain, dict_ = enel_transf.transform(XTrain)
XTrain = enel_transf.nearest_products(neight_,dict_,XTrain)

print(XTrain.shape)

XTest, dict_ = EnelWindSpeedTransformation().transform(XTest)
XTest = enel_transf.nearest_products(neight_,dict_,XTest)

print(XTest.shape)
Linear_fit().fitting(XTrain, YTrain, XTest,YTest )

#Power_fit().fitting(XTrain, YTrain, XTest,YTest)
#Polynomial_fit().fitting(XTrain, YTrain, XTest,YTest)

#YTrain = np.array(YTrain)
#YTest = np.array(YTest)
#Log_y_fit().fitting(XTrain, YTrain, XTest,YTest)

#Inverse_y_fit().fitting(XTrain, YTrain, XTest, YTest)


#Enel_power3().fitting(XTrain, YTrain, XTest,YTest)
#Enel_gaussianKernel().fitting(XTrain, YTrain, XTest,YTest)


#Inverse_x_fit().fitting(XTrain, YTrain, XTest, YTest)

#Sqrt_fit().fitting(XTrain, YTrain, XTest,YTest)

#Log_x_fit().fitting(XTrain, YTrain, XTest,YTest)

#Log_xy_fit.fitting(XTrain, YTrain, XTest,YTest)

#X = np.array([[1,2,3],[0,1,2]])

#transf = Gaussian(sigma = 0.5)
#x_transf = transf.transform(X)