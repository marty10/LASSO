# coding: utf-8
import numpy as np
from sklearn.cross_validation import train_test_split
from Enel_utils import compute_angle, create_dict_direction
from ExtractResult import Result
from Fit import Linear_fit
from Transformation import Enel_powerCurveTransformation, EnelWindSpeedTransformation, \
    Enel_directionPowerCurveTransformation, Enel_turbineTransformation

file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")
enel_dict = results.extract_dict()
Coord, Coord_turb, power_curve = results.extract_coords()


compute_single = 1
compute_levels = 1
compute_not_levels = 0
compute_single_mean = 0
values_TM = np.array([[24,281], [24,214]])

XTrain, YTrain, XTest, YTest = results.extract_train_test()


##transformation of data
X = np.concatenate((XTrain, XTest), axis = 0)
X_speed,_,_ = EnelWindSpeedTransformation().transform(X)
print("wind speed computed")

print("-------")

output_dict = dict.fromkeys(np.arange(0,49),np.array([[]], dtype = "int64"))

k_levels = np.arange(0,12).reshape([12,1])
for key in np.arange(0,49):
    current_values = np.arange(key*12,key*12+12).reshape([12,1])
    output_dict[key] = np.concatenate((current_values,k_levels), axis = 1)

enel_transf = Enel_turbineTransformation()
X_transf, matrix_turbs = enel_transf.transform(X_speed, power_curve)
print("single transformation done")

if compute_single:

    XTrain_transf = X_transf[:XTrain.shape[0],:]
    XTest_transf = X_transf[XTrain.shape[0]:,:]

    print("loss 1 turibina solo sul validation")
    XTrain_, XVal_, YTrain_, YVal_ = train_test_split(XTrain_transf, YTrain, test_size=0.33,random_state=0)
    Linear_fit().fitting(XTrain_, YTrain_, XVal_,YVal_, [])

    print("loss 1 turbina solo sul test")
    Linear_fit().fitting(XTrain_transf, YTrain, XTest_transf,YTest,values_TM)

print("-------------")