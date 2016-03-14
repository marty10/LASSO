# coding: utf-8
import numpy as np
from sklearn.cross_validation import train_test_split
from Enel_utils import compute_angle, create_dict_direction
from ExtractResult import Result
from Fit import Linear_fit
from Transformation import Enel_powerCurveTransformation, EnelWindSpeedTransformation, \
    Enel_directionPowerCurveTransformation

file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")
enel_dict = results.extract_dict()
Coord, Coord_turb, power_curve = results.extract_coords()

values_TM = np.array([[24,281], [24,214]])

XTrain, YTrain, XTest, YTest = results.extract_train_test()

XTrain, YTrain, XTest, YTest = results.extract_train_test()
enel_dict = results.extract_dict()
Coord, Coord_turb, power_curve = results.extract_coords()

angles_coord_turb = compute_angle(Coord, Coord_turb)

##transformation of data
X = np.concatenate((XTrain, XTest), axis = 0)
enel_transf = Enel_powerCurveTransformation()

enel_transf = Enel_powerCurveTransformation()
X_angle,_,dict_ = enel_transf.compute_angle_matrix(X)
output_dict = dict.fromkeys(np.arange(0,49),np.array([[]], dtype = "int64"))

k_levels = np.arange(0,12).reshape([12,1])
for key in np.arange(0,49):
    current_values = np.arange(key*12,key*12+12).reshape([12,1])
    output_dict[key] = np.concatenate((current_values,k_levels), axis = 1)


X_transf = np.array([[]])

X_speed,_ = EnelWindSpeedTransformation().transform(X)
print("wind speed computed")

enel_transf = Enel_directionPowerCurveTransformation()
X_transf, matrix_turbs = enel_transf.transform(X_angle, angles_coord_turb, X_speed, power_curve, Coord, Coord_turb)
print("single transformation done")
X_transf_1,_ = enel_transf.transformPerTurbine(matrix_turbs, enel_dict, X, power_curve, np.array([[]]),output_dict)
print("transformation per turbine done")
XTrain_transf = X_transf_1[:XTrain.shape[0],:]
XTest_transf = X_transf_1[XTrain.shape[0]:,:]


print("loss mean turbina solo sul validation")
XTrain_, XVal_, YTrain_, YVal_ = train_test_split(XTrain_transf, YTrain, test_size=0.33,random_state=0)
Linear_fit().fitting(XTrain_, YTrain_, XVal_,YVal_, [])

print("loss mean turbina solo sul test")
Linear_fit().fitting(XTrain_transf, YTrain, XTest_transf,YTest,values_TM)


print("loss mean turbine insieme")
X_transf = np.concatenate((X_transf,X_transf_1), axis = 1)
XTrain_transf = X_transf[:XTrain.shape[0],:]
XTest_transf = X_transf[XTrain.shape[0]:,:]

print("loss mean turbina solo sul validation")
XTrain_, XVal_, YTrain_, YVal_ = train_test_split(XTrain_transf, YTrain, test_size=0.33,random_state=0)
Linear_fit().fitting(XTrain_, YTrain_, XVal_,YVal_, [])

print("loss mean turbina solo sul test")
Linear_fit().fitting(XTrain_transf, YTrain, XTest_transf,YTest,values_TM)