import numpy as np
from sklearn.cross_validation import train_test_split
from Enel_utils import compute_angular_coefficient, extract_direction, create_dict_direction
from ExtractResult import Result
from Fit import Linear_fit
from Transformation import Enel_powerCurveTransformation


##load the dataset
file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")
enel_dict = results.extract_dict()
Coord, Coord_turb, power_curve = results.extract_coords()

values_TM = np.array([[24,281], [24,214]])

XTrain, YTrain, XTest, YTest = results.extract_train_test()

angular_coeffs = compute_angular_coefficient(Coord,Coord_turb)
directions = extract_direction(angular_coeffs)

print("direzioni estratte")
enel_transf = Enel_powerCurveTransformation()
XTrain_angle,dict_ = enel_transf.compute_angle_matrix(XTrain)
directions_train = extract_direction(XTrain_angle)

print("direction train estratte")
turbine_dict = create_dict_direction(directions_train,directions)
np.savez("ENEL_2014/turbine_180", turbine_dict = turbine_dict)

print("turbine dict estratto")
output_dict_ = dict.fromkeys(np.arange(0,49),np.array([[]], dtype = "int64"))

XTrain_transf = np.array([[]])
XTest_transf = np.array([[]])

XTrain_transf, output_dict_ = enel_transf.transform1(turbine_dict, enel_dict, XTrain, power_curve,0, XTrain_transf, output_dict_)
np.savez("ENEL_2014/turbine_180", turbine_dict = turbine_dict, XTrain_transf = XTrain_transf)
print("XTrain 1 turbina fatto")

XTest_transf, _ = enel_transf.transform1(turbine_dict, enel_dict, XTest, power_curve,0, XTest_transf, output_dict_.copy())
np.savez("ENEL_2014/turbine_180", turbine_dict = turbine_dict, XTrain_transf = XTrain_transf, XTest_transf = XTest_transf)
print("XTest media turbina fatto")
##analyze turbine alone
print("loss 1 turibina solo sul validation")
XTrain_, XVal_, YTrain_, YVal_ = train_test_split(XTrain_transf, YTrain, test_size=0.33,random_state=0)
Linear_fit().fitting(XTrain_, YTrain_, XVal_,YVal_)

print("loss 1 turibina solo sul test")
Linear_fit().fitting(XTrain_transf, YTrain, XTest_transf,YTest,values_TM)


##compute mean turbine
enel_transf = Enel_powerCurveTransformation()
XTrain_transf, output_dict_ = enel_transf.transform1(turbine_dict, enel_dict, XTrain, power_curve,1, XTrain_transf, output_dict_)
XTest_transf, _ = enel_transf.transform1(turbine_dict, enel_dict, XTest, power_curve,1, XTest_transf, output_dict_.copy())

print("loss media turibina solo sul validation")
XTrain_, XVal_, YTrain_, YVal_ = train_test_split(XTrain_transf, YTrain, test_size=0.33,random_state=0)
Linear_fit().fitting(XTrain_, YTrain_, XVal_,YVal_)

print("loss media turibina solo sul test")
Linear_fit().fitting(XTrain_transf, YTrain, XTest_transf,YTest,values_TM)