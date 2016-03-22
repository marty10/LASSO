import numpy as np
import sys
from Enel_utils import find_nearest_turbine, compute_angle
from ExtractResult import Result
from Transformation import Enel_powerCurveTransformation, EnelWindSpeedTransformation, \
    Enel_directionPowerCurveTransformation

sys.argv[1:2] = [str(x) for x in sys.argv[1:2]]
filename = sys.argv[1]
threshold_dir = (int)(sys.argv[2])

results = Result(filename, "lasso")

XTrain, YTrain, XTest, YTest = results.extract_train_test()
weights_list = results.extract_weights()
mses = results.extract_mses()
XTrain_transf, XTest_transf = results.extract_data_transf()
XTrain_, YTrain_, XVal_, YVal_ = results.extract_train_val()
saved_indexes_list = results.get_saved_indexes()
XTrain_ValNoCenter, YTrainVal_noCenter, XVal_noCenter,YVal_noCenter = results.extract_train_val_no_centered()

enel_dict = results.extract_dict()

file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")


##transformation of data
XTrain, YTrain, XTest, YTest = results.extract_train_test()
enel_dict = results.extract_dict()
Coord, Coord_turb, power_curve = results.extract_coords()

angles_coord_turb,_ = compute_angle(Coord, Coord_turb)
output_dict = dict.fromkeys(np.arange(0,49),np.array([[]], dtype = "int64"))

k_levels = np.arange(0,12).reshape([12,1])
for key in np.arange(0,49):
    current_values = np.arange(key*12,key*12+12).reshape([12,1])
    output_dict[key] = np.concatenate((current_values,k_levels), axis = 1)

X = np.concatenate((XTrain, XTest), axis = 0)
enel_transf = Enel_powerCurveTransformation()
X_angle,_,_ = enel_transf.compute_angle_matrix(X)

X_speed,_,_ = EnelWindSpeedTransformation().transform(X)
print("wind speed computed")

enel_transf = Enel_directionPowerCurveTransformation()
X_transf, dict_sample_turb = enel_transf.transform(X_angle, angles_coord_turb, X_speed, power_curve, Coord, Coord_turb, threshold_dir=threshold_dir)
print("single transformation done")

_, output_dict = enel_transf.transformPerTurbineLevel(dict_sample_turb, enel_dict, X, power_curve, X_transf,output_dict)
print("transformation per turbine done")



np.savez(filename, dict_ = output_dict, saved_indexes_list = saved_indexes_list,
            mses = mses, weights_list = weights_list, XTrain = XTrain, XTest = XTest, YTest = YTest,
            YTrain = YTrain, XTrainTransf_ = XTrain_transf, XTestTransf_ = XTest_transf, XTrain_ValNoCenter = XTrain_ValNoCenter,
           XValTransf_noCenter = XVal_noCenter, YTrainVal_noCenter = YTrainVal_noCenter, YVal_noCenter = YVal_noCenter,
             XTrain_Val = XTrain_, XVal = XVal_ , YVal_ = YVal_, YTrain_Val = YTrain_ )