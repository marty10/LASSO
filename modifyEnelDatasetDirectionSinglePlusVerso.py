import numpy as np
import sys
from Enel_utils import find_nearest_turbine, compute_angle
from ExtractResult import Result
from Transformation import Enel_powerCurveTransformation, EnelWindSpeedTransformation

sys.argv[1:] = [str(x) for x in sys.argv[1:]]
filename = sys.argv[1]
results = Result(filename, "lasso")

XTrain, YTrain, _, _ = results.extract_train_test()
weights_list = results.extract_weights()
mses = results.extract_mses()
XTrain_transf, XTest_transf = results.extract_data_transf()
XTrain_, YTrain_, XVal_, YVal_ = results.extract_train_val()
saved_indexes_list = results.get_saved_indexes()
XTrain_ValNoCenter, YTrainVal_noCenter, XVal_noCenter,YVal_noCenter = results.extract_train_val_no_centered()
output_dict = results.extract_dict()

file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")

_, _, XTest, YTest = results.extract_train_test()
enel_transf = Enel_powerCurveTransformation()

X_speed,_ = EnelWindSpeedTransformation().transform(X_Test)

Coord, Coord_turb, power_curve = results.extract_coords()

angles_coord_turb,verso_turb_point = compute_angle(Coord, Coord_turb)
X_angle,x_verso,_ = enel_transf.compute_angle_matrix(XTest)
enel_transf = Enel_directionVersoPowerCurveTransformation()

for d,dir_turbine in enumerate(np.array([-0.5,0.5,1,-1])):
    X_1, matrix_turbs = enel_transf.transform(X_angle, angles_coord_turb, X_speed, power_curve, Coord, Coord_turb, x_verso, dir_turbine,verso_turb_point)
    if d==0:
        XTest_transf = X_1.copy()
    else:
        XTest_transf = np.concatenate((XTest_transf, X_1), axis = 1)

#output_dict = dict.fromkeys(np.arange(0,49),np.array([[]], dtype = "int64"))
# current_dim =0
# k_levels = np.arange(0,12).reshape([12,1])
# for current_dim in range(4):
#     for key in np.arange(0,49):
#         current_values = np.arange(current_dim*X_speed.shape[1]+key*12,current_dim*X_speed.shape[1]+key*12+12).reshape([12,1])
#         values_plus_key = np.concatenate((current_values,k_levels), axis = 1)
#         if output_dict[key].shape[1]==0:
#             output_dict[key] = values_plus_key
#         else:
#             output_dict[key] = np.concatenate((output_dict[key],values_plus_key), axis = 0)
# print("wind speed computed")


np.savez(filename, dict_ = output_dict, saved_indexes_list = saved_indexes_list,
            mses = mses, weights_list = weights_list, XTrain = XTrain, XTest = XTest, YTest = YTest,
            YTrain = YTrain, XTrainTransf_ = XTrain_transf, XTestTransf_ = XTest_transf, XTrain_ValNoCenter = XTrain_ValNoCenter,
           XValTransf_noCenter = XVal_noCenter, YTrainVal_noCenter = YTrainVal_noCenter, YVal_noCenter = YVal_noCenter,
             XTrain_Val = XTrain_, XVal = XVal_ , YVal_ = YVal_, YTrain_Val = YTrain_ )