import numpy as np
from Enel_utils import find_turbines_nearest_points
from ExtractResult import Result
from Transformation import Enel_powerCurveTransformation
import sys
from pprint import pprint

sys.argv[1:2] = [str(x) for x in sys.argv[1:2]]
filename = sys.argv[1]
until_k = (int)(sys.argv[2])
k = (int)(sys.argv[3])

results = Result(filename, "lasso")

XTrain, YTrain, XTest, YTest = results.extract_train_test()
print(XTrain.shape[1])
weights_list = results.extract_weights()
mses = results.extract_mses()
XTrain_transf, XTest_transf = results.extract_data_transf()
print(XTrain_transf.shape[1])
XTrain_, YTrain_, XVal_, YVal_ = results.extract_train_val()
saved_indexes_list = results.get_saved_indexes()
XTrain_ValNoCenter, YTrainVal_noCenter, XVal_noCenter,YVal_noCenter = results.extract_train_val_no_centered()



file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")

enel_dict = results.extract_dict()
Coord, Coord_turb, power_curve = results.extract_coords()

turbine_dict = find_turbines_nearest_points(Coord,Coord_turb,k=k)
enel_transf = Enel_powerCurveTransformation()
d, output_dict = enel_transf.transform(turbine_dict, enel_dict, XTrain, power_curve,k,sum_until_k=until_k)
print(d.shape[1])
pprint(output_dict)

np.savez(filename, dict_ = output_dict, saved_indexes_list = saved_indexes_list,
            mses = mses, weights_list = weights_list, XTrain = XTrain, XTest = XTest, YTest = YTest,
            YTrain = YTrain, XTrainTransf_ = XTrain_transf, XTestTransf_ = XTest_transf, XTrain_ValNoCenter = XTrain_ValNoCenter,
           XValTransf_noCenter = XVal_noCenter, YTrainVal_noCenter = YTrainVal_noCenter, YVal_noCenter = YVal_noCenter,
             XTrain_Val = XTrain_, XVal = XVal_ , YVal_ = YVal_, YTrain_Val = YTrain_ )