from sklearn.cross_validation import train_test_split

from Enel_utils import find_turbines_nearest_points
from ExtractResult import Result
from Fit import Linear_fit
from Transformation import Enel_powerCurveTransformation
import numpy as np

##load the dataset
file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")

XTrain, YTrain, XTest, YTest = results.extract_train_test()

enel_dict = results.extract_dict()
Coord, Coord_turb, power_curve = results.extract_coords()

### save the dataset
# enel_dataset = Enel_dataset(folder_train, folder_test, label_file,centerdata=False)
# dict_ = enel_dataset.dict_
# Coord, Coord_turb = enel_dataset.extract_coordinates(folder="ENEL_2014/PSC/")
# power_curve = enel_dataset.extract_power_curve(file_name_power_curve)
# XTrain, YTrain, XTest, YTest = enel_dataset.get_data()
# np.savez("Enel_dataset.npz", XTrain = XTrain, YTrain = YTrain,XTest = XTest,YTest = YTest, Coord = Coord,Coord_turb = Coord_turb, power_curve=power_curve, dict_ = dict_)
enel_transf = Enel_powerCurveTransformation()
output_dict_ = dict.fromkeys(np.arange(0,49),np.array([[]], dtype = "int64"))
XTrain_tmp = np.array([[]])
XTest_tmp = np.array([[]])

k_1 = 1
turbine_dict = find_turbines_nearest_points(Coord,Coord_turb,k_1)
XTrain_tmp, output_dict_ = enel_transf.transform(turbine_dict, enel_dict, XTrain, power_curve,0, XTrain_tmp, output_dict_)
#XTest_tmp, _ = enel_transf.transform_nearest_1(turbine_dict, enel_dict, XTest, power_curve,k_1, XTest_tmp, output_dict_.copy())

#Linear_fit().fitting(XTrain_tmp, YTrain, XTest_tmp,YTest)
#print(XTrain_tmp.shape[1])

for k in range(1,50):
    print(k, "vicini")
    turbine_dict = find_turbines_nearest_points(Coord,Coord_turb,k)
    XTrain_tmp, output_dict_ = enel_transf.transform(turbine_dict, enel_dict, XTrain, power_curve,1, XTrain_tmp, output_dict_)
        #XTest_tmp, _ = enel_transf.transform(turbine_dict, enel_dict, XTest, power_curve,k_1, XTest_tmp, output_dict_.copy())
    print(XTrain_tmp.shape)

#XTest_, _ = enel_transf.transform(turbine_dict,enel_dict,XTest,power_curve,k,1)
    XTrain_, XVal_, YTrain_, YVal_ = train_test_split(XTrain_tmp, YTrain, test_size=0.33,random_state=0)

    Linear_fit().fitting(XTrain_, YTrain_, XVal_,YVal_)
