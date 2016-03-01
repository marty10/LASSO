from ExtractDataset import Enel_dataset
import numpy as np

from ExtractResult import Result

file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")

XTrain, YTrain, XTest, YTest = results.extract_train_test()
Coord, Coord_turb, power_curve = results.extract_coords()

dict_ = dict.fromkeys(np.arange(0,49),np.array([]))

for key in (list)(dict_.keys()):
    dict_[key] = np.arange(key*24,key*24+24)

np.savez("ENEL_2014/Enel_dataset.npz", XTrain = XTrain, YTrain = YTrain,XTest = XTest,YTest = YTest, Coord = Coord,Coord_turb = Coord_turb, power_curve=power_curve, dict_ = dict_)
