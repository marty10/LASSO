from ExtractResult import Result
from Transformation import EnelWindSpeedTransformation
import numpy as np
import sys
from utility import find_nearest

folder_train = "ENEL_2014/PSC/0-23_0001-0049/"
folder_test = "ENEL_2014/PSC/24-47_0001-0049/"
label_file = "ENEL_2014/PSC/Metering_2011-2014_UTC.txt"

file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")

XTrain, YTrain, XTest, YTest = results.extract_train_test()

Coord = np.load("ENEL_2014/Coord.npz")["Coord"]

sys.argv[1:] = [int(x) for x in sys.argv[1:]]
k = sys.argv[1]

neight_= find_nearest(Coord,k)

enel_transf = EnelWindSpeedTransformation()
XTrain, dict_ = enel_transf.transform(XTrain)
XTrain, output_dict = enel_transf.nearest_products_levels(neight_,dict_,XTrain)

np.savez("ENEL_2014/Product_level_2_dict", dict_ = output_dict)