from ExtractDataset import Enel_dataset
import numpy as np

folder_train = "ENEL_2014/PSC/0-23_0001-0049/"
folder_test = "ENEL_2014/PSC/24-47_0001-0049/"
label_file = "ENEL_2014/PSC/Metering_2011-2014_UTC.txt"
file_name_power_curve =  "ENEL_2014/PSC_Pot_curves.txt"

### save the dataset
enel_dataset = Enel_dataset(folder_train, folder_test, label_file,centerdata=False)
dict_ = enel_dataset.dict_
Coord, Coord_turb = enel_dataset.extract_coordinates(folder="ENEL_2014/PSC/")
power_curve = enel_dataset.extract_power_curve(file_name_power_curve)
XTrain, YTrain, XTest, YTest =  enel_dataset.get_data()

np.savez("ENEL_2014/Enel_dataset.npz", XTrain = XTrain, YTrain = YTrain,XTest = XTest,YTest = YTest, Coord = Coord,Coord_turb = Coord_turb, power_curve=power_curve, dict_ = dict_)
