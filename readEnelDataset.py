from sklearn.cross_validation import train_test_split
from Enel_utils import find_nearest_turbine, get_products_points, find_nearest, find_turbines_nearest_points
from ExtractDataset import Enel_dataset
from ExtractResult import Result
from Fit import Linear_fit
from Transformation import EnelWindSpeedTransformation, Enel_powerCurveTransformation
import sys
import numpy as np

folder_train = "ENEL_2014/PSC/0-23_0001-0049/"
folder_test = "ENEL_2014/PSC/24-47_0001-0049/"
label_file = "ENEL_2014/PSC/Metering_2011-2014_UTC.txt"
file_name_power_curve =  "ENEL_2014/PSC_Pot_curves.txt"

##load the dataset
file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")

XTrain, YTrain, XTest, YTest = results.extract_train_test()

enel_dict = results.extract_dict()
Coord, Coord_turb, power_curve = results.extract_coords()

### save the dataset
enel_dataset = Enel_dataset(folder_train, folder_test, label_file,centerdata=False)
# dict_ = enel_dataset.dict_
# Coord, Coord_turb = enel_dataset.extract_coordinates(folder="ENEL_2014/PSC/")
# power_curve = enel_dataset.extract_power_curve(file_name_power_curve)
# XTrain, YTrain, XTest, YTest =  enel_dataset.get_data()
#np.savez("Enel_dataset.npz", XTrain = XTrain, YTrain = YTrain,XTest = XTest,YTest = YTest, Coord = Coord,Coord_turb = Coord_turb, power_curve=power_curve, dict_ = dict_)

for k in range(2,20):

    print(k, "vicini")
    turbine_dict = find_turbines_nearest_points(Coord,Coord_turb,k)

    #neight_ = nearest_mean_turbine(turbine_dict)
    #XTrain, YTrain, XTest, YTest = enel_dataset.get_data()
    #np.savez("Enel_dataset.npz", XTrain = XTrain, XTest = XTest, YTrain = YTrain, YTest = YTest)


    enel_transf = Enel_powerCurveTransformation()
    XTrain_, output_dict = enel_transf.transform(turbine_dict, enel_dict, XTrain, power_curve,k,1)

    print(XTrain_.shape)

    #XTest_, _ = enel_transf.transform(turbine_dict,enel_dict,XTest,power_curve,k)
    XTrain_, XVal_, YTrain_, YVal_ = train_test_split(XTrain_, YTrain, test_size=0.33,random_state=0)
    #YTrain = (YTrain/39)**(1./3)
    #YTest = (YTest/39)**(1./3)
    #XTrain = XTrain**3
    #XTest = XTest**3

    Linear_fit().fitting(XTrain_, YTrain_, XVal_,YVal_)

    #Power_fit().fitting(XTrain, YTrain, XTest,YTest)
    #Polynomial_fit().fitting(XTrain, YTrain, XTest,YTest)

    #YTrain = np.array(YTrain)
    #YTest = np.array(YTest)
    #Log_y_fit().fitting(XTrain, YTrain, XTest,YTest)

    #Inverse_y_fit().fitting(XTrain, YTrain, XTest, YTest)


    #Enel_power3().fitting(XTrain, YTrain, XTest,YTest)
    #Enel_gaussianKernel().fitting(XTrain, YTrain, XTest,YTest)


    #Inverse_x_fit().fitting(XTrain, YTrain, XTest, YTest)

    #Sqrt_fit().fitting(XTrain, YTrain, XTest,YTest)

    #Log_x_fit().fitting(XTrain, YTrain, XTest,YTest)

    #Log_xy_fit.fitting(XTrain, YTrain, XTest,YTest)

    #X = np.array([[1,2,3],[0,1,2]])

    #transf = Gaussian(sigma = 0.5)
    #x_transf = transf.transform(X)