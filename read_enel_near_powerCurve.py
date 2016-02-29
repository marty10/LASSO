from Enel_utils import find_nearest_turbine
from ExtractResult import Result
from Fit import Linear_fit
from Transformation import EnelWindSpeedTransformation
import numpy as np

file_coord = np.load("ENEL_2014/Coord.npz")
Coord = file_coord["Coord"]
Coord_turb = file_coord["Coord_turb"]
power_curve = file_coord["power_curve"]

file = "ENEL_2014/Enel_dataset.npz"
results = Result(file, "lasso")

XTrain, YTrain, XTest, YTest = results.extract_train_test()
enel_transf = EnelWindSpeedTransformation()
XTrain, dict_ = enel_transf.transform(XTrain)
XTest, dict_ = EnelWindSpeedTransformation().transform(XTest)

for k in range(5,10):

    turbine_dict = find_nearest_turbine(Coord,Coord_turb,k)

    XTrain_, output_dict = enel_transf.nearest_mean_turbine(turbine_dict,dict_,XTrain, power_curve)
    print(XTrain_.shape)

    XTest_, _ = enel_transf.nearest_mean_turbine(turbine_dict,dict_,XTest,power_curve)
    print(XTest_.shape)

    Linear_fit().fitting(XTrain_, YTrain, XTest_,YTest)

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