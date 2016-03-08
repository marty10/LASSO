import abc
from abc import ABCMeta

from sklearn.linear_model.base import center_data
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from Kernel import Scalar_kernel, Gaussian_kernel
from Lasso_utils import compute_lasso
from utility import center_test



class Fit:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def fitting(self, X, Y, XTest, YTest):
        """Fit data"""

class Enel_gaussianKernel(Fit):
    def __init__(self,):
        pass

    def fitting(self,XTrain, YTrain, XTest,YTest):

        XTrain_transf = XTrain
        XTest_transf = XTest

        YTest_transf = Gaussian_kernel(sigma = 0.5).transform_y(XTest_transf,YTest)
        YTrain_transf = Gaussian_kernel(sigma = 0.5).transform_y(XTrain_transf,YTrain)

        XTrain_transf= Gaussian_kernel(sigma = 0.5).transform(XTrain_transf)
        XTest_transf= Gaussian_kernel(sigma = 0.5).transform(XTest_transf)


        ##centratura dei dati
        XTrain_transf, YTrain_, X_mean, y_mean, X_std = center_data(XTrain_transf, YTrain_transf, fit_intercept=True, normalize = True)
        XTest_transf, YTest_ = center_test(XTest_transf,YTest_transf,X_mean,y_mean,X_std)

        new_loss, _ = compute_lasso(XTrain_transf, YTrain_, XTest_transf, YTest_, score ="r2_score")

        print("loss enel velocita :", new_loss )
        return XTrain_transf, XTest_transf

class Enel_scalarKernel(Fit):
    def __init__(self,):
        pass

    def fitting(self,XTrain, YTrain, XTest,YTest):

        XTrain_transf = XTrain
        XTest_transf = XTest

        YTest_transf = Scalar_kernel().transform_y(XTest_transf,YTest)
        YTrain_transf = Scalar_kernel().transform_y(XTrain_transf,YTrain)

        XTrain_transf, dict_ = Scalar_kernel().transform(XTrain_transf)
        XTest_transf, dict_ = Scalar_kernel().transform(XTest_transf)


        ##centratura dei dati
        XTrain_transf, YTrain_, X_mean, y_mean, X_std = center_data(XTrain_transf, YTrain_transf, fit_intercept=True, normalize = True)
        XTest_transf, YTest_ = center_test(XTest_transf,YTest_transf,X_mean,y_mean,X_std)

        new_loss, _ = compute_lasso(XTrain_transf, YTrain_, XTest_transf, YTest_, score ="r2_score")

        print("loss enel velocita :", new_loss )
        return XTrain_transf, XTest_transf, dict_

# class Enel_power3(Fit):
#     def __init__(self,):
#         pass
#
#     def fitting(self,XTrain, YTrain, XTest,YTest):
#
#         transf = EnelTransformation()
#         XTrain_transf,_ = transf.transform(XTrain)
#         XTest_transf,_ = transf.transform(XTest)
#
#         ##centratura dei dati
#         XTrain_transf, YTrain_, X_mean, y_mean, X_std = center_data(XTrain_transf, YTrain, fit_intercept=True, normalize = True)
#         XTest_transf, YTest_ = center_test(XTest_transf,YTest,X_mean,y_mean,X_std, normalize=True)
#
#         new_loss, _ = compute_lasso(XTrain_transf, YTrain_, XTest_transf, YTest_, score ="r2_score")
#
#         print("loss enel velocita :", new_loss )
#         return XTrain_transf, XTest_transf


class Linear_fit(Fit):
    def __init__(self,):
        pass

    def fitting(self, XTrain, YTrain, XTest,YTest,values_TM ):
        ##center data
        XTrain_, YTrain_, X_mean, y_mean, X_std = center_data(XTrain, YTrain, fit_intercept=True, normalize = True)
        XTest_, YTest_ = center_test(XTest,YTest,X_mean,y_mean,X_std, normalize=True)

        ##compute linear lasso
        new_loss, beta = compute_lasso(XTrain_, YTrain_, XTest_, YTest_,score = "r2_score", values_TM = values_TM)
        print("loss lineare", new_loss)



class Power_fit(Fit):
    def __init__(self,):
        pass

    def fitting(self,XTrain, YTrain, XTest,YTest ):

        ###trasformazione non lineare
        for degree in range(2,5):
            poly = PolynomialFeatures(degree=degree,include_bias=False)
            XTrain_transf = np.power(XTrain,degree)
            XTest_transf = np.power(XTest,degree)

            ##centratura dei dati
            XTrain_transf, YTrain_, X_mean, y_mean, X_std = center_data(XTrain_transf, YTrain, fit_intercept=True, normalize = True)
            XTest_transf, YTest_ = center_test(XTest_transf,YTest,X_mean,y_mean,X_std)

            new_loss, _ =compute_lasso(XTrain_transf, YTrain_, XTest_transf, YTest_,score = "r2_score")

            print("loss polinomio grado", str(degree),":", new_loss )


class Polynomial_fit(Fit):
    def __init__(self,):
        pass

    def fitting(self,XTrain, YTrain, XTest,YTest ):

        ###trasformazione non lineare
        for degree in range(2,3):
            poly = PolynomialFeatures(degree=degree,include_bias=False)
            XTrain_transf = poly.fit_transform(XTrain)
            XTest_transf = poly.fit_transform(XTest)

            ##centratura dei dati
            XTrain_transf, YTrain_, X_mean, y_mean, X_std = center_data(XTrain_transf, YTrain, fit_intercept=True, normalize = True)
            XTest_transf, YTest_ = center_test(XTest_transf,YTest,X_mean,y_mean,X_std)

            new_loss, _ =compute_lasso(XTrain_transf, YTrain_, XTest_transf, YTest_,score = "r2_score")

            print("loss polinomio grado", str(degree),":", new_loss )

class Log_y_fit(Fit):
    def __init__(self,):
        pass

    def fitting(self,XTrain, YTrain, XTest,YTest):

        YTrain_ = np.log(YTrain)
        if np.isnan(YTrain_).any():
            print("log y nan")
            return
        YTest_ = np.log(YTest).any()
        if np.isnan(YTest_):
            print("log y nan")
            return
        XTrain_, YTrain_, X_mean, y_mean, X_std = center_data(XTrain, YTrain_, fit_intercept=True, normalize = True)
        XTest_, YTest_ = center_test(XTest,YTest_,X_mean,y_mean,X_std)

        new_loss, _ = compute_lasso(XTrain_, YTrain_, XTest_, YTest_,score = "r2_score")

        print("loss log(y) :", new_loss )


class Inverse_y_fit(Fit):
    def __init__(self,):
        pass

    def fitting(self,XTrain, YTrain, XTest, YTest):

        YTrain_ = 1./np.array(YTrain)
        if np.isnan(YTrain_).any():
            print("inverse nan")
            return
        YTest_ = 1./np.array(YTest)
        if np.isnan(YTest_).any():
            print("inverse nan")
            return

        XTrain_, YTrain_, X_mean, y_mean, X_std = center_data(XTrain, YTrain_, fit_intercept=True, normalize = True)
        XTest_, YTest_ = center_test(XTest,YTest_,X_mean,y_mean,X_std)

        new_loss, _ =compute_lasso(XTrain_, YTrain_, XTest_, YTest_,score = "r2_score")

        print("loss inverse :", new_loss )

class Inverse_x_fit(Fit):
    def __init__(self,):
        pass

    def fitting(self,XTrain, YTrain, XTest,YTest):

        XTrain_transf = 1./XTrain
        if np.isnan(XTrain_transf).any():
            print("inverse x nan")
            return
        XTest_transf = 1./np.array(XTest)
        if np.isnan(XTest_transf).any():
            print("inverse x nan")
            return

        XTrain_, YTrain_, X_mean, y_mean, X_std = center_data(XTrain_transf, YTrain, fit_intercept=True, normalize = True)
        XTest_, YTest_ = center_test(XTest_transf,YTest,X_mean,y_mean,X_std)

        new_loss, _ =compute_lasso(XTrain_, YTrain_, XTest_, YTest_,score = "r2_score")

        print("loss inverse x:", new_loss )

class Sqrt_fit(Fit):
    def __init__(self,):
        pass

    def fitting(self,XTrain, YTrain, XTest,YTest):
        YTrain_ = np.sqrt(YTrain)
        if np.isnan(YTrain_).any():
            print("sqrt nan")
            return
        YTest_ = np.sqrt(YTest)
        if np.isnan(YTest_).any():
            print("sqrt nan")
            return

        XTrain_, YTrain_, X_mean, y_mean, X_std = center_data(XTrain, YTrain_, fit_intercept=True, normalize = True)
        XTest_, YTest_ = center_test(XTest,YTest_,X_mean,y_mean,X_std)

        new_loss, _ = compute_lasso(XTrain_, YTrain_, XTest_, YTest_,score = "r2_score")

        print("loss sqrt(y) :", new_loss )

class Log_x_fit(Fit):
    def __init__(self,):
        pass

    def fitting(self, XTrain, YTrain, XTest,YTest):

        XTrain_transf = np.log(XTrain)
        if np.isnan(XTrain_transf).any():
            print("log x nan")
            return
        XTest_transf = np.log(XTest)

        if np.isnan(XTest_transf).any():
            print("log x nan")
            return

        ##centratura dei dati
        XTrain_transf, YTrain_, X_mean, y_mean, X_std = center_data(XTrain_transf, YTrain, fit_intercept=True, normalize = True)
        XTest_transf, YTest_ = center_test(XTest_transf,YTest,X_mean,y_mean,X_std)

        new_loss, beta_ordered_values = compute_lasso(XTrain_transf, YTrain_, XTest_transf, YTest_,score = "r2_score")

        print("loss log(x) :", new_loss )

class Log_xy_fit(Fit):
    def __init__(self,):
        pass

    def fitting(self,XTrain, YTrain, XTest,YTest):

        YTrain_ = np.log(YTrain)
        if np.isnan(YTrain_).any():
            print("log y nan")
            return
        YTest_ = np.log(YTest)
        if np.isnan(YTest_).any():
            print("log y nan")
            return

        XTrain_transf = np.log(XTrain)
        if np.isnan(XTrain_transf):
            print("log x nan")
            return
        XTest_transf = np.log(XTest)
        if np.isnan(XTest_transf):
            print("log x nan")
            return

        ##centratura dei dati
        XTrain_transf, YTrain_, X_mean, y_mean, X_std = center_data(XTrain_transf, YTrain_, fit_intercept=True, normalize = True)
        XTest_transf, YTest_ = center_test(XTest_transf,YTest_,X_mean,y_mean,X_std)

        new_loss,_ = compute_lasso(XTrain_transf, YTrain_, XTest_transf, YTest_, score = "r2_score")

        print("loss log(y) e log(x) :", new_loss )