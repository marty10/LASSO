from abc import ABCMeta
import abc
import math
from sklearn import linear_model
from sklearn.base import BaseEstimator
import numpy as np
import timeit
import numpy.linalg as li
from sklearn.linear_model.base import center_data
import scipy
from FeatureSelectionRules import FeatureSelectionRule, HSIC_Criterion
from HSIC import HSIC


class LASSOEstimator(BaseEstimator):
    def __init__(self, algorithm, fit_intercept=True):
        self._estimator_type = "regressor"
        self.beta = 0
        self.lambda_lasso = 0
        self.algorithm = algorithm
        self.fit_intercept = fit_intercept

    def get_params(self, deep):  # solo quelli del costruttore
        return {"algorithm": self.algorithm}

    def set_params(self, **params):
        self.lambda_lasso = params["alpha"]
        return self

    def fit(self, x, y, verbose=True, **params):
        return self.algorithm.fit(x, y, self, verbose, **params)

    def predict(self, x):
        return np.dot(x, self.beta)

    def evaluate_loss(self, x, y):

        """cost function of L1 least square
        f(x) = (1/2) || Ax - b ||^2 + lambda || x ||_1"""

        if isinstance(x, np.ndarray):
            dot = np.dot
        elif isinstance(x, scipy.sparse.csr_matrix):
            dot = scipy.sparse.csr_matrix.dot
        else:
            raise TypeError('The matrix A must be numpy.ndarray or scipy.sparse.csr_matrix')

        return 0.5 * np.sum((dot(x, self.beta) - y) ** 2.0) + self.lambda_lasso * np.sum(abs(self.beta))


class Algorithm:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def fit(self, x, y, **params):
        """fit"""


class Shooting(Algorithm):
    def __init__(self, tol=1e-4, max_iter=100000000):
        self.num_iter = 0
        self.t = 0
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, x, y, model, verbose, **params):
        lasso_lambda = model.lambda_lasso
        t_1 = timeit.default_timer()
        mse = linear_model.LinearRegression(fit_intercept=False)
        mse.fit(x, y)
        model.beta = mse.coef_
        XY = np.dot(x.transpose(), y)
        X2 = np.dot(x.transpose(), x)
        X2B = np.dot(X2, model.beta)

        history = None
        if verbose:
            print(' iter | loss ')
            history = [0.0] * self.max_iter
        for it in xrange(self.max_iter):
            beta_old = np.copy(model.beta)
            for j in range(len(model.beta)):
                beta_j = model.beta[j]
                # if beta_j != 0:  #####126000 vs 138
                x_j2 = X2[j, j]
                s_j = XY[j] - X2B[j] + x_j2 * beta_j
                if s_j - lasso_lambda > 0:
                    beta_j = (s_j - lasso_lambda) / x_j2
                elif s_j + lasso_lambda < 0:
                    beta_j = (s_j + lasso_lambda) / x_j2
                else:
                    beta_j = 0
                model.beta[j] = beta_j
                X2B += X2[:, j] * (model.beta[j] - beta_old[j])
            if it % 100 == 0 and verbose:
                history[it] = model.evaluate_loss(x, y)
                print(' %3d  | %.8f ' % (it + 1, history[it]))
            if sum(abs(model.beta)) == 0 or sum(abs(beta_old - model.beta)) / sum(abs(model.beta)) < self.tol:
                break

        t_2 = timeit.default_timer() - t_1
        self.t = t_2
        self.num_iter = it


class ADMM(Algorithm):
    def __init__(self, rho=2.0, alpha=1.0,
                 max_iter=100000000, abs_tol=1e-6, rel_tol=1e-4):
        self.rho = rho  # step size
        self.alpha = alpha  # over relaxation parameter
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.max_iter = max_iter

    def fit(self, X, y, model, verbose, **params):
        global dot_prod, lin_solve, vec_norm
        if isinstance(X, np.ndarray):
            mat_prod = np.dot
            dot_prod = np.dot
            lin_solve = np.linalg.solve
        elif isinstance(X, scipy.sparse.csr_matrix):
            mat_prod = lambda X, x: scipy.sparse.csr_matrix.dot(X, x)
            dot_prod = lambda x, y: scipy.sparse.csr_matrix.dot(x, y).ravel()
            lin_solve = lambda X, x: scipy.sparse.linalg.gmres(X, x)[0]
        else:
            raise Exception('Invalid matrix type: dense or csr_matrix should be specified')
        n, p = X.shape

        xTy = dot_prod(X.T, y)
        x = np.zeros(p)
        z = np.zeros(p)
        u = np.zeros(p)

        L, U = self.factor(X, self.rho)
        history = None
        if verbose:
            print(' iter | loss ')
            history = [0.0] * self.max_iter
        for it in xrange(self.max_iter):
            # x-update
            q = xTy + self.rho * z - u  # temporary value

            if n >= p:  # if skinny
                x = lin_solve(U, lin_solve(L, q))
            else:  # if fat
                Xq = dot_prod(X, q)
                tmp = lin_solve(U, lin_solve(L, Xq))
                x = (q / self.rho) - (np.dot(X.T, tmp) / (self.rho ** 2))

            # z-update with relaxation
            z_old = z.copy()
            x_hat = self.alpha * x + (1 - self.alpha) * z_old
            value = x_hat + u
            z = np.maximum(0.0, value - model.lambda_lasso / self.rho) - np.maximum(0.0,
                                                                                    -value - model.lambda_lasso / self.rho)
            # u-update
            u = u + (x_hat - z)

            # Stopping
            # r_norm = la.norm(x - z)
            # s_norm = la.norm(-self.rho * (z - z_old))

            # eps_pri = np.sqrt(d) * self.abs_tol + self.rel_tol * max(li.norm(x), li.norm(-z))
            # eps_dual = np.sqrt(d) * self.abs_tol + self.rel_tol * li.norm(self.rho * u)

            # if (r_norm < eps_pri) and (s_norm < eps_dual):
            if sum(abs(z)) == 0 or sum(abs(z_old - z)) / sum(abs(z)) < self.rel_tol:
                break
            model.beta = z
            if verbose:
                history[it] = model.evaluate_loss(X, y)
                print(' %3d  | %.8f ' % (it + 1, history[it]))
                # anyway
        model.beta = z

    def factor(self, X, rho):
        n, p = X.shape
        if n >= p:
            L = li.cholesky(np.dot(X.T, X) + rho * np.eye(p))
        else:
            L = li.cholesky(np.eye(n) + 1.0 / rho * np.dot(X, X.T))

        return L, L.T  # L, U


class ISTA(Algorithm):
    def __init__(self, tol=1e-4, rho=0.1, max_iter=100000000):
        self.tol = tol
        self.rho = rho
        self.max_iter = max_iter

    def fit(self, X, y, model, verbose, **params):

        self.rho = 1. / (np.linalg.norm(X, 2) ** 2)
        XT = X.transpose()
        XTY = np.dot(XT, y)
        XTX = np.dot(XT, X)
        m, p = X.shape
        beta_old = np.zeros(p)
        model.beta = np.zeros(p)
        history = None
        if verbose:
            print(' iter | loss ')
            history = [0.0] * self.max_iter
        for it in range(self.max_iter):
            inner = beta_old - self.rho * (np.dot(XTX, beta_old) - XTY)
            model.beta = np.maximum(0.0, inner - model.lambda_lasso * self.rho) - np.maximum(0.0,
                                                                                             -inner - model.lambda_lasso * self.rho)

            if sum(abs(model.beta)) == 0 or sum(abs(model.beta - beta_old)) / sum(abs(model.beta)) < self.tol:
                break
            beta_old = model.beta.copy()
            if verbose:
                history[it] = model.evaluate_loss(X, y)
                print(' %3d  | %.8f ' % (it + 1, history[it]))


class FISTA(Algorithm):
    def __init__(self, tol=1e-4, rho=0.1, max_iter=100000000):
        self.tol = tol
        self.rho = rho
        self.max_iter = max_iter

    def fit(self, X, y, model, verbose=True, **params):
        m, p = X.shape
        t_old = 1.0
        beta_old = np.zeros(p)
        model.beta = np.zeros(p)
        inner = np.zeros(p)
        self.rho = 1. / (np.linalg.norm(X, 2) ** 2)
        XT = X.transpose()
        XTY = np.dot(XT, y)
        XTX = np.dot(XT, X)

        history = None
        if verbose:
            print(' iter | loss ')
            history = [0.0] * self.max_iter

        for it in range(self.max_iter):
            inner = inner - self.rho * (np.dot(XTX, beta_old) - XTY)
            model.beta = self.shrinkage(inner, model.lambda_lasso * self.rho)
            t = (1.0 + math.sqrt(1.0 + 4.0 * t_old * t_old)) / 2.0
            inner = model.beta + ((t_old - 1.0) / t) * (model.beta - beta_old)
            t_old = t
            if verbose:
                history[it] = model.evaluate_loss(X, y)
                print(' %3d  | %.8f ' % (it + 1, history[it]))
            if sum(abs(model.beta)) == 0 or sum(abs(model.beta - beta_old)) / sum(abs(model.beta)) < self.tol:
                break
            beta_old = model.beta.copy()
            # v = np.dot(XT, np.dot(X, beta_old))/max(li.norm(np.dot(XT, np.dot(X, beta_old))))
        return history

    def shrinkage(self, x, thr):
        '''
        soft shrinkage operator
        @param[out] y - numpy.ndarray
        @param[in] x - numpy.ndarray
        @param[in] thr - threshold
        '''
        return np.maximum(0.0, x - thr) - np.maximum(0.0, -x - thr)


class modifiedShooting(Algorithm):
    def __init__(self, tol=1e-4, max_iter=10000000):
        self.num_iter = 0
        self.t = 0
        self.tol = tol
        self.max_iter = max_iter
        #self.lambda_max = lambda_max

    def fit(self, x, y, model, verbose, **params):
        print("START")
        lasso_lambda = model.lambda_lasso

        mse = linear_model.LinearRegression(fit_intercept=False)
        mse.fit(x, y)
        model.beta = mse.coef_

        t_1 = timeit.default_timer()
        hsic = HSIC()
        HSIC_KK, HSIC_KL = hsic.HSIC_Measures(x,y)


        print("measure computed", timeit.default_timer()-t_1)

        active_set = set()
        t_3 = timeit.default_timer()
        feat_select = HSIC_Criterion(active_set)
        j_max= feat_select.apply_wrapper_rule(HSIC_KK, HSIC_KL, model.beta)

        active_set.add(j_max)
        t_2 = timeit.default_timer()
        print("tempo HSIC criterion", t_2-t_3)

        XY = np.dot(x.transpose(), y)
        X2 = np.dot(x.transpose(), x)
        X2B = np.dot(X2, model.beta)
        XBeta = np.dot(x, model.beta)


        history = None
        if verbose:
            print(' iter | loss ')
            history = [0.0] * self.max_iter

        for it in xrange(self.max_iter):
            beta_old = np.copy(model.beta)
            for j in active_set:
                beta_j = model.beta[j]
                if beta_j != 0:  #####126000 vs 138

                    x_j2 = X2[j, j]
                    s_j = XY[j] - X2B[j] + x_j2 * beta_j
                    if s_j - lasso_lambda > 0:
                        beta_j = (s_j - lasso_lambda) / x_j2
                    elif s_j + lasso_lambda < 0:
                        beta_j = (s_j + lasso_lambda) / x_j2
                    else:
                        beta_j = 0
                    model.beta[j] = beta_j
                    X2B += X2[:, j] * (model.beta[j] - beta_old[j])
            #if it % 1 == 0 and verbose:
            if verbose:
                history[it] = model.evaluate_loss(x, y)
                print(' %3d  | %.8f ' % (it + 1, history[it]))
            #if sum(abs(model.beta)) == 0 or sum(abs(beta_old - model.beta)) / sum(abs(model.beta)) < self.tol:
                #break
            if len(active_set)==x.shape[1]:
                active_set.clear()
                print("clear fatto")

            j_max = feat_select.apply_wrapper_rule(HSIC_KK, HSIC_KL, model.beta)
            active_set.add(j_max)
            if len(active_set)==100:
                n_informative_zero = [a for a in active_set if a<=100]
                print("numero_informative", len(n_informative_zero))

        t_2 = timeit.default_timer() - t_1
        self.t = t_2
        self.num_iter = it

    def dpp_rule(self, x, y, model, X2B, X2, beta_old):
        p = x.shape[1]
        lambda_lasso_max = np.empty([p])
        XBeta = np.dot(x, model.beta)
        for j in range(p):
            lambda_lasso_max[j] = abs(np.dot(x[:, j].transpose(), y - XBeta))
            print("lambda_lasso_max",lambda_lasso_max[j])
            if lambda_lasso_max[j] < self.lambda_max - li.norm(x[:,j]) * li.norm(y) * (
                self.lambda_max - model.lambda_lasso) / model.lambda_lasso:
                model.beta[j] = 0
                print("",self.lambda_max - li.norm(x[:,j]) * li.norm(y) * (
                self.lambda_max - model.lambda_lasso) / model.lambda_lasso)
                X2B += X2[:, j] * (model.beta[j] - beta_old[j])

    # def estimate_lambda_max(self,x,y,XBeta):
    #      p = x.shape[1]
    #      for j in range(p):
    #          """"""
            #lambda_lasso_max[j] = abs(np.dot(x[:, j].transpose(), y - XBeta))


