from abc import ABCMeta
import abc
import math
from sklearn import linear_model
from sklearn.base import BaseEstimator
import numpy as np
import timeit
import numpy.linalg as li
import scipy
from utility import assign_weights


class LASSOEstimator(BaseEstimator):
    def __init__(self, algorithm, fit_intercept=False):
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

    def fit(self, x, y, verbose=False,**params):
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

        return 0.5 * np.sum((dot(x, self.beta) - y) ** 2.0) #+ self.lambda_lasso * np.sum(abs(self.beta))


class Algorithm:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def fit(self, x, y, **params):
        """fit"""


class Shooting(Algorithm):
    def __init__(self,  weights = None, warm_start = None,  tol=1e-4, max_iter=100000000):
        self.num_iter = 0
        self.t = 0
        self.tol = tol
        self.max_iter = max_iter
        self.weights = weights
        self.warm_start = warm_start

    def fit(self, x, y, model, verbose ,**params):
        if self.weights == None:
            self.weights = np.ones(x.shape[1])
        n_samples = x.shape[0]
        lasso_lambda = model.lambda_lasso*self.weights*n_samples
        assert(len(lasso_lambda) == x.shape[1])
        #print("current lambda", model.lambda_lasso)


        t_1 = timeit.default_timer()
        mse = linear_model.LinearRegression(fit_intercept=False)
        mse.fit(x, y)

        model.beta = mse.coef_
        #model.beta = self.warm_start
        XY = np.dot(x.transpose(), y)
        X2 = np.dot(x.transpose(), x)
        X2B = np.dot(X2, model.beta)
        history = None
        if verbose:
            print(' iter | loss ')
            history = [0.0] * self.max_iter
        for it in range(self.max_iter):
            beta_old = np.copy(model.beta)
            for j in range(len(model.beta)):
                beta_j = model.beta[j]
                x_j2 = X2[j, j]
                s_j = 2*(XY[j] - X2B[j] + x_j2 * beta_j)
                if s_j - lasso_lambda[j] > 0:
                    beta_j = (s_j - lasso_lambda[j]) / 2*x_j2
                elif s_j + lasso_lambda[j] < 0:
                    beta_j = (s_j + lasso_lambda[j]) / 2*x_j2
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
    def __init__(self, rho=0.1, alpha=1.0,
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
            dot_prod = np.dot
            lin_solve = np.linalg.solve
        elif isinstance(X, scipy.sparse.csr_matrix):
            dot_prod = lambda x, y: scipy.sparse.csr_matrix.dot(x, y).ravel()
            lin_solve = lambda X, x: scipy.sparse.linalg.gmres(X, x)[0]
        else:
            raise Exception('Invalid matrix type: dense or csr_matrix should be specified')
        n, p = X.shape

        xTy = dot_prod(X.T, y)
        z = np.zeros(p)
        u = np.zeros(p)

        L, U = self.factor(X, self.rho)
        history = None
        if verbose:
            print(' iter | loss ')
            history = [0.0] * self.max_iter
        for it in range(self.max_iter):
            # x-update
            q = xTy + self.rho * (z - u)  # temporary value

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

            model.beta = z
            if verbose:
                history[it] = model.evaluate_loss(X, y)
                print(' %3d  | %.8f ' % (it + 1, history[it]))
                # anyway
            if sum(abs(z)) == 0 or sum(abs(z_old - z)) / sum(abs(z)) < self.rel_tol:
                break
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

class ShootingModified(Algorithm):
    def __init__(self,  weights = None, warm_start = None,  tol=1e-6, max_iter=100000000):
        self.num_iter = 0
        self.t = 0
        self.tol = tol
        self.max_iter = max_iter
        self.weights = weights
        self.warm_start = warm_start

    def fit(self, x, y, model, verbose ,**params):
        if self.weights == None:
            self.weights = np.ones(x.shape[1])
        n_samples = x.shape[0]
        lasso_lambda = model.lambda_lasso*self.weights*n_samples
        assert(len(lasso_lambda) == x.shape[1])
        print("current lambda", model.lambda_lasso)


        t_1 = timeit.default_timer()
        mse = linear_model.LinearRegression(fit_intercept=False)
        mse.fit(x, y)

        model.beta = mse.coef_
        #model.beta = self.warm_start
        XY = np.dot(x.transpose(), y)
        X2 = np.dot(x.transpose(), x)
        X2B = np.dot(X2, model.beta)

        history = None
        if verbose:
            print(' iter | loss ')
            history = [0.0] * self.max_iter
        for it in range(self.max_iter):
            beta_old = np.copy(model.beta)
            for j in range(len(model.beta)):
                beta_j = model.beta[j]
                x_j2 = X2[j, j]
                s_j = 2*(XY[j] - X2B[j] + x_j2 * beta_j)
                if s_j - lasso_lambda[j] > 0:
                    beta_j = (s_j - lasso_lambda[j]) / 2*x_j2
                elif s_j + lasso_lambda[j] < 0:
                    beta_j = (s_j + lasso_lambda[j]) / 2*x_j2
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

class ShootingEnet(Algorithm):
    def __init__(self,  weights = None, warm_start = None,  alpha = 0.5, tol=1e-4, max_iter=10000):
        self.num_iter = 0
        self.t = 0
        self.tol = tol
        self.max_iter = max_iter
        self.weights = weights
        self.warm_start = warm_start
        self.alpha = alpha

    def fit(self, x, y, model, verbose ,**params):
        if self.weights == None:
            self.weights = np.ones(x.shape[1])
        n_samples = x.shape[0]
        lasso_lambda = model.lambda_lasso*self.weights*n_samples
        assert(len(lasso_lambda) == x.shape[1])
        print("current lambda", model.lambda_lasso)


        t_1 = timeit.default_timer()
        mse = linear_model.LinearRegression(fit_intercept=False)
        mse.fit(x, y)

        model.beta = mse.coef_
        #model.beta = self.warm_start
        XY = np.dot(x.transpose(), y)
        X2 = np.dot(x.transpose(), x)
        X2B = np.dot(X2, model.beta)

        history = None

        if verbose:
            print(' iter | loss ')
            history = [0.0] * self.max_iter
        for it in range(self.max_iter):
            beta_old = np.copy(model.beta)
            for j in range(len(model.beta)):
                beta_j = model.beta[j]
                x_j2 = X2[j, j]
                s_j = 2*(XY[j] - X2B[j] + x_j2 * beta_j)
                if s_j - lasso_lambda[j]*self.alpha > 0:
                    beta_j = (s_j - lasso_lambda[j]) / 2*x_j2+lasso_lambda[j]*(1-self.alpha)
                elif s_j + lasso_lambda[j]*self.alpha < 0:
                    beta_j = (s_j + lasso_lambda[j]) / 2*x_j2+lasso_lambda[j]*(1-self.alpha)
                else:
                    beta_j = 0
                model.beta[j] = beta_j
                X2B += X2[:, j] * (model.beta[j] - beta_old[j])
            if it % 10 == 0 and verbose:
                history[it] = model.evaluate_loss(x, y)
                print(' %3d  | %.8f ' % (it + 1, history[it]))
            if sum(abs(model.beta)) == 0 or sum(abs(beta_old - model.beta)) / sum(abs(model.beta)) < self.tol:
                break

        t_2 = timeit.default_timer() - t_1
        self.t = t_2
        self.num_iter = it


# class LARS(Algorithm):
#     def __init__(self,  weights = None, warm_start = None,  alpha = 0.5, tol=1e-4, max_iter=10000):
#         self.num_iter = 0
#         self.t = 0
#         self.tol = tol
#         self.max_iter = max_iter
#         self.weights = weights
#         self.warm_start = warm_start
#         self.alpha = alpha
#
#     def fit(self, X, y, **params):
#         # n is the number of variables, p is the number of "predictors" or
#         # basis vectors
#
#         # the predictors are assumed to be standardized and y is centered.
#
#         # in the example of the prostate data n would be the number
#         # n = number of data points, p = number of predictors
#         n,p = X.shape
#
#         # mu = regressed version of y sice there are no predictors it is initially the
#         # zero vector
#         mu = np.zeros(n)
#
#         # active set and inactive set - they should invariably be complements
#         act_set = []
#         inact_set = range(p)
#
#         k = 0
#         vs = 0
#         nvs = min(n-1,p)
#
#         beta = np.zeros((2*nvs,p))
#
#         maxiter = nvs * 8
#
#         # current regression coefficients and correlation with residual
#         beta = np.zeros((p+1,p))
#         corr = np.zeros((p+1,p))
#
#          # initial cholesky decomposition of the gram matrix
#         # since the active set is empty this is the empty matrix
#          R = zeros((0,0))
#
#         while vs < nvs and k < maxiter:
#             print "new iteration: vs = ", vs, " nvs = ", nvs, " k = ", k
#             print "mu.shape = ", mu.shape
#             #print "mu = ", mu
#
#             # compute correlation with inactive set
#             # and element that has the maximum correlation
#         # add the variables one at a time
#         for k in xrange(p):
#             print "NEW ITERATION k = ", k, " active_set = ", act_set
#
#             # compute the current correlation
#              c = dot(X.T, y - mu)
#     -        #c = c.reshape(1,len(c))
#     -        jia = argmax(abs(c[inact_set]))
#     -        j = inact_set[jia]
#     -        C = c[j]
#
#     -        print "predictor ", j, " max corr with w/ current residual: ", C
#     -        print "adding ", j, " to active set"
#     -
#     -        print "R shape before insert: ", R.shape
#     -
#     +        print "current correlation = ", c
#     +
#     +        # store the result
#     +        corr[k,:] = c
#     +
#     +        # choose the predictor with the maximum correlation and add it to the active
#     +        # set
#     +        jmax = inact_set[argmax(abs(c[inact_set]))]
#     +        C = c[jmax]
#     +
#     +        print "iteration = ", k, " jmax = ", jmax, " C = ", C
#     +
#              # add the most correlated predictor to the active set
#     -        R = cholinsert(R,X[:,j],X[:,act_set])
#     -        act_set.append(j)
#     -        inact_set.remove(j)
#     -        vs += 1
#     -
#     -        print "R shape after insert ", R.shape
#     +        R = cholinsert(R,X[:,jmax],X[:,act_set])
#     +        act_set.append(jmax)
#     +        inact_set.remove(jmax)
#
#     -        print "active set = ", act_set
#     -        print "inactive set = ", inact_set
#     -
#              # get the signs of the correlations
#              s = sign(c[act_set])
#              s = s.reshape(len(s),1)
#     -        #print "R.shape = ", R.shape
#     -        #print "s.shape = ", s.shape
#     -
#     -        # move in the direction of the least squares solution
#     +        print "sign = ", s
#
#     +        # move in the direction of the least squares solution restricted to the active
#     +        # set
#     +
#              GA1 = solve(R,solve(R.T, s))
#              AA = 1/sqrt(sum(GA1 * s))
#              w = AA * GA1
#     -
#     -        # equiangular direction - this should be a unit vector
#     -        print "X[:,act_set].shape = ",X[:,act_set].shape
#     -        #print "w.shape = ",w.shape
#     -
#     +
#     +        print "AA = ", AA
#     +        print "w = ", w
#     +
#              u = dot(X[:,act_set], w).reshape(-1)
#
#     -        #print "norm of u = ", norm(u)
#     -        #print "u.shape = ", u.shape
#     +        print "norm of u = ", norm(u)
#     +        print "u.shape = ", u.shape
#
#              # if this is the last iteration i.e. all variables are in the
#              # active set, then set the step toward the full least squares
#              # solution
#     -        if vs == nvs:
#     +        if k == p:
#                  print "last variable going all the way to least squares solution"
#                  gamma = C / AA
#              else:
#                  a = dot(X.T,u)
#                  a = a.reshape((len(a),))
#     +
#                  tmp = r_[(C - c[inact_set])/(AA - a[inact_set]),
#                           (C + c[inact_set])/(AA + a[inact_set])]
#     +
#                  gamma = min(r_[tmp[tmp > 0], array([C/AA]).reshape(-1)])
#     +
#     +        print "ITER k = ", k, ", gamma = ", gamma
#
#              mu = mu + gamma * u
#
#              if beta.shape[0] < k:
#                  beta = c_[beta, zeros((beta.shape[0],))]
#              beta[k+1,act_set] = beta[k,act_set] + gamma*w.T.reshape(-1)
#     -
#     -        k += 1
#     +
#     +    return beta, corr