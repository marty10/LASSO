import timeit
from scipy.spatial.distance import squareform, pdist
from sklearn.externals import joblib
from ExtractDataset import Dataset
from sklearn.metrics.pairwise import pairwise_distances
import numpy.linalg as li
from Transformation import PolinomialTransformation

__author__ = 'Martina'
import numpy as np

ext_model = ".pkl"
n_samples = 2000
n_features = 3000
n_informative = 1000
transformation = PolinomialTransformation(degree = 2)
dataset = Dataset(n_samples, n_features, n_informative = n_informative, transformation = transformation)

X = dataset.XTransf
print(X)
Y = dataset.Y
p = X.shape[1]

y_j = np.atleast_1d(Y)
if np.prod(Y.shape) == len(Y):
    y_j = y_j[:, None]
y_j = np.atleast_2d(y_j)
b = squareform(pdist(y_j))
B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

n = X.shape[0]
dcov2_yy = (B * B).sum() / float(n * n)
sqrt_dcov2_yy = np.sqrt(dcov2_yy)


dcor = np.empty(p)
for j in range(0,p):
    x_j = (X[:,j])
    t_1 = timeit.default_timer()
    x_j = np.atleast_1d(X[:,j])

    if np.prod(x_j.shape) == len(x_j):
        x_j = x_j[:, None]
    x_j = np.atleast_2d(x_j)
    if y_j.shape[0] != x_j.shape[0]:
        raise ValueError('Number of samples must match')
    t_2 = timeit.default_timer()-t_1
    #print("init", t_2)
    #t_4 = timeit.default_timer()
    #a = pairwise_distances(x_j)
    a = squareform(pdist(x_j))
    #t_3 = timeit.default_timer()
    #print("pdist", t_3-t_4)

    A = a - np.mean(a,axis=0) - np.mean(a,axis=1) + np.mean(a)
    #t_6 = timeit.default_timer()
    #print(t_6-t_3)
    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcor[j] = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * sqrt_dcov2_yy)
    #print dcor
    #t_5 = timeit.default_timer()
    #print("prodotto", t_5-t_6)
    #print("ciclo", t_5-t_1)

joblib.dump(dcor, 'distance_cor_poly'+ext_model, compress=9)

