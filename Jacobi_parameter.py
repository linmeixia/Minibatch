import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from dppy.finite_dpps import FiniteDPP

# Function for estimating the Jacobi parameters
def fit_Jacobi_parameters(X):
    mu1 = X.mean(axis=0)
    Xs = X * X
    mu2 = Xs.mean(axis=0)
    tmp = (mu1 + 1) / 2 + (mu2 - 1) / 4 - (mu1 + 1) * (mu1 + 1) / 4
    t = (1 - mu1) / 2 * ((1 + mu1) * (1 - mu1) / 4 / tmp - 1)
    a = t - 1
    b = (1 + mu1) / (1 - mu1) * t - 1
    jac_params = np.column_stack((a,b))
    jac_params = np.minimum(np.maximum(jac_params,-0.5),0.5)
    print('Estimated beta distribution parameter is\n', jac_params)
    return jac_params
