import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from dppy.finite_dpps import FiniteDPP
import math

# Function for generating data such that x_i is uniformly in [-1,1]^d
def generate_data_uniform(N,d):
    X = 2 * np.random.rand(d * N).reshape((N, d)) - 1
    theta_gen = np.random.normal(0, 1, size=(d, 1))
    tmp = np.dot(X,theta_gen)
    std_e = np.std(tmp)/3
    y = np.dot(X,theta_gen) + std_e*np.random.normal(0, 1, size=(N, 1))
    y = y + np.random.normal(0.1, 0.5, size=(N, 1))
    y = np.minimum(np.maximum(y,-0.95),0.95)
    return X, y

# Function for generating data such that x_i follows beta distribution in [-1,1]^d
def generate_data_beta(N,d):
    a = np.random.uniform(0.6, 1.4, size=(d, 1))
    b = np.random.uniform(0.6, 1.4, size=(d, 1))
    print('True beta distribution parameter is\n', np.column_stack((b - 1, a - 1)))
    X = (2.0 * np.random.beta(a, b, size=(d,N))) - 1.0
    X = X.T    
    theta_gen = np.random.normal(0, 1, size=(d, 1))
    tmp = np.dot(X,theta_gen)
    std_e = np.std(tmp)/3
    y = np.dot(X,theta_gen) + std_e*np.random.normal(0, 1, size=(N, 1))
    return X, y

# Function for generating data such that x_i follows a mixture of Gaussian distributions
def generate_data_mixture_Gaussian(N,d,m):
    mean_total = np.zeros((m,d))
    for i in range(m):
        mean_total[i,:] = np.random.uniform(-0.8, 0.8, size=(1,d))
    cov_total = np.random.uniform(0.55, 0.55, size=(m,))#(0.001, 0.005, size=(m,))
    num =  np.array(list(range(1, m+1)))
    num = num*math.floor(N/np.sum(num))
    num[m-1] = num[m-1] + N - np.sum(num)
    X = np.random.multivariate_normal(mean_total[0,:], cov_total[0]*np.identity(d), num[0])
    for i in range(1,m):
        Xtmp = np.random.multivariate_normal(mean_total[i,:], cov_total[i]*np.identity(d), num[i])
        X = np.concatenate((X, Xtmp))
    X = np.minimum(np.maximum(X,-0.95),0.95)
    #X = X[np.random.permutation(np.array(list(range(N)))),:]
    theta_gen = np.random.normal(0, 1, size=(d, 1))
    tmp = np.dot(X,theta_gen)
    std_e = np.std(tmp)/3
    y = np.dot(X,theta_gen) + std_e*np.random.normal(0, 1, size=(N, 1))
    y = y + np.random.normal(0.1, 0.5, size=(N, 1))
    y = np.minimum(np.maximum(y,-0.95),0.95)
    return X, y