import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from dppy.finite_dpps import FiniteDPP

# Function for generating data
def generate_data(N,m):
    a = np.random.uniform(0.6, 1.4, size=(m, 1))
    b = np.random.uniform(0.6, 1.4, size=(m, 1))
    print('True beta distribution parameter is\n', np.column_stack((b - 1, a - 1)))
    X = (2.0 * np.random.beta(a, b, size=(m,N))) - 1.0
    X = X.T
    # X = np.random.uniform(-1, 1, size=(N, m))
    theta_gen = np.random.normal(0, 1, size=(m, 1))
    y = np.sign(np.dot(X, theta_gen))
    tmp = np.argwhere(y == 0)
    if tmp.size > 0:
        y[tmp] = 1
    return X, y

# Function for dividing the data into batches
def get_batches(X,y,batch_size,row,sampleops,i):
    if sampleops.name == 'iid':
        idx = random.sample(range(row), batch_size)
    if sampleops.name == 'dpp':
        idx = sampleops.DPP_list[i]
    idx = np.sort(idx)
    X_new = X[idx,:]
    y_new = y[idx]
    return X_new, y_new, idx

# Function for computing the gradient
def get_gradient(X,y,theta,weight,loss_type,elambda):
    if loss_type == 'linear_regression':
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y
        gradient = np.dot(X.T, weight * loss) + elambda * theta
    elif loss_type == 'logistic_regression':
        hypothesis = np.dot(X, theta)
        loss = - y * (1 - 1 / (1 + np.exp(- hypothesis * y)))
        gradient = np.dot(X.T, weight * loss) + elambda * theta
    return gradient

# Function for computing the function value
def get_fun_value(X,y,theta,row,loss_type,elambda):
    if loss_type == 'linear_regression':
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y
        fun_value = 0.5 * np.dot(loss.T,loss) / row + 0.5 * elambda * np.dot(theta.T,theta)
    elif loss_type == 'logistic_regression':
        hypothesis = np.dot(X, theta)
        fun_value = np.sum(np.log(1 + np.exp(-hypothesis * y))) / row + 0.5 * elambda * np.dot(theta.T,theta)
    return fun_value

# Function for estimating the Jacobi parameters
def generate_Jacobi_parameters(X):
    ## Method 1, to satisfy the first and second moment of the empirical distribution of X
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
    ## Method 2, randomy generate
    #jac_params = 0.5 - np.random.rand(m, 2)
    return jac_params

# Function for generating the DPP kernel
def generate_DPP_kernel(X,N,p):
    jac_params = generate_Jacobi_parameters(X)  # Jacobi ensemble parameters
    dpp = MultivariateJacobiOPE(N, jac_params)
    Kq = dpp.K(X)
    qX = dpp.eval_w(X)
    tmp = np.sqrt(np.diag(qX))
    Kq = np.dot(np.dot(tmp,Kq),tmp) / N

    ## check the projection
    evals_large_sparse, evecs_large_sparse = largest_eigsh(Kq, p, which='LM')
    print(evals_large_sparse)
    ####

    gammatilde = stats.gaussian_kde(X.T)
    gammatilde.set_bandwidth(bw_method='silverman')
    gammatildeX = gammatilde.evaluate(X.T)
    tmp = np.sqrt(np.asmatrix(qX/gammatildeX))
    mK = np.dot(tmp.T,tmp)
    Ktilde = Kq*mK
    Ktilde = Ktilde / N
    evals_large_sparse, evecs_large_sparse = largest_eigsh(Ktilde, p, which='LM')
    evals_large_sparse = np.ones(p)
    Ktilde = np.dot(evecs_large_sparse,evecs_large_sparse.T)
    return evals_large_sparse, evecs_large_sparse, Ktilde

# Function for sampling the DPP
def generate_DPP_list_of_samples(eig_vals, eig_vecs, maxit):
    DPP = FiniteDPP(kernel_type='correlation',projection=True,
                    **{'K_eig_dec': (eig_vals, eig_vecs)})
    rng = np.random.RandomState(0)
    for _ in range(maxit):
        DPP.sample_exact(mode='GS', random_state=rng)
    return DPP.list_of_samples

# Mini-Batch SGD
def MiniBatchSGD(X,y,theta,loss_type,elambda,batch_size,maxiter,sampleops,thetastar=np.nan):
    row, col = X.shape
    if sampleops.name == 'iid':
        weight = (1 / batch_size) * np.ones((row, 1))
    if sampleops.name == 'dpp':
        weight = 1 / sampleops.kernel_diag
        weight = weight.reshape((row,1))
    draw_figure = 1
    if draw_figure:
        loss_total = np.array(maxiter*[[0.0]])
        if ~np.isnan(thetastar).all():
            error = np.array(maxiter*[[0.0]])
    for i in range(maxiter):
        X_batch, y_batch, idx = get_batches(X,y,batch_size,row,sampleops,i)
        gradient = get_gradient(X_batch,y_batch,theta,weight[idx],loss_type,elambda)
        theta = theta - (0.1 / np.power(i+1,0.5)) * gradient
        if draw_figure == 1:
            loss_total[i] = get_fun_value(X,y,theta,row,loss_type,elambda)
            if ~np.isnan(thetastar).all():
                error[i] = np.linalg.norm(theta - thetastar, 2)
    if np.isnan(thetastar).all():
        return theta, loss_total
    if ~np.isnan(thetastar).all():
        return theta, loss_total, error

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    class sample_ops:
        def __init__(self):
            self.name = []
            self.kernel = []

    N, m = 1000, 2
    losstype = 'linear_regression'
    X, y = generate_data(N, m)
    theta0 = np.array(m*[[0.0]])
    lambda_input = 1

    p = N
    ops = sample_ops()
    ops.name = 'iid'
    theta_true, loss_total = MiniBatchSGD(X, y, theta0, loss_type=losstype, elambda=lambda_input,
                            batch_size=p, maxiter=1000, sampleops=ops)

    p = 100
    maxit = 100

    ops1 = sample_ops()
    ops1.name = 'iid'
    theta1, loss_total1, error1 = MiniBatchSGD(X, y, theta0, loss_type=losstype, elambda=lambda_input,
                         batch_size=p, maxiter=maxit, sampleops=ops1, thetastar=theta_true)

    ops2 = sample_ops()
    ops2.name = 'dpp'
    eig_vals, eig_vecs, Ktilde = generate_DPP_kernel(X,N,p)
    ops2.kernel_diag = np.diag(Ktilde)
    ops2.DPP_list = generate_DPP_list_of_samples(eig_vals, eig_vecs, maxit)
    theta2, loss_total2, error2 = MiniBatchSGD(X, y, theta0, loss_type=losstype, elambda=lambda_input,
                         batch_size=p, maxiter=maxit, sampleops=ops2, thetastar=theta_true)

    inv = np.linalg.inv(np.dot(X.T, X) + N * lambda_input * np.identity(m))
    rhs = np.dot(X.T, y)
    theta_direct = np.dot(inv, rhs)
    #print(theta_direct)
    #print(theta_true)
    #print(theta1)
    #print(theta2)

    # plot1 = plt.figure(1)
    # plt.plot(range(maxit), loss_total1, label='iid')
    # plt.plot(range(maxit), loss_total2, label='dpp')
    # plt.xlabel('iteration number')
    # plt.ylabel('function value')
    # plot1.suptitle('Function value v.s. iteration number')
    # plt.legend()
    #
    # plot2 = plt.figure(2)
    # plt.plot(range(maxit), error1, label='iid')
    # plt.plot(range(maxit), error2, label='dpp')
    # plt.xlabel('iteration number')
    # plt.ylabel('$||\Theta^k-\Theta^*||_2$')
    # plot2.suptitle('Error v.s. iteration number')
    # plt.legend()
    #
    # plt.show()



    # p = 20
    # maxit = 3000
    # ops2 = sample_ops()
    # ops2.name = 'dpp'
    # eig_vals, eig_vecs, Ktilde = generate_DPP_kernel(X, N, p)
    # ops2.kernel_diag = np.diag(Ktilde)
    # ops2.DPP_list = generate_DPP_list_of_samples(eig_vals, eig_vecs, maxit)
    # theta3, loss_total3, error3 = MiniBatchSGD(X, y, theta0, loss_type=losstype,
    #                                            learning_rate=0.1, batch_size=p, maxiter=maxit, sampleops=ops2,
    #                                            thetastar=theta_true)
    #
    # # p = 40
    # maxit = 1500
    # ops2 = sample_ops()
    # ops2.name = 'dpp'
    # eig_vals, eig_vecs, Ktilde = generate_DPP_kernel(X, N, p)
    # ops2.kernel_diag = np.diag(Ktilde)
    # ops2.DPP_list = generate_DPP_list_of_samples(eig_vals, eig_vecs, maxit)
    # theta4, loss_total4, error4 = MiniBatchSGD(X, y, theta0, loss_type=losstype,
    #                                            learning_rate=0.1, batch_size=p, maxiter=maxit, sampleops=ops2,
    #                                            thetastar=theta_true)
    #
    # p = 60
    # maxit = 1000
    # ops2 = sample_ops()
    # ops2.name = 'dpp'
    # eig_vals, eig_vecs, Ktilde = generate_DPP_kernel(X, N, p)
    # ops2.kernel_diag = np.diag(Ktilde)
    # ops2.DPP_list = generate_DPP_list_of_samples(eig_vals, eig_vecs, maxit)
    # theta5, loss_total5, error5 = MiniBatchSGD(X, y, theta0, loss_type=losstype,
    #                                            learning_rate=0.1, batch_size=p, maxiter=maxit, sampleops=ops2,
    #                                            thetastar=theta_true)
    #
    # plot1 = plt.figure(1)
    # print(loss_total3.shape)
    # print(loss_total4.shape)
    # print(loss_total5.shape)
    # plt.plot(range(3000), loss_total3, label='dpp_p20')
    # plt.plot(range(1500), loss_total4, label='dpp_p40')
    # plt.plot(range(1000), loss_total5, label='dpp_p60')
    # plt.xlabel('iteration number')
    # plt.ylabel('function value')
    # plot1.suptitle('Function value v.s. iteration number')
    # plt.legend()
    #
    # plot2 = plt.figure(2)
    # plt.plot(range(3000), error3, label='dpp_p20')
    # plt.plot(range(1500), error4, label='dpp_p40')
    # plt.plot(range(1000), error5, label='dpp_p60')
    # plt.xlabel('iteration number')
    # plt.ylabel('$||\Theta^k-\Theta^*||_2$')
    # plot2.suptitle('Error v.s. iteration number')
    # plt.legend()
    #
    # plt.show()