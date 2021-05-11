import numpy as np
import random
import matplotlib.pyplot as plt

# Function for dividing the data into batches
def get_batches(X,y,batch_size,row,sample_ops):
    if sample_ops == 'iid':
        idx = random.sample(range(row), batch_size)
    X_new = X[idx,:]
    y_new = y[idx]
    return X_new, y_new

# Function for computing the gradient
def get_gradient(X,y,theta,batch_size,loss_type):
    if loss_type == 'linear_regression':
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y
        X_trans = X.transpose()
        gradient = np.dot(X_trans, loss) / batch_size
    elif loss_type == 'logistic_regression':
        hypothesis = np.dot(X, theta)
        loss = -y*(1-1/(1+np.exp(-hypothesis * y)))
        X_trans = X.transpose()
        gradient = np.dot(X_trans, loss) / batch_size
    return gradient

# Function for computing the function value
def get_fun_value(X,y,theta,m,loss_type):
    if loss_type == 'linear_regression':
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y
        fun_value = 0.5 * np.dot(loss.transpose(), loss) / m
    elif loss_type == 'logistic_regression':
        hypothesis = np.dot(X, theta)
        fun_value = np.sum(np.log(1 + np.exp(-hypothesis * y))) / m
    return fun_value

# Mini-Batch SGD
def MiniBatchSGD(X,y,theta,loss_type,learning_rate,batch_size,maxiter,sample_ops):
    row, col = X.shape
    draw_figure = 1
    if draw_figure:
        loss_total = np.array(maxiter*[[0.0]])
    for i in range(0,maxiter):
        X_batch, y_batch = get_batches(X,y,batch_size,row,sample_ops)
        gradient = get_gradient(X_batch,y_batch,theta,batch_size,loss_type)
        theta = theta - learning_rate * gradient
        if draw_figure == 1:
            loss_total[i] = get_fun_value(X,y,theta,m,loss_type)
    if draw_figure:
        plt.plot(range(maxiter), loss_total)
        plt.show()
    return theta

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    m = 1000
    n = 200
    X = np.random.normal(0,1,size=(m, n))
    y = np.random.normal(0,1,size=(m, 1))
    # y[1] = 0
    # y = np.sign(y)
    # tmp = np.argwhere(y==0)
    # if tmp.size > 0:
    #     y[tmp] = 1
    theta0 = np.array(n*[[0]])
    theta = MiniBatchSGD(X, y, theta0, loss_type='linear_regression',
                         learning_rate=0.1, batch_size=100, maxiter=200, sample_ops='iid')
