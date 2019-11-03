""" Section 1: Importing libraries into Python  """

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

""" Section 2: Defining all functions """

""" Section 2.1: Defining test function """

def TF(X):
    '''
    Test function: Rosenbrock function (http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html)
    '''
    n = np.shape(X)[0]
    z = 0
    for i in range(n-1):
        z += ((1-X[i])**2) + (100*(X[i+1] - X[i]**2)**2)
    return z

""" Section 2.2: Defining a function to compute the gradient """

def central_finite_diff(f,X):
        Delta = np.sqrt(np.finfo(float).eps) #step-size is taken as the square root of the machine precision
        X     = np.ndarray.astype(X,float)
        n     = np.shape(X)[0]
        dX    = np.zeros([n,1])
        for j in range(n):
            X_dash1      = np.copy(X)
            X_dash2      = np.copy(X)
            X_dash1[j]  += Delta/2
            X_dash2[j]  += - Delta/2
            dX[j]        = (f(X_dash1) - f(X_dash2))/Delta

        return dX

""" Section 2.3: Defining the optimisation algorithms """

def Adam(f,compute_grad,X,n_iter,beta1,beta2):
    '''
    "Adam" (adaptive moment estimation) algorithm combines gradient descent with momentum and RMSprop algorithms. Additionally I've included "backtracking" for the alpha (learning rate)

    f = objective function
    compute_grad = function estimating the gradient (here will use 'central_finite_diff')
    X = input vector (starting point)
    n_iter = number of iterations
    beta1 = parameter for taking an exponentially weighted average for the gradient descent with momentum component of this algorithm (default let = 0.9)
    beta2 = parameter for the RMSprop component of this algorithm (default let = 0.9)
    '''
    alpha         = 1e-2
    alpha_history = np.array([alpha])
    alpha_history = alpha_history.reshape(np.shape(alpha_history)[0],1)
    f_history     = np.empty((1,0))
    X_history     = np.empty((np.shape(X)[0],0))
    eps           = 1e-8
    VdX           = np.zeros([np.shape(X)[0],1])
    SdX           = np.zeros([np.shape(X)[0],1])
    for i in range(n_iter):
        X         = X.reshape(np.shape(X)[0],1)
        X_history = np.append(X_history,X,axis=1)
        OldVal    = f(X)
        f_history = np.append(f_history,OldVal)
        dX        = central_finite_diff(f,X)
        VdX       = (beta1*VdX + (1-beta1)*dX)/(1-beta1**(i+1))
        SdX       = beta2*SdX + (1-beta2)*dX**2
        X_new     = X - alpha*(VdX/(np.sqrt(SdX)+eps))
        NewVal    = f(X_new)
        counter   = 0
        while OldVal < NewVal:
            counter  += 1
            alpha     = alpha/2
            dX        = central_finite_diff(f,X)
            VdX       = beta1*VdX + (1-beta1)*dX/(1-beta1**(i+1))
            SdX       = beta2*SdX + (1-beta2)*dX**2
            X_new     = X - alpha*(VdX/(np.sqrt(SdX)+eps))
            NewVal    = f(X_new)
            if counter > 100:
                print('Cannot find suitable alpha')
                break

        if np.count_nonzero(alpha_history == alpha) == 0:
            alpha_history = np.append(alpha_history,alpha)
            alpha_history = alpha_history.reshape(np.shape(alpha_history)[0],1)

        alpha     = 2*np.mean(alpha_history)
        X         = X_new

    CurrentVal = f(X)

    return X,CurrentVal,f_history,X_history


def gradient_descent(f,compute_grad,X,n_iter):
    alpha         = 1e-2
    alpha_history = np.array([alpha])
    alpha_history = alpha_history.reshape(np.shape(alpha_history)[0],1)
    f_history     = np.empty((1,0))
    X_history     = np.empty((np.shape(X)[0],0))
    for i in range(n_iter):
        X         = X.reshape(np.shape(X)[0],1)
        X_history = np.append(X_history,X,axis=1)
        OldVal    = f(X)
        f_history = np.append(f_history,OldVal)
        dX        = central_finite_diff(f,X)
        X_new     = X - alpha*dX
        NewVal    = f(X_new)
        counter   = 0
        while OldVal < NewVal:
            counter  += 1
            alpha     = alpha/2
            dX        = central_finite_diff(f,X)
            X_new     = X - alpha*dX
            NewVal    = f(X_new)
            if counter > 100:
                print('Cannot find suitable alpha')
                break

        if np.count_nonzero(alpha_history == alpha) == 0:
            alpha_history = np.append(alpha_history,alpha)
            alpha_history = alpha_history.reshape(np.shape(alpha_history)[0],1)

        alpha     = 2*np.mean(alpha_history)
        X         = X_new

    CurrentVal = f(X)

    return X,CurrentVal,f_history,X_history


""" Section 3: Main script """

X = np.array([4,4])
A = Adam(TF,central_finite_diff,X,100,0.9,0.9)
B = gradient_descent(TF,central_finite_diff,X,100)
#print(A)
#print(B)

""" Section 4: Plotting results """

""" Figure 1: Iteration path on contour plot"""

n_points = 25
x_1      = np.linspace(-8,8,n_points)
x_2      = np.linspace(-8,8,n_points)
x_Adam   = A[3]
x_GD     = B[3]
x        = [[x,y] for x in x_1 for y in x_2]
x        = np.array(x)
f        = TF(x.T)
f_copy   = f.reshape((n_points,n_points),order ='F')
fig, ax  = plt.subplots()
CS       = ax.contour(x_1,x_2,np.log(f_copy),15)
ax.clabel(CS,fontsize=10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Rosenbrock function')
ax.plot(x_Adam[0,:],x_Adam[1,:],'b--',label='Adam algorithm')
ax.plot(x_Adam[0,:],x_Adam[1,:],'bx',mew=2)
ax.plot(x_GD[0,:],x_GD[1,:],'m--',label='Gradient descent')
ax.plot(x_GD[0,:],x_GD[1,:],'mo',mew=2)
ax.legend(loc='lower left')
plt.show()

""" Figure 2: Objective function evaluation vs. number of iterations """

f_eval_Adam  = A[2]
f_eval_GD    = B[2]
#f_eval_Adam = f_eval_Adam[5:100:1]
#f_eval_GD   = f_eval_GD[5:100:1]

fig2   = plt.figure()
n_iter = 100
iter   = np.arange(n_iter)
plt.plot(iter,f_eval_Adam,'b-',label='Adam')
plt.plot(iter,f_eval_GD,'m-',label='Gradient Descent')
plt.legend(loc='upper right')
plt.ylabel('Objective function output')
plt.xlabel('Number of iterations')
plt.show()
