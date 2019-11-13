""" Section 1: Importing libraries into Python """

import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt


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


""" Section 2.3: Defining the BFGS algorithm """

def BFGS(f,comp_grad,X0,n_iter):
    '''
    BFGS is a quasi-Newton optimisation algorithm. Equations defining this algorithm used from Nocedal and Wright's 'Numerical Optimization' textbook

    f = objective function
    comp_grad = function estimating the gradient (here will use 'central_finite_diff')
    X0 = input vector (starting point)
    n_iter = number of iterations
    '''
    size      = X0.shape[0]
    #initialize parameters for BFGS algorithm
    X1        = X2 = grad1 = grad2 = H1 = H2 = S = Y = np.empty((size,0))
    I         = np.identity(size)
    X0        = X0.reshape(X0.shape[0],1)
    eps       = 1e-10 #used in the initial estimation of the Hessian in the "warming up phase" of the BFGS algorithm
    f_history = np.empty((1,0))
    X_history = np.empty((size,0))

    for i in range(n_iter):
        if i == 0:
            X1    = X0
            grad1 = comp_grad(f,X1)
            H1    = I*(grad1*eps)
            X2    = X1 - np.dot(H1,grad1)

        elif i == 1:
            X1    = X2
            grad1 = comp_grad(f,X1)
            H2    = I*(grad1*eps)
            X2    = X1 - np.dot(H1,grad1)

        else:
            X1    = X2
            H1    = H2
            grad1 = comp_grad(f,X1)
            X2    = X1 - np.dot(H1,grad1)
            grad2 = comp_grad(f,X2)
            S     = X2 - X1
            Y     = grad2 - grad1
            rho   = 1/(np.dot(np.transpose(Y),S)+eps) #added eps into denominator to ensure the algorithm doesn't crash at the optimum - otherwise at the optimum the gradient --> 0 and Y --> 0 and dividing by 0 is infeasible
            H2    = np.dot(np.dot((I - rho*np.dot(S,np.transpose(Y))),H1),(I - rho*np.dot(Y,np.transpose(S)))) + (rho*np.dot(S,np.transpose(S)))

        X_history = np.append(X_history,X1,axis=1)
        f_history = np.append(f_history,f(X1))

        Sol       = f(X2)

    return X2,Sol,X_history,f_history


""" Section 2.4: Defining the Adam algorithm """

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

    Sol = f(X)

    return X,Sol,X_history,f_history


""" Section 3: Main module """

X = np.array([4,4])
n_iter = 100
A = Adam(TF,central_finite_diff,X,n_iter,0.9,0.9)
B = BFGS(TF,central_finite_diff,X,n_iter)

print('Solution with the Adam algorithm in {} iterations is: {}'.format(n_iter,A[0:2]))
print('\n')
print('Solution with the BFGS algorithm in {} iterations is: {}'.format(n_iter,B[0:2]))


""" Section 4: Plotting results """

f_eval_Adam = A[3]
f_eval_BFGS = B[3]

fig = plt.figure()
iter = np.arange(n_iter)
plt.plot(iter,f_eval_Adam,'b-',label='Adam')
plt.plot(iter,f_eval_BFGS,'m-',label='BFGS')
plt.yscale('log')
plt.legend(loc='lower left')
plt.ylabel('Objective function output')
plt.xlabel('Number of iterations')
plt.show()
