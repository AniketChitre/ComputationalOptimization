""" Section 0: Introduction/Context """

'''
The code in this file develops on the 'week 1' gradient descent (GD) and simulated annealing (SA) algorithms by extending them from 2 to n-dimensions. The test function utilised is the "Rosenbrock function", which is also generalised to higher dimensions.

The GD and SA are able to take an input vector of dimension n (i.e. X E R^n) and will recognise the number of variables (x1,x2,...,xn) from the size of the vector and perform the optimization accordingly.

Additionally, the GD algorithm is also extended to take an input matrix (i.e. X E R^mxn, where there are x1,x2,...,xn variables and m rows of starting points). This way by feeding in a matrix you can probe different starting points for the local search (LS) algorithm in one run.

Furthermore, compared to 'week 1' the GD algorithm has been modified with "backtracking" which causes the 'alpha' (i.e. learning rate) to be varied to improve the algorithm performance.

The final section of the script feeds in examples to the algorithms to illustrate the functionality of this code.

***P.S. Throughout this script there are a lot of "try or except:" structures as this allows the functionality of either a input vector X or input matrix X to be fed into the functions without dimensions mismatch blowing the script up***
'''

""" Section 1: Importing libraries into Python  """

import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

""" Section 2: Defining all functions """

""" Section 2.1: Defining test function """

def TF(X):
    '''
    Test function: Rosenbrock function (http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html)
    '''
    try:
        rows = np.shape(X)[0]
        cols = np.shape(X)[1]
        z    = np.zeros([rows,1])
        for r in range(rows):
            for i in range(cols-1):
                z[r] += ((1-X[r][i])**2) + (100*(X[r][i+1] - X[r][i]**2)**2)

    except:
        cols = np.shape(X)[0]
        z    = np.zeros([1,1])
        for i in range(cols-1):
            z += ((1-X[i])**2) + (100*(X[i+1] - X[i]**2)**2)

    return z

""" Section 2.2: Defining plotting functions """

def fxn_iter_plot(n_iter,f_eval,X0):
    '''
    Figure to show the function output vs. number of iterations

    input:
    n_iter = number of iterations
    f_eval = vector storing the function outputs (function evaluations)
    X0     = starting point (to create legend when results from different starting points are plotted on the same figure)
    '''
    #plt.figure() #comment in or out depending on whether you want results showing on separate figures or not
    iter       = np.arange(n_iter) #creates points to plot against
    intial_val = np.array2string(X0) #need to convert array to string to display as a legend label
    plt.plot(iter,f_eval,'-',label=intial_val)
    plt.legend(loc='upper right')
    plt.ylabel('Function output')
    plt.xlabel('Number of iterations')
 


def iteration_path(x1,x2):
    '''
    For 2d functions this will plot how (x1,x2) vary from the initial search point to the finally obtained 'solution' (algorithm output)
    '''
    plt.figure() #creates a new figure
    plt.plot(x1[0],x2[0],'o',label='initial point') #plots the intial point as a circle
    plt.plot(x1[1:-1],x2[1:-1],'.') #plots the intermediate solution space as dots
    plt.plot(x1[-1],x2[-1],'x',label='final point') #plots the final point as a cross
    plt.legend(loc='upper left')
    plt.xlabel('x1')
    plt.ylabel('x2')


""" Section 2.3: Defining functions to compute the gradient and gradient descent """

'''
Two functions are defined to compute the gradient - a forward finite differences method and central finite differences method. The latter is a more accurate approximation of the gradient, however, is also doubly as computationally intensive, so there is a trade-off - can call either function into the gradient descent local search algorithm in section 2.4
'''

def forward_finite_diff(f,X):
    '''
    input:
    f = objective function
    X = test space
    '''
    Delta = np.sqrt(np.finfo(float).eps) #step-size is taken as the square root of the machine precision - gives the smallest representable number such that 1.0 + eps != 1.0
    X     = np.ndarray.astype(X,float) #converts the input vector to a 'float' type - ensures 'type' compatibility issues are avoided
    try: #if X is an input matrix
        rows = np.shape(X)[0]
        cols = np.shape(X)[1] #this line fails if X is an input vector and causes the code to skip to the except: statement
        dX   = np.zeros([rows,cols])
        for i in range(rows):
            for j in range(cols):
                X_dash         = np.copy(X)
                X_dash[i][j]  += Delta
                dX[i][j]       = (f(X_dash[i,:]) - f(X[i,:]))/Delta

    except: #if X is an input vector
        cols = np.shape(X)[0]
        dX   = np.zeros([1,cols])
        for j in range(cols):
            X_dash      = np.copy(X)
            X_dash[j]  += Delta
            dX[0][j]    = (f(X_dash) - f(X))/Delta

    return dX


def central_finite_diff(f,X):
    Delta = np.sqrt(np.finfo(float).eps)
    X = np.ndarray.astype(X,float)
    try:
        rows = np.shape(X)[0]
        cols = np.shape(X)[1]
        dX   = np.zeros([rows,cols])
        for i in range(rows):
            for j in range(cols):
                X_dash1         = np.copy(X)
                X_dash2         = np.copy(X)
                X_dash1[i][j]  += Delta/2
                X_dash2[i][j]  += -Delta/2
                dX[i][j]        = (f(X_dash1[i,:]) - f(X_dash2[i,:]))/Delta

    except:
        cols = np.shape(X)[0]
        dX   = np.zeros([1,cols])
        for j in range(cols):
            X_dash1      = np.copy(X)
            X_dash2      = np.copy(X)
            X_dash1[j]  += Delta/2
            X_dash2[j]  += - Delta/2
            dX[0][j]     = (f(X_dash1) - f(X_dash2))/Delta

    return dX


def gradient_descent(X,dX,alpha):
    X = X - alpha*dX
    return X


""" Section 2.4: Defining the local search algorithms """

def LS1(f,compute_grad,X0,n_iter):
    '''
    Gradient descent algorithm with backtracking

    input:
    f            = objective function
    compute_grad = calls functions to estimate the gradient (either by forward or central finite differences)
    X0           = search start point
    n_iter       = number of iterations
    '''
    X             = X0
    alpha         = 1e-2 #initializes value for 'alpha' - the "learning rate"
    alpha_history = np.array([alpha]) #creates an empty array to store the values of alpha, as wish to keep a running record, which will be used to declare the updated value of alpha to be used for the next iteration


    #need lots of "try and except" statements to make the code work for both vector or matrix inputs
    # initializing empty arrays with the right dimensions is very important

    #Sol_history will store the x values which the algorithm goes through
    try:
        Sol_history = np.empty((np.shape(X)[0],np.shape(X)[1],0)) #3d array
    except:
        Sol_history = np.empty((0,np.shape(X)[0]))

    #f_history will store the evaluation of the function - the function outputs through n_iter iterations
    try:
        f_history = np.empty((np.shape(X)[0],0))
    except:
        f_history = np.empty((1,0))

    for i in range(n_iter):
        OldVal  = f(X)
        dX      = compute_grad(f,X)
        X_new   = gradient_descent(X,dX,alpha)
        NewVal  = f(X_new)
        counter = 0
        while np.any(OldVal < NewVal):
            counter  += 1
            alpha     = alpha/2 #it keeps halving alpha till a suitable value is found such that the new evaluation is an improvement on the previous one
            dX        = compute_grad(f,X)
            X_new     = gradient_descent(X,dX,alpha)
            NewVal    = f(X_new)
            if counter > 100: #this counter prevents it getting stuck in the while loop indefinitely
                print('Cannot find suitable alpha')
                break

        X = X_new #update the new X value before it feeds into the next loop round

        if np.count_nonzero(alpha_history == alpha) == 0: #if it's a new alpha then append it in the record of alphas
            alpha_history = np.append(alpha_history,alpha)

        alpha = 2*np.mean(alpha_history) #update alpha as this


        '''
        The next sub-section calculates & plots how the function output changes with increasing number of iterations
        '''
        try:
            f_history = np.append(f_history,NewVal,axis=1)
        except:
            f_history = np.append(f_history,NewVal)

        try:
            #Sol_history = np.dstack((Sol_history,X))  #either of these two options work - just comment out one ...
            Sol_history = np.append(Sol_history,np.atleast_3d(X),axis=2)
        except:
            Sol_history = np.append(Sol_history,X,axis=0)


        try:
            f_history_final = f_history[:,-1]
            f_history_final = f_history_final.reshape(np.shape(f_history_final)[0],1)
            Final           = np.append(Sol_history[:,:,-1],f_history_final,axis=1)
        except:
            Final = np.append(Sol_history[-1,:],f_history[-1])

    plt.figure()
    try:
        rows = np.shape(X)[0]
        for r in range(rows):
            fxn_iter_plot(n_iter,f_history[r,:],X0[r,:])
    except:
        fxn_iter_plot(n_iter,f_history,X0)

    plt.show()


    '''
    This next sub-section calculates the number of evaluations computed to compare how computationally intensive different algorithms are
    '''
    if compute_grad == central_finite_diff:
        multiplier = 2
    else:
        multiplier = 1
    #because the central finite difference method goes half a delta either way (hence multiplier = 2), but the forward finite differences only use delta in the forward direct (hence multiplier = 1)

    try:
        rows     = np.shape(X)[0]
        cols     = np.shape(X)[1]

        num_eval = n_iter*(((rows*cols)*multiplier)+(rows*cols))

    except:
        cols     = np.shape(X)[0]

        num_eval = n_iter*(((rows*cols)*multiplier)+(rows*cols))

    print('Number of evaluations performed by gradient descent algorithm: {}'.format(num_eval))


    '''
    The next subsection only works if the input vector is 2d --> then it'll show how the solution space is search from the initial point X0 to the final point the algorithm finds
    '''
    if np.shape(X) == (1,2):
        x1 = Sol_history[:,0]
        x2 = Sol_history[:,1]
        iteration_path(x1,x2)

        plt.show()


    return Final



def LS2(f,X0,sigma,n_iter,n_s,T0,alpha,r):
    '''
    Local search algorithm 2: Simulated annealing algorithm - information on simulated annealing from: https://tinyurl.com/y2nhoohq

    input:
      f      = objective function
      X0     = search start point
      sigma  = std dev (used in cov matrix)
      n_iter = number of iterations, number of sample points
      T0     = 'starting temperature'
      alpha  = 'cooling rate'
      r      = 'probability of acceptance' - n.b. final 3 parameters refer to simulated annealing algorithm method
    '''
    mu   = X0 #initialize the mean as the search start point
    dims = np.shape(X0)[0] #dimensions
    cov  = np.zeros([dims,dims]) #need a nxn covariance matrix where n is the number of x variables (i.e. x1,...xn)

    for i in range(dims):
        cov[i][i] = sigma
    #making the diagnols of the covariance matrix sigma and leaving all the off-diagnols as zero - the significance of non-zero off diagnols in cov(X) is that the variables have some sort of correlation with eachother

    CurrentMin  = f(mu)
    T           = T0
    Sol_history = np.empty((dims,0))
    f_history   = np.empty((1,0))

    for i in range(n_iter):
        Pts        = np.random.multivariate_normal(mu,cov,n_s) #Generates random variables distributed around exisitng search point, which are distrubted according to a multivariate Gaussian - used this video to help write this line: https://www.youtube.com/watch?v=r5_zH-Tj53M
        SampleEval = f(Pts) #evalaute the function at the selected random sample points
        df         = pd.DataFrame(Pts) #DataFrame from Python Panda library
        df2        = df.assign(Output=SampleEval)
        FindMin    = df2.loc[df2['Output'].idxmin()] #KEY CODE!!! This is the line which searches for the index where the minimum solution from n_s samples is
        SampleMin  = pd.Series(FindMin).values #Sorts out a Python object compatibility issue between list and integer values
        T          = alpha*T #on every iteration the 'temperature is reduced (like a geometric progression) - this is key to the simulated annealing algorithm, because at the beginning the algorithm has a high chance of accepting a 'worse' solution to avoid getting stuck in local optima and trying to find the global optima, however, the temperature needs to reduce so that the algorithm tightens towards only accepting better solutions as n_iter increases

        # SampleMin[-1] because the last element captures the actual function output
        if SampleMin[-1] < CurrentMin: #if a sample point is better than the current solution, update this
            CurrentMin = SampleMin[-1]
            CurrentSol = SampleMin[:-1]
            mu         = SampleMin[:-1]
        else: #otherwise check if it meets the criteria set out by the simulated annealing algorithm
            diff = SampleMin[-1] - CurrentMin #this number will always be positive
            P    = math.exp(-diff/T)
            if P > r:
                CurrentMin = SampleMin[-1]
                CurrentSol = SampleMin[:-1]
                mu         = SampleMin[:-1]

        Current_Sol = CurrentSol.reshape(CurrentSol.shape[0],1)
        Sol_history = np.append(Sol_history,Current_Sol,axis=1)
        f_history   = np.append(f_history,CurrentMin)

    plt.figure()
    fxn_iter_plot(n_iter,f_history,X0) #plots how the function output changes with the number of iterations
    plt.show()

    '''
    If it's a 2d input vector then this will plot which points have been searched by the SA algorithm
    '''
    if dims == 2:
        x1 = Sol_history[0,:]
        x2 = Sol_history[1,:]
        iteration_path(x1,x2)

        plt.show()


    num_eval = n_s*n_iter
    print('Number of evaluations performed by simulated annealing algorithm: {}'.format(num_eval))


    return [CurrentSol,CurrentMin]



""" Section 3: Main module (call functions/algorithms) """

#test inputs
X = np.array([4,4]) #basic 2d test case
Y = np.array([2,2,2]) #extending to a higher dimension - would work to n
Z = np.array([[1.5,1.5],[2,2],[2.5,2.5]])#input can also be a matrix with multiple rows

print('\n')
print('Guide to read the solutions:')
print('For a n-input vector (i.e. x1,...,xn) the first n values show the "optimal" x values (solution points) and the final value is the actual output of the function at the aforementioned point')
print('\n')

print('Gradient descent algorithm for an 2d input vector:')

result1 = LS1(TF,central_finite_diff,X,20)
print(result1)
print('\n')

print('Gradient descent algorithm for an 3d input vector:')

result1 = LS1(TF,central_finite_diff,Y,20)
print(result1)
print('\n')

print('Simulated annealing algorithm with a 2d input vector:')

result2 = LS2(TF,X,1,100,50,500,0.9,0.2)
print(result2)
print('\n')

print('Simulated annealing algorithm with a 3d input vector:')

result2 = LS2(TF,Y,1,100,50,500,0.9,0.2)
print(result2)
plt.savefig('Simulated_annealing')

print('\n')


print('Gradient descent algorithm with a matrix input (3,2):')

result3 = LS1(TF,forward_finite_diff,Z,20)
print(result3)
print('\n')
