""" Section 1: Importing libraries into Python: """
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


""" Section 2: Defining all functions """

""" Section 2.1: Defining test functions: """
# Taking the simplest case scenario and beginning with test functions in just 2 dimensions ...

def TF1(x1,x2):
    '''
    Test function 1: 'Quadratic-style'/sum of squares function (CONVEX)
    '''
    y = (x1+3)**2 + (x2-4)**2 #optima @ X*=(-3,4) f(X*)=0
    return y

def TF2(x1,x2):
    '''
    Test function 2: Rosenbrock function (NON-CONVEX): http://mathworld.wolfram.com/RosenbrockFunction.html
    '''
    a=1;b=100 #these are the usual parameters for the Rosenbrock function --> gloab min @ (a,a**2) i.e. (1,1)
    z = ((a-x1)**2) + (b*(x2 - x1**2)**2)
    return z

""" Section 2.2: Defining a 3d plotting function to visualise the above test functions """

def surfplot(f,lim): #copied how to do this surface plot from: https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
    fig =plt.figure()
    ax  =fig.add_subplot(111,projection='3d')
    x   =np.arange(-lim,lim,0.05)
    y   =np.arange(-lim,lim,0.05)
    X,Y =np.meshgrid(x,y)
    zs  =np.array(f(np.ravel(X),np.ravel(Y)))
    Z   =zs.reshape(X.shape)

    ax.plot_surface(X,Y,Z)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f')

""" Section 2.3: Defining finite difference approx. of derivative and gradient descent functions """

def central_difference(f,x1,x2):
    '''
    (Central finite difference approximation of derivative -  see https://en.wikipedia.org/wiki/Finite_difference)

    input:
    f     = objective function
    x1,x2 = variables
    '''
    Delta = np.sqrt(np.finfo(float).eps) # Step size is taken as square root of the machine precision - gives the smallest representable number such that 1.0 + eps != 1.0
    dx1   = (f(x1 + (Delta/2),x2) - f(x1 - (Delta/2),x2))/Delta
    dx2   = (f(x1,x2 + (Delta/2)) - f(x1, x2 - (Delta/2)))/Delta
    dX = [dx1,dx2]
    return dX

def gradient_descent(dx,x,alpha):
    '''
    gradient descent algorithm

    input:
    alpha = 'learning rate' - typically chosen between 1e-6 and 1e-3
    '''
    x = x - alpha*dx
    return x


""" Section 2.4: Defining the local search algorithm functions """

def LS1(f,X0,sigma,n_iter,n_s,T0,alpha,r):
    '''
    Local search algorithm 1: Simulated annealing algorithm - excellent information on simulated annealing from: https://tinyurl.com/y2nhoohq

    input:
      f      = objective function
      X0     = search start point
      sigma  = std dev (used in cov matrix)
      n_iter = number of iterations, number of sample points
      T0     = 'starting temperature'
      alpha  = 'cooling rate'
      r      = 'probability of acceptance' - n.b. final 3 parameters refer to simulated annealing algorithm method
    '''
    x1         =X0[0];x2=X0[1] #initialization
    mu         =np.array([x1,x2])
    cov        =np.array([[sigma,0],[0,sigma]]) #covariance matrix - off-diagnols = 0,0 means we do NOT assume any correlations between x1,x2 apriori
    CurrentMin =f(x1,x2)
    T          =T0
    for i in range(n_iter):
        Pts        = np.random.multivariate_normal(mu,cov,n_s) #Generates random variables distributed around exisitng search point, which are distrubted according to a multivariate Gaussian - used this video to help write this line: https://www.youtube.com/watch?v=r5_zH-Tj53M
        SampleEval =f(Pts[:,0],Pts[:,1]) #evalaute the function at the selected random sample points
        df         = pd.DataFrame(Pts) #DataFrame from Python Panda library
        df.columns = ['x1','x2']#labelling columns 1 and 2, before 3rd column containing function output/evaluation is appended onto the data-structure
        df2        = df.assign(Output=SampleEval)
        FindMin    = df2.loc[df2['Output'].idxmin()] #KEY CODE!!! This is the line which searches for the index where the minimum solution from n_s samples is
        SampleMin  = pd.Series(FindMin).values #There was some Python object compatibility issue between list and integer values I think ... this line sorts it
        T          = alpha*T #on every iteration the 'temperature is reduced (like a geometric progression) - this is key to the simulated annealing algorithm, because at the beginning the algorithm has a high chance of accepting a 'worse' solution to avoid getting stuck in local optima and trying to find the global optima, however, the temperature needs to reduce so that the algorithm tightens towards only accepting better solutions as n_iter increases
        if SampleMin[2] < CurrentMin: #if a sample point is better than the current solution, update this
            CurrentMin = SampleMin[2]
            CurrentSol = np.array([SampleMin[0],SampleMin[1]])
            mu         = np.array([SampleMin[0],SampleMin[1]])
        else: #otherwise check if it meets the criteria set out by the simulated annealing algorithm ...
            diff = SampleMin[2] - CurrentMin #this number will always be positive
            P    = math.exp(-diff/T)
            if P > r:
                CurrentMin = SampleMin[2]
                CurrentSol = np.array([SampleMin[0],SampleMin[1]])
                mu         = np.array([SampleMin[0],SampleMin[1]])
        #print(i,CurrentMin) #very useful for troubleshooting to print values and see what is happening on each iteration...
    return [CurrentSol,CurrentMin] #desired output from function


def LS2(f,finite_diff,X0,alpha,n_iter):
    '''
    Local search algorithm 2: Finite differences method

    input:
    f           = objective function
    finite_diff = function for approximating the derivative
    X0          = search start points
    alpha       = 'learning rate'
    n_iter      = number of iterations
    '''
    x1 = X0[0]; x2 = X0[1]
    for i in range(n_iter):
        dX         = finite_diff(f,x1,x2)
        x1         = gradient_descent(dX[0],x1,alpha)
        x2         = gradient_descent(dX[1],x2,alpha)
        CurrentVal = f(x1,x2)
        #print(i,CurrentVal) #can choose to print iterations to see what is happening
    Sol = [x1,x2]
    return [Sol,CurrentVal]

""" Section 3: Main module (calling functions/algorithms) """

# Using (4,4) as starting point for both functions

print('Simulated annealing algorithm:\n')
SA1 = LS1(TF1,[4,4],1,100,50,500,0.9,0.2) #these are all parameters you could massively play with - 'hyperparameters'
SA2 = LS1(TF2,[4,4],1,100,50,500,0.9,0.2)

print('Solution for test function 1 (sum of squares) is: {}'.format(SA1))
print('Solution for test function 2 (Rosenbrock function) is: {}'.format(SA2))
print('\n')
print('My comments: The sum of squares function is quite easy to optimise, however the result for the Rosenbrock function is very interesting...with the simulated annealing algorithm it helps the solver not get stuck in as many local optima, so compared to without the whole "temperature" part being added I noticed this algorithm is able to reach closer to the true optima of (1,1) --> 0, however it also will produce some results further away from the optima, as sometimes it accepts worse solutions - I have not played around with the parameters enough yet... ')

print('\n')


print('Finite differences method: \n')

finite1 = LS2(TF1,central_difference,[4,4],1e-3,5000)
print('Solution for test function 1 (sum of squares): {}'.format(finite1))

print('\n')

finite2 = LS2(TF2,central_difference,[4,4],1e-3,100)
print('Solution for test function 2 (Rosenbrock function): {}'.format(finite2))
print('\n')
print('Either I have made a mistake codding the finite differences algorithm (but I do not think so because it works for TF1), so I would conclude the algorithm works terribly for the Rosenbrock function, as it blows up very quickly. Evaluated at the starting point of (4,4) it is equal to 14409 and then the way dx is definied if you divide f(x) by delta, which is very small, this makes the function diverge further very quickly :( - need to use completely different parameters for this...) ')


print('\n')
print('And 3d plots of the test functions look like: ')
surfplot(TF1,5) #plotting surface z for x1,x2 between -5 to 5
surfplot(TF2,5)
