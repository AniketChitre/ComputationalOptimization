""" Section 0: Background - This script runs a dynamic optimisation of a bioprocess control problem by applying a global (incomplete) optimisation algorithm (simulated annealing). It's an example problem illustrating this concept. """

""" Section 1: Importing libraries into Python """

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
import pandas as pd

""" Section 2: Defining bioprocess simulation """

#Parameters in kinetic model defined using a dictionary
params={'u_m' : 0.0923*0.62, 'k_s' : 178.85, 'k_i' : 447.12, 'K_N' : 393.10, 'u_d' : 0.001, 'Y_nx' : 504.49, 'k_m' : 2.544*0.62,  'k_sq' : 23.51, 'k_iq' : 800.0, 'k_d' : 0.281, 'K_Np' : 16.89}
#I don't actually know the units that all this is in

def BioProcess(state,t,LI,FCn):
    #state vector
    Cx     = state[0]
    Cn     = state[1]
    Cqc    = state[2]
    #constants - calling values from dictionary
    u_m    = params['u_m']; k_s = params['k_s'];
    k_i    = params['k_i']; K_N = params['K_N'];
    u_d    = params['u_d']; Y_nx = params['Y_nx'];
    k_m    = params['k_m']; k_sq = params['k_sq'];
    k_iq   = params['k_iq']; k_d = params['k_d'];
    K_Np   = params['K_Np'];
    #Kinetic model equations
    dCxdt  = u_m * LI/(LI+k_s+LI**2/k_i) * Cx * Cn/(Cn+K_N) - u_d*Cx
    dCndt  = - Y_nx * u_m * LI/(LI+k_s+LI**2/k_i) * Cx * Cn/(Cn+K_N) + FCn
    dCqcdt = k_m * LI/(LI+k_sq+LI**2/k_iq) * Cx - k_d*Cqc/(Cn+K_Np)
    return np.array([dCxdt,dCndt,dCqcdt],dtype='float64')

#LI - light intensity and FCn - nutrient inflowrate; These are the two key decision variables for this optimisation problem - these are the bioprocess controls

t_0 = 0
Total_time = 12*24
steps_n = 24 #The total time horizion is split into 24 steps
t = np.linspace(0,Total_time,steps_n)


""" Section 3: Defining optimisation methods """

def SA(f,X0,sigma,n_iter,n_s,T0,alpha,r,LB,UB):
    '''
    Simulated annealing algorithm - excellent information on simulated annealing from: https://tinyurl.com/y2nhoohq

    input:
      f      = objective function
      X0     = search start point
      sigma  = std dev (used in cov matrix)
      n_iter = number of iterations
      n_s    = number of sample points
      T0     = 'starting temperature'
      alpha  = 'cooling rate'
      r      = 'probability of acceptance' - n.b. T0, alpha, r parameters refer to simulated annealing algorithm method
      LB     = lower bounds
      UB     = upper bounds
    '''
    mu   = X0
    dims = np.shape(X0)[0]
    cov  = np.zeros([dims,dims])
    for i in range(dims):
        cov[i][i] = sigma
    CurrentMin  = f(mu)
    T           = T0
    Sol_history = np.empty((dims,0))
    f_history   = np.empty((1,0))

    for i in range(n_iter):

        Pts        = np.random.multivariate_normal(mu,cov,n_s) #Generates random variables distributed around exisitng search point, which are distrubted according to a multivariate Gaussian - used this video to help write this line: https://www.youtube.com/watch?v=r5_zH-Tj53M

        #The selected points need to be constrained within set upper and lower bounds
        for k in range(n_s): #loops through each set of sample variables one at a time
            for j in range(0,4): #variables refering to the light intensity
                if (Pts[k,j]   >= UB[0]):
                    Pts[k,j]    = UB[0]
                elif (Pts[k,j] <= LB[0]):
                    Pts[k,j]    = LB[0]
                else:
                    Pts[k,j]    = Pts[k,j]
            for j in range(4,8): #variables refering to the nutrient inflowrate
                if (Pts[k,j]   >= UB[1]):
                    Pts[k,j]    = UB[1]
                elif (Pts[k,j] <= LB[1]):
                    Pts[k,j]    = LB[1]
                else:
                    Pts[k,j]    = Pts[k,j]

        #The integration function numerically solving the bioprocess model cannot take multiple rows of data at once, so each set needs to be evaluated one at a time and then stored back into the "SampleEval" matrix
        SampleEval = np.empty(n_s)
        for k in range(n_s):
            SampleEval[k] = f(Pts[k,:]) #evalaute the function at the selected random sample points
        SampleEval = SampleEval.reshape(SampleEval.shape[0],1)

        df         = pd.DataFrame(Pts) #DataFrame from Python Panda library
        df2        = df.assign(Output=SampleEval)
        FindMin    = df2.loc[df2['Output'].idxmin()] #KEY CODE!!! This is the line which searches for the index where the minimum solution from n_s samples is
        SampleMin  = pd.Series(FindMin).values #There was some Python object compatibility issue between list and integer values I think ... this line sorts it
        T          = alpha*T #on every iteration the 'temperature is reduced (like a geometric progression) - this is key to the simulated annealing algorithm, because at the beginning the algorithm has a high chance of accepting a 'worse' solution to avoid getting stuck in local optima and trying to find the global optima, however, the temperature needs to reduce so that the algorithm tightens towards only accepting better solutions as n_iter increases

        if SampleMin[-1] < CurrentMin: #if a sample point is better than the current solution, update this
            CurrentMin = SampleMin[-1]
            CurrentSol = SampleMin[:-1]
            mu         = SampleMin[:-1]
        else: #otherwise check if it meets the criteria set out by the simulated annealing algorithm ...
            diff = SampleMin[-1] - CurrentMin #this number will always be positive
            P    = math.exp(-diff/T)
            if P > r:
                CurrentMin = SampleMin[-1]
                CurrentSol = SampleMin[:-1]
                mu         = SampleMin[:-1]

        #print(i,CurrentMin) #very useful for troubleshooting to print values and see what is happening on each iteration...
        Current_Sol = CurrentSol.reshape(CurrentSol.shape[0],1)
        Sol_history = np.append(Sol_history,Current_Sol,axis=1)
        f_history   = np.append(f_history,CurrentMin)

    return [CurrentSol,CurrentMin,Sol_history,f_history]


""" Section 4: Defining objective function """

#The 24 step total time horizion is further sub-divided into 4 piecewise-constant control actions - there are 8 variables in total, as 4 LI and 4 FCn variables - more decision variables could be chosen, however, it starts to make the programme increasingly computationally intensive/slow, so to illustrate the dynamic optimisation approach, this script sticks to just 4 constant time intervals over which the light intensity & nutrient inflowrates are optimised subject to a certain objective function below

def ProcessScore(X):
    #Piecewise-constant controls for light intensity and nutrient inflowrates
    L1  = X[0]; L2 = X[1]; L3 = X[2]; L4 = X[3]
    N1  = X[4]; N2 = X[5]; N3 = X[6]; N4 = X[7]
    LI  = np.zeros(steps_n)
    FCn = np.zeros(steps_n)
    LI[0:6]  = L1; LI[6:12]  = L2; LI[12:18]  = L3; LI[18:24]  = L4
    FCn[0:6] = N1; FCn[6:12] = N2; FCn[12:18] = N3; FCn[18:24] = N4

    #initial concentrations of x,n,qc
    initial_state = [0.1,150,0]
    Cx            = [initial_state[0]]
    Cn            = [initial_state[1]]
    Cqc           = [initial_state[2]]

    for i in range(len(t)-1):
        ts            = [t[i],t[i+1]] #time step over which odeint is integrating over
        state         = odeint(BioProcess,initial_state,ts,args=(LI[i],FCn[i])) #KEY STEP - integrates the bioprocess model!
        initial_state = state[-1,:3] #extracts the last row as the new initial state
        Cx.append(initial_state[0])
        Cn.append(initial_state[1])
        Cqc.append(initial_state[2])

    cost = 5 #Weighting for the cost of the nutrient feed
    z    = - Cqc[-1] + cost*sum(X[4:8])#The simualted annealing algorithm is written for a minimisation problem so to max product = min - product; additionally, the objective function includes a cost for the amount of nutrient feed used

    return z


""" Section 5: Main module calling dynamic optimisation """

X      = np.array([200,200,200,200,1.0,1.0,1.0,1.0]) #initial conditions
LB     = np.array([100,0.0])
UB     = np.array([300,3.0])
sigma  = 1
n_iter = 200; n_s = 50
T0     = 600; alpha = 0.9; r = 0.5

Result = SA(ProcessScore,X,sigma,n_iter,n_s,T0,alpha,r,LB,UB)


""" Section 6: Plotting the optimisation """

def ProcessScore_Plot(X): #modifying the above function to include plotting
    #Piecewise-constant controls for light intensity and nutrient inflowrates
    L1  = X[0]; L2 = X[1]; L3 = X[2]; L4 = X[3]
    N1  = X[4]; N2 = X[5]; N3 = X[6]; N4 = X[7]
    LI  = np.zeros(steps_n)
    FCn = np.zeros(steps_n)
    LI[0:6]  = L1; LI[6:12]  = L2; LI[12:18]  = L3; LI[18:24]  = L4
    FCn[0:6] = N1; FCn[6:12] = N2; FCn[12:18] = N3; FCn[18:24] = N4

    #initial concentrations of x,n,qc
    initial_state = [0.1,150,0]
    Cx            = [initial_state[0]]
    Cn            = [initial_state[1]]
    Cqc           = [initial_state[2]]

    for i in range(len(t)-1):
        ts            = [t[i],t[i+1]] #time step over which odeint is integrating over
        state         = odeint(BioProcess,initial_state,ts,args=(LI[i],FCn[i]))
        initial_state = state[-1,:3] #extracts the last row as the new initial state
        Cx.append(initial_state[0])
        Cn.append(initial_state[1])
        Cqc.append(initial_state[2])

    plt.figure()
    plt.plot(t,Cx,label='Cx') # biomass
    plt.plot(t,Cn,label='Cn') # nutrient
    plt.plot(t,Cqc,label='Cq') # desired product
    plt.step(t,LI,label='LI') #light intensity
    plt.ylabel('some numbers')
    plt.xlabel('time')
    plt.legend()
    plt.show()

    return Cx,Cn,Cqc,LI

for i in range(n_iter): #loops through the steps taken by the simulated annealing algorithm
    Visualise = ProcessScore_Plot(Result[2][:,i]) #calls the plotting function defined within ProcesScore_Plot and it's running the integration model for Results[2][:,i] which calls upon the solution history/variables that the simulated annealing algorithm goes through

    
