# Design of Computational Optimisation Algorithms
### About Me and Project Scope

Final-year MEng Chemical Engineering undergraduate at Imperial College London working on this side-project under the supervision of Dr Antonio Del Rio Chanona during Oct-Dec 2019. The purpose of this project is to introduce myself to Python programming and explore various computational optimisation technqiues. The following algorithms have been coded: Simulated Annealing, Gradient Descent, Adam, BFGS and Dynamic Optimisation. My repository is broken down into different "milestones" representing the chronological order in which this work develops. Some basic background and discussion can be found in this ReadMe complementing the attached Python files. 


## Milestone 1  

Goal to code two local search algorithms in 2d: 
- Simulated annealing 
- Gradient descent 

And implement them on minimising two test-functions: 
- Simple sum-of-squares function (convex)
- Rosenbrock function (non-convex)

### Simulated Annealing

Randomly take N sample points normally distributed within a finite region of a "starting point": 

<a href="https://www.codecogs.com/eqnedit.php?latex=x_s&space;\sim&space;N(x_0,\sigma)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_s&space;\sim&space;N(x_0,\sigma)" title="x_s \sim N(x_0,\sigma)" /></a>

Evaluate the function at all these sample points and select the minimum. If the minimum is lower than the function evaluation at the starting point, then update this best sample point to be the new starting point for the next iteration of the algorithm.

If not then compute:

<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;=&space;\exp(\frac{-c}{T})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P&space;=&space;\exp(\frac{-c}{T})" title="P = \exp(\frac{-c}{T})" /></a>

where c is the change in the evaluation function and T is the current temperature. If P > r (some set criteria between 0 and  1) then accept the sample point as the new starting point, even if it is worse than the original starting point/trial solution. 

Additionally, the temperature is a function of the number of iterations - it starts high and decreases - in my code according to: 

<a href="https://www.codecogs.com/eqnedit.php?latex=T&space;=&space;\beta&space;T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T&space;=&space;\beta&space;T" title="T = \beta T" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /></a> is a "cooling rate" which reduces the temperature like a geometric progression. The result of this is that at the beginning the algorithm has a high chance of accepting a 'worse' solution to avoid getting stuck in local optima, however, the temperature needs to lower with increasing iterations so that the algorithm tightens towards only accepting better solutions. 

### Gradient Descent 

The gradient descent algorithm works on the principle that the gradient of a scalar-function is a vector field that gives the greatest rate of change; therefore, to traverse towards the minima we wish to follow the updates on each iteration of: 

<a href="https://www.codecogs.com/eqnedit.php?latex=X_{t&plus;1}&space;=&space;X_t&space;-&space;\frac{\partial}{\partial&space;X}&space;J(X)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{t&plus;1}&space;=&space;X_t&space;-&space;\frac{\partial}{\partial&space;X}&space;J(X)" title="X_{t+1} = X_t - \frac{\partial}{\partial X} J(X)" /></a>

Here alpha is the "learning rate" and the negative sign shows us we are travelling towards the minimum along the steepest path of gradient descent. 

## Milestone 2 

The milestone 2 code file builds upon the computational optimization techniques explored in milestone 1 to: 
- Vectorise the gradient descent and simulated annealing algorithms to higher dimensions (i.e. extending them from taking an input of a 2d vector to being able to process a n-dimension vector/function) 
- Improve the gradient descent algorithm through introduction of "backtracking" - this involves adjusting the "alpha" (learning rate) in the algorithm such that the result from the next iteration is always better than the previous one 
- Visualise some analysis of the algorithms. For example, the figure on the left shows how the output of the simulated annealing algorithm behaves as a function of the number of iterations for a minimisation problem. And the figure to its right shows how a 2d search space is explored by a gradient descent algorithm.

![](Images/Simulated_annealing.png) ![](Images/gradient_descent_search_space.png)

The image on the left illustrates the above discussion that initially the simulated annealing algorithm will accept worse solutions to avoid getting stuck in local optima, however, later on it attempts to tighten towards a global optima. 

### Backtracking 

With a small enough learning rate (alpha) the objective function should monotonically decrease by following gradient descent, however, if the learning rate is too large the solution can diverge away from any optima, so if the updated evaluation of the function does not provide a better solution then we follow: 

<a href="https://www.codecogs.com/eqnedit.php?latex=f(X_{t&plus;1})&space;-&space;f(X_t)&space;>&space;0&space;\rightarrow&space;\alpha&space;=&space;\alpha/2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(X_{t&plus;1})&space;-&space;f(X_t)&space;>&space;0&space;\rightarrow&space;\alpha&space;=&space;\alpha/2" title="f(X_{t+1}) - f(X_t) > 0 \rightarrow \alpha = \alpha/2" /></a>

Alpha is halved until the updated solution is better than the original trial solution. The milestone 2 code file presented stores the history of "successful" alphas used and the starting trial value for alpha for the next iteration is taken as:

<a href="https://www.codecogs.com/eqnedit.php?latex=2&space;\cdot&space;\mu_{alpha\&space;history&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2&space;\cdot&space;\mu_{alpha\&space;history&space;}" title="2 \cdot \mu_{alpha\ history }" /></a> 

The times 2 ensuring alpha doesn't only decrease, as the trade-off of too small an alpha is that it can take too many iterations for the algorithm to reach the solution.


## Milestone 3 

### Adam Optimisation

Implemented the "Adam" (adaptive moment estimation) algorithm - this is a combination of the gradient descent with momentum and RMSprop algorithms. (Additionally, backtracking has been included.)

The key concepts behind Adam can be summarised in the following 3 equations: 

<a href="https://www.codecogs.com/eqnedit.php?latex=VdX&space;=&space;\frac{1}{1&space;-&space;\beta_1&space;^&space;{(t&plus;1)}}&space;(\beta_1&space;\cdot&space;VdX&space;&plus;&space;(1&space;-&space;\beta_1)\cdot&space;dX)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?VdX&space;=&space;\frac{1}{1&space;-&space;\beta_1&space;^&space;{(t&plus;1)}}&space;(\beta_1&space;\cdot&space;VdX&space;&plus;&space;(1&space;-&space;\beta_1)\cdot&space;dX)" title="VdX = \frac{1}{1 - \beta_1 ^ {(t+1)}} (\beta_1 \cdot VdX + (1 - \beta_1)\cdot dX)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=SdX&space;=&space;\beta_2&space;\cdot&space;SdX&space;&plus;&space;(1-\beta_2)&space;\cdot&space;dX^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SdX&space;=&space;\beta_2&space;\cdot&space;SdX&space;&plus;&space;(1-\beta_2)&space;\cdot&space;dX^2" title="SdX = \beta_2 \cdot SdX + (1-\beta_2) \cdot dX^2" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\{&space;X_{t&plus;1}&space;=&space;X_t&space;-&space;\alpha&space;(\frac{VdX}{\sqrt{SdX}&plus;\epsilon})\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{&space;X_{t&plus;1}&space;=&space;X_t&space;-&space;\alpha&space;(\frac{VdX}{\sqrt{SdX}&plus;\epsilon})\}" title="\{ X_{t+1} = X_t - \alpha (\frac{VdX}{\sqrt{SdX}+\epsilon})\}" /></a>

The 1st equation is an implementation of gradient descent with momentum - where exponetially weighted averaging is used. Then the 2nd equation is an implementation of the RMSprop algorithm. Finally equation 3 combines these two techniques into the gradient descent update step. The algorithm loops through these computations for n iterations. 

Adam shows significantly improved performance than a simple gradient descent algorithm (even with backtracking), as illustrated by convergence to the (1,1) minima on the Rosenbrock test function in the figure below: 

![](Images/iteration_path_contour_plot.png)

## Milestone 4 

### BFGS Algorithm 

Implemented the BFGS algorithm - this is a quasi-Newton method. Along the line of algorithms coded in the previous weeks (gradient descent and the Adam algorithm) the BFGS method also employs a line search strategy. This means the algorithm chooses a direction to search along to find a new iterate with a lower objective function value. 

The BFGS algorithm is a development on Newton's method which can be summarised with the equation: 

<a href="https://www.codecogs.com/eqnedit.php?latex=X_{k&plus;1}=X_{k}&space;-&space;\nabla_{xx}^2(X_k)\nabla&space;f(X_k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{k&plus;1}=X_{k}&space;-&space;\nabla_{xx}^2(X_k)\nabla&space;f(X_k)" title="X_{k+1}=X_{k} - \nabla_{xx}^2(X_k)\nabla f(X_k)" /></a>

A significant drawback, however, of Newton's method is that it is computationally intesive - both computing the Hessian and inverting that result is computationally expensive. The BFGS algorithm implemented is more efficient as it avoids both of these operations. The BFGS approach uses the structure: 

<a href="https://www.codecogs.com/eqnedit.php?latex=X_{k&plus;1}=X_{k}&space;-&space;B_k^{-1}\nabla&space;f(X_k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{k&plus;1}=X_{k}&space;-&space;B_k^{-1}\nabla&space;f(X_k)" title="X_{k+1}=X_{k} - B_k^{-1}\nabla f(X_k)" /></a>

where the BFGS formula for updating the Hessian approaximation is: 

<a href="https://www.codecogs.com/eqnedit.php?latex=B_{k&plus;1}&space;=&space;B_k&space;-&space;\frac{B_{k}S_{k}S_{k}^TB_{k}}{S_{k}^TB_kS_k}&space;&plus;&space;\frac{Y_{k}Y_{k}^T}{Y_{k}^TS_k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B_{k&plus;1}&space;=&space;B_k&space;-&space;\frac{B_{k}S_{k}S_{k}^TB_{k}}{S_{k}^TB_kS_k}&space;&plus;&space;\frac{Y_{k}Y_{k}^T}{Y_{k}^TS_k}" title="B_{k+1} = B_k - \frac{B_{k}S_{k}S_{k}^TB_{k}}{S_{k}^TB_kS_k} + \frac{Y_{k}Y_{k}^T}{Y_{k}^TS_k}" /></a>

with: 

<a href="https://www.codecogs.com/eqnedit.php?latex=S_k&space;=&space;X_{k&plus;1}&space;-&space;X_k&space;\&space;and&space;\&space;Y_k&space;=&space;\nabla&space;f_{k&plus;1}&space;-&space;\nabla&space;f_k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S_k&space;=&space;X_{k&plus;1}&space;-&space;X_k&space;\&space;and&space;\&space;Y_k&space;=&space;\nabla&space;f_{k&plus;1}&space;-&space;\nabla&space;f_k" title="S_k = X_{k+1} - X_k \ and \ Y_k = \nabla f_{k+1} - \nabla f_k" /></a>

However, this would still require inverting a matrix so the BFGS algorithm proposes the formulation:

<a href="https://www.codecogs.com/eqnedit.php?latex=let&space;\&space;\&space;B_k^{-1}&space;=&space;H_k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?let&space;\&space;\&space;B_k^{-1}&space;=&space;H_k" title="let \ \ B_k^{-1} = H_k" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\rho_k&space;=&space;\frac{1}{Y_k^TS_k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\rho_k&space;=&space;\frac{1}{Y_k^TS_k}" title="\rho_k = \frac{1}{Y_k^TS_k}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=H_{k&plus;1}&space;=&space;(I&space;-&space;\rho_kS_kY_k^T)H_k(I&space;-&space;\rho_kY_kS_k^T)&space;&plus;&space;\rho_kS_kS_k^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H_{k&plus;1}&space;=&space;(I&space;-&space;\rho_kS_kY_k^T)H_k(I&space;-&space;\rho_kY_kS_k^T)&space;&plus;&space;\rho_kS_kS_k^T" title="H_{k+1} = (I - \rho_kS_kY_k^T)H_k(I - \rho_kY_kS_k^T) + \rho_kS_kS_k^T" /></a>

These equations are implemented in the milestone 4 code. To actually implement this algorithm one needs an initial approximation for <a href="https://www.codecogs.com/eqnedit.php?latex=H_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H_0" title="H_0" /></a>. I have used a basic gradient descent step for the first 2 iterations as a "warming up phase" before transitioning into the BFGS algorithm. 

The below plots show the performance of this algorithm vs the Adam algorithm tested on the Rosenbrock function, which has a global minima evaluating to 0. The left figure represents how the objective function is minimised for a 2d test function, with starting point X0 = (4,4) and the right one is an extension to higher dimensions (4d), with a starting point X0 = (2,2,2,2)

![](Images/BFGSvsAdam_2d_X0=(4,4).png) ![](Images/BFGSvsAdam_4d_X0=(2,2,2,2).png)


## Milestone 5 

![](Images/BioprocessDynamicOptimisation.GIF)
