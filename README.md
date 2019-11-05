# Design of Computational Optimization Algorithms
### About Me and Project Scope

Final-year MEng Chemical Engineering undergraduate at Imperial College London interested in optimization and machine learning. Working under the supervision of Dr Antonio Del Rio Chanona on this side-project to introduce myself to Python programming and further explore optimization, especially computational techniques. 

Objective is to have weekly project goals exploring new ideas in the field. 

## Week 1 

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

## Week 2 

The week 2 code file builds upon the computational optimization techniques explored in week 1 to: 
- Vectorise the gradient descent and simulated annealing algorithms to higher dimensions (i.e. extending them from taking an input of a 2d vector to being able to process a n-dimension vector/function) 
- Improve the gradient descent algorithm through introduction of "backtracking" - this involves adjusting the "alpha" (learning rate) in the algorithm such that the result from the next iteration is always better than the previous one 
- Visualise some analysis of the algorithms. For example, the figure on the left shows how the output of the simulated annealing algorithm behaves as a function of the number of iterations for a minimisation problem. And the figure to its right shows how a 2d search space is explored by a gradient descent algorithm.

![](Images/Simulated_annealing.png) ![](Images/gradient_descent_search_space.png)

The image on the left illustrates the above discussion that initially the simulated annealing algorithm will accept worse solutions to avoid getting stuck in local optima, however, later on it attempts to tighten towards a global optima. 

### Backtracking 

With a small enough learning rate (alpha) the objective function should monotonically decrease by following gradient descent, however, if the learning rate is too large the solution can diverge away from any optima, so if the updated evaluation of the function does not provide a better solution then we follow: 

<a href="https://www.codecogs.com/eqnedit.php?latex=f(X_{t&plus;1})&space;-&space;f(X_t)&space;>&space;0&space;\rightarrow&space;\alpha&space;=&space;\alpha/2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(X_{t&plus;1})&space;-&space;f(X_t)&space;>&space;0&space;\rightarrow&space;\alpha&space;=&space;\alpha/2" title="f(X_{t+1}) - f(X_t) > 0 \rightarrow \alpha = \alpha/2" /></a>

Alpha is halved until the updated solution is better than the original trial solution. The week 2 code file presented stores the history of "successful" alphas used and the starting trial value for alpha for the next iteration is taken as:

<a href="https://www.codecogs.com/eqnedit.php?latex=2&space;\cdot&space;\mu_{alpha\&space;history&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2&space;\cdot&space;\mu_{alpha\&space;history&space;}" title="2 \cdot \mu_{alpha\ history }" /></a> 

The times 2 ensuring alpha doesn't only decrease, as the trade-off of too small an alpha is that it can take too many iterations for the algorithm to reach the solution.


## Week 3

### Adam Optimisation

Implemented the "Adam" (adaptive moment estimation) algorithm - this is a combination of the gradient descent with momentum and RMSprop algorithms. (Additionally, backtracking has been included to appropriately adjust the learning rate, alpha). 

The key concepts behind Adam can be summarised in the following 3 equations: 

<a href="https://www.codecogs.com/eqnedit.php?latex=VdX&space;=&space;\frac{1}{1&space;-&space;\beta_1&space;^&space;{(t&plus;1)}}&space;(\beta_1&space;\cdot&space;VdX&space;&plus;&space;(1&space;-&space;\beta_1)\cdot&space;dX)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?VdX&space;=&space;\frac{1}{1&space;-&space;\beta_1&space;^&space;{(t&plus;1)}}&space;(\beta_1&space;\cdot&space;VdX&space;&plus;&space;(1&space;-&space;\beta_1)\cdot&space;dX)" title="VdX = \frac{1}{1 - \beta_1 ^ {(t+1)}} (\beta_1 \cdot VdX + (1 - \beta_1)\cdot dX)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=SdX&space;=&space;\beta_2&space;\cdot&space;SdX&space;&plus;&space;(1-\beta_2)&space;\cdot&space;dX^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SdX&space;=&space;\beta_2&space;\cdot&space;SdX&space;&plus;&space;(1-\beta_2)&space;\cdot&space;dX^2" title="SdX = \beta_2 \cdot SdX + (1-\beta_2) \cdot dX^2" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\{&space;X_{t&plus;1}&space;=&space;X_t&space;-&space;\alpha&space;(\frac{VdX}{\sqrt{SdX}&plus;\epsilon})\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{&space;X_{t&plus;1}&space;=&space;X_t&space;-&space;\alpha&space;(\frac{VdX}{\sqrt{SdX}&plus;\epsilon})\}" title="\{ X_{t+1} = X_t - \alpha (\frac{VdX}{\sqrt{SdX}+\epsilon})\}" /></a>

The 1st equation is an implementation of gradient descent with momentum - where exponetially weighted averaging is used. Then the 2nd equation is an implementation of the RMSprop algorithm. Finally equation 3 combines these two techniques into the gradient descent update step. The algorithm loops through these computations for n iterations. 

Adam shows significantly improved performance than a simple gradient descent algorithm (even with backtracking), as illustrated by convergence to the (1,1) minima on the Rosenbrock test function in the figure below: 

![](Images/iteration_path_contour_plot.png)

